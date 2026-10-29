"""Optional numerical acceleration and reproducible sample parallelism."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np


_WORKER_FUNCTION = None
_WORKER_CONTEXT = None
_WORKER_LABEL = "solver"
_WORKER_TOTAL = 0
_WORKER_SAMPLE_INDEX = None
_WORKER_PROGRESS_INTERVAL = 30.0
_WORKER_PROGRESS_ENABLED = False
_WORKER_PROGRESS_LAST = 0.0


try:
    from numba import njit as _numba_njit
except Exception:  # Numba is optional for the minimal data-reading install.
    _numba_njit = None


def njit(function=None, **options):
    """Use cached Numba compilation when available, otherwise keep Python."""
    def decorate(target):
        if _numba_njit is None:
            return target
        return _numba_njit(cache=True, **options)(target)

    return decorate(function) if function is not None else decorate


def independent_seeds(seed: int, samples: int) -> list[int]:
    """Create per-sample seeds independent of process count and scheduling."""
    sequence = np.random.SeedSequence(int(seed))
    return [int(child.generate_state(1, dtype=np.uint64)[0]) for child in sequence.spawn(samples)]


def _set_worker_progress(*, label: str, total: int, interval: float,
                         enabled: bool, sample_index: int | None = None) -> None:
    global _WORKER_LABEL, _WORKER_TOTAL, _WORKER_SAMPLE_INDEX
    global _WORKER_PROGRESS_INTERVAL, _WORKER_PROGRESS_ENABLED
    global _WORKER_PROGRESS_LAST
    _WORKER_LABEL = str(label)
    _WORKER_TOTAL = int(total)
    _WORKER_SAMPLE_INDEX = sample_index
    _WORKER_PROGRESS_INTERVAL = float(interval)
    _WORKER_PROGRESS_ENABLED = bool(enabled)
    _WORKER_PROGRESS_LAST = 0.0


def report_solver_progress(step: int, total_steps: int, *, phase: str = "time-step") -> None:
    """Print throttled progress for the solver currently running in a worker."""
    global _WORKER_PROGRESS_LAST
    if not _WORKER_PROGRESS_ENABLED or _WORKER_SAMPLE_INDEX is None:
        return
    now = time.monotonic()
    step = int(step)
    total_steps = max(1, int(total_steps))
    if step < total_steps and now - _WORKER_PROGRESS_LAST < _WORKER_PROGRESS_INTERVAL:
        return
    percent = 100.0 * min(max(step, 0), total_steps) / total_steps
    print(
        f"{_WORKER_LABEL}: sample={_WORKER_SAMPLE_INDEX}/{_WORKER_TOTAL} "
        f"{phase}={step}/{total_steps} ({percent:.1f}%)",
        file=sys.stdout, flush=True,
    )
    _WORKER_PROGRESS_LAST = now


def _initialize_worker(function, context, label, total, interval, enabled):
    global _WORKER_FUNCTION, _WORKER_CONTEXT
    _WORKER_FUNCTION, _WORKER_CONTEXT = function, context
    _set_worker_progress(label=label, total=total, interval=interval, enabled=enabled)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"


def _run_worker(task):
    index, seed = task
    _set_worker_progress(
        label=_WORKER_LABEL, total=_WORKER_TOTAL, interval=_WORKER_PROGRESS_INTERVAL,
        enabled=_WORKER_PROGRESS_ENABLED, sample_index=index + 1,
    )
    return _WORKER_FUNCTION(_WORKER_CONTEXT, seed)


def parallel_samples(function: Callable, context: dict, seeds: list[int], workers: int,
                     *, description: str = "Generating samples", show_progress: bool = True,
                     progress_interval: float = 30.0):
    """Evaluate independent samples in stable input order using worker processes."""
    from tqdm.auto import tqdm

    workers = int(workers)
    if workers < 1:
        raise ValueError("generation.workers must be at least 1")
    progress_interval = float(progress_interval)
    if progress_interval <= 0.0:
        raise ValueError("generation.progress_interval must be positive")

    total = len(seeds)
    reported_workers = min(workers, total) if total else 0
    interactive = bool(show_progress and sys.stdout.isatty())
    started = time.monotonic()
    last_report = started
    last_completion = started

    def format_duration(seconds: float) -> str:
        seconds = max(int(seconds), 0)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def report(completed: int, active: int, *, force: bool = False) -> None:
        nonlocal last_report
        if not show_progress or interactive:
            return
        now = time.monotonic()
        if not force and now - last_report < progress_interval:
            return
        elapsed = now - started
        rate = completed / elapsed if elapsed > 0.0 else 0.0
        eta = (total - completed) / rate if rate > 0.0 else float("inf")
        eta_text = format_duration(eta) if np.isfinite(eta) else "unknown"
        percent = 100.0 * completed / total if total else 100.0
        idle = now - last_completion
        print(
            f"{description}: completed={completed}/{total} ({percent:.1f}%) "
            f"elapsed={format_duration(elapsed)} eta={eta_text} rate={rate:.3f} sample/s "
            f"workers={reported_workers} active={active} no_completion={format_duration(idle)}",
            file=sys.stdout, flush=True,
        )
        last_report = now

    if workers == 1 or len(seeds) <= 1:
        progress = tqdm(
            total=total, desc=description, unit="sample", dynamic_ncols=True,
            mininterval=0.5, disable=not interactive, file=sys.stdout,
        )
        results = []
        report(0, min(total, 1), force=True)
        try:
            for index, seed in enumerate(seeds):
                _set_worker_progress(
                    label=description, total=total, interval=progress_interval,
                    enabled=show_progress and not interactive, sample_index=index + 1,
                )
                try:
                    results.append(function(context, seed))
                except BaseException as error:
                    print(
                        f"{description}: sample {index + 1}/{total} failed "
                        f"(seed={seed}, {type(error).__name__}: {error}).",
                        file=sys.stderr, flush=True,
                    )
                    raise RuntimeError(
                        f"{description}: sample {index + 1}/{total} failed (seed={seed})"
                    ) from error
                last_completion = time.monotonic()
                progress.update(1)
                report(len(results), int(len(results) < total))
        finally:
            progress.close()
        report(total, 0, force=True)
        return results
    # Each process receives the read-only context once, then tasks only carry a
    # small seed. On Linux, fork also shares these arrays copy-on-write. Worker
    # initialisation restricts nested BLAS/OpenMP pools to one thread. Keep only
    # one future per worker so a failed sample does not make shutdown drain the
    # entire dataset before surfacing its exception.
    effective_workers = min(workers, len(seeds))
    executor = ProcessPoolExecutor(
        max_workers=effective_workers, initializer=_initialize_worker,
        initargs=(function, context, description, total, progress_interval,
                  show_progress and not interactive),
    )
    results = [None] * total
    pending = {}
    next_index = 0
    progress = tqdm(
        total=total, desc=description, unit="sample", dynamic_ncols=True,
        mininterval=0.5, disable=not interactive, file=sys.stdout,
    )

    def submit_one() -> None:
        nonlocal next_index
        if next_index < total:
            seed = seeds[next_index]
            pending[executor.submit(_run_worker, (next_index, seed))] = (next_index, seed)
            next_index += 1

    for _ in range(effective_workers):
        submit_one()
    completed = 0
    report(completed, len(pending), force=True)
    try:
        while pending:
            done, _ = wait(
                tuple(pending), timeout=progress_interval, return_when=FIRST_COMPLETED,
            )
            if not done:
                progress.refresh()
                report(completed, len(pending), force=True)
                continue
            for future in done:
                index, seed = pending.pop(future)
                try:
                    results[index] = future.result()
                except BaseException as error:
                    print(
                        f"{description}: sample {index + 1}/{total} failed "
                        f"(seed={seed}, {type(error).__name__}: {error}); "
                        f"cancelling remaining work.",
                        file=sys.stderr, flush=True,
                    )
                    raise RuntimeError(
                        f"{description}: sample {index + 1}/{total} failed (seed={seed})"
                    ) from error
                completed += 1
                last_completion = time.monotonic()
                progress.update(1)
                submit_one()
            report(completed, len(pending))
    except BaseException:
        for future in pending:
            future.cancel()
        raise
    finally:
        progress.close()
        executor.shutdown(wait=True)
    report(completed, 0, force=True)
    return results
