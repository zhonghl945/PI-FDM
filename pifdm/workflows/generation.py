"""Shared dataset-generation workflow.

Case-specific, human-editable settings live in ``case_studies/*/generate.py``.
This module only executes and validates those settings.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

from pifdm.cases import get_case_adapter


def generate_dataset(config: dict, *, split: str, output: str | Path,
                     visualization_dir: str | Path | None = None,
                     visualization_samples: int = 0):
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    effective = deepcopy(config)
    if split == "test":
        effective["generation"] = {
            **effective.get("generation", {}),
            **effective.get("test_generation", {}),
        }
    worker_override = os.environ.get("PIFDM_GENERATION_WORKERS")
    if worker_override is not None:
        workers = int(worker_override)
        if workers < 1:
            raise ValueError("PIFDM_GENERATION_WORKERS must be at least 1")
        effective.setdefault("generation", {})["workers"] = workers
    effective.setdefault("generation", {})["progress_label"] = f"{effective['case']} {split}"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    adapter = get_case_adapter(effective["case"])
    generated = adapter.generate(effective, output)
    data = adapter.load(generated)
    adapter.validate(data)
    figure_paths = []
    if visualization_samples:
        if visualization_samples < 0:
            raise ValueError("visualization_samples must be non-negative")
        from pifdm.evaluation.visualization import plot_dataset_sample
        figure_dir = Path(visualization_dir or output.parent / "visualizations" / split)
        for index in range(min(int(visualization_samples), len(data.fields))):
            figure_path = figure_dir / f"sample_{index:04d}.png"
            plot_dataset_sample(data, index, figure_path,
                                title=f"{adapter.name} {split} reference solution")
            figure_paths.append(str(figure_path.resolve()))
    result = {
        "case": adapter.name,
        "split": split,
        "output": str(generated.resolve()),
        "shape": list(data.fields.shape),
        "samples": int(data.fields.shape[0]),
        "visualizations": figure_paths,
    }
    print(json.dumps(result, ensure_ascii=False))
    return generated
