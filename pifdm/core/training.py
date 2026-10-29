"""Training utilities with cluster-ready numerical features."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch


@dataclass
class PhysicsSchedule:
    weight: float = 0.0
    start_step: int = 0
    warmup_steps: int = 0

    def at(self, step: int) -> float:
        if step < self.start_step:
            return 0.0
        if self.warmup_steps <= 0:
            return self.weight
        return self.weight * min(1.0, (step - self.start_step + 1) / self.warmup_steps)


class ExponentialMovingAverage:
    def __init__(self, model, decay=0.995, update_every=10):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = float(decay)
        self.update_every = int(update_every)
        self.updates = 0

    @torch.no_grad()
    def update(self, source):
        self.updates += 1
        if self.updates % self.update_every:
            return
        source_parameters = dict(source.named_parameters())
        for name, parameter in self.model.named_parameters():
            parameter.lerp_(source_parameters[name].detach(), 1.0 - self.decay)
        source_buffers = dict(source.named_buffers())
        for name, buffer in self.model.named_buffers():
            buffer.copy_(source_buffers[name])

    def state_dict(self):
        return {"model": self.model.state_dict(), "updates": self.updates,
                "decay": self.decay, "update_every": self.update_every}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.updates = int(state.get("updates", 0))

    def to(self, device):
        self.model.to(device)
        return self


class Trainer:
    """One-step trainer supporting AMP, accumulation, scheduling and EMA."""
    def __init__(self, model, optimizer, *, schedule=None, boundary_schedule=None,
                 lr_scheduler=None, grad_clip=1.0,
                 gradient_accumulation_steps=1, mixed_precision=False, ema_decay=None, ema_update_every=10):
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule or PhysicsSchedule()
        self.boundary_schedule = boundary_schedule or PhysicsSchedule()
        self.lr_scheduler = lr_scheduler
        self.grad_clip = float(grad_clip)
        self.accumulation = max(1, int(gradient_accumulation_steps))
        self.mixed_precision = bool(mixed_precision)
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision and torch.cuda.is_available())
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision and torch.cuda.is_available())
        self.ema = (ExponentialMovingAverage(model, ema_decay, ema_update_every)
                    if ema_decay is not None and float(ema_decay) > 0 else None)
        self.optimizer.zero_grad(set_to_none=True)

    def step(self, batch: torch.Tensor, step: int, *, final_step: bool = False,
             physics_step: int | None = None, **kwargs):
        weight = self.schedule.at(step if physics_step is None else physics_step)
        boundary_weight = self.boundary_schedule.at(
            step if physics_step is None else physics_step
        )
        # The collocation objective locally disables autocast for coordinate
        # derivatives.  Keep the much larger neural/data forward under AMP,
        # including on PI steps, when mixed precision was requested.
        reference = batch["field"] if isinstance(batch, dict) else batch
        use_amp = self.mixed_precision and reference.device.type == "cuda"
        with torch.autocast(device_type=reference.device.type, dtype=torch.float16, enabled=use_amp):
            losses = self.model(
                batch, lambda_phy=weight, lambda_bc=boundary_weight, **kwargs
            )
            scaled_loss = losses["total_loss"] / self.accumulation
        self.scaler.scale(scaled_loss).backward()
        if (step + 1) % self.accumulation == 0 or final_step:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # GradScaler silently skips optimizer.step() when gradients
            # overflow.  The scheduler and EMA must not advance in that case.
            optimizer_stepped = (not self.scaler.is_enabled()
                                 or self.scaler.get_scale() >= scale_before)
            self.optimizer.zero_grad(set_to_none=True)
            if optimizer_stepped:
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.model)
        return losses
