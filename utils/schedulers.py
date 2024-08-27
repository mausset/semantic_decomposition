# Yoinked from v-jepa repository on GitHub

import math

from torch.optim.lr_scheduler import LRScheduler


class LinearSchedule(object):

    def __init__(self, start, end, duration):
        self.start = start
        self.end = end
        self.duration = duration

    def __call__(self, t):
        if self.start > self.end:
            return max(
                self.start + (self.end - self.start) * t / self.duration, self.end
            )
        return min(self.start + (self.end - self.start) * t / self.duration, self.end)


class WarmupCosineSchedule(LRScheduler):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.0,
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step_count = 0.0
        self.last_epoch = -1

        super().__init__(optimizer, last_epoch)

    def get_lr(self):  # type: ignore
        if self._step_count < self.warmup_steps:
            progress = float(self._step_count) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step_count - self.warmup_steps) / float(
                max(1, self.T_max)
            )
            new_lr = max(
                self.final_lr,
                self.final_lr
                + (self.ref_lr - self.final_lr)
                * 0.5
                * (1.0 + math.cos(math.pi * progress)),
            )

        return [new_lr for _ in self.optimizer.param_groups]


class CosineWDSchedule(object):

    def __init__(self, optimizer, ref_wd, T_max, final_wd=0.0):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = new_wd
        return new_wd
