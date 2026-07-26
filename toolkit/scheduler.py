import torch
from typing import List, Optional, Sequence
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION, get_constant_schedule_with_warmup


def _parse_step_targets(targets: Sequence) -> List[tuple]:
    # normalize to a sorted list of (step, lr) tuples
    if targets is None:
        raise ValueError(
            "lr_scheduler_params must contain a 'targets' key for the step_targets scheduler."
        )
    if len(targets) == 0:
        raise ValueError("step_targets 'targets' must contain at least one [step, lr] pair.")

    parsed = []
    for target in targets:
        if len(target) != 2:
            raise ValueError(
                f"step_targets 'targets' entries must be [step, lr] pairs, got {target}"
            )
        parsed.append((int(target[0]), float(target[1])))

    return sorted(parsed, key=lambda pair: pair[0])


def _build_step_targets_lambda(targets: List[tuple], base_lr: float):
    # LambdaLR multiplies each param group's initial_lr by the factor we return,
    # so convert the absolute target lr into a factor against base_lr.
    def lr_lambda(current_step: int) -> float:
        if current_step <= targets[0][0]:
            return targets[0][1] / base_lr
        if current_step >= targets[-1][0]:
            return targets[-1][1] / base_lr

        for i in range(len(targets) - 1):
            start_step, start_lr = targets[i]
            end_step, end_lr = targets[i + 1]
            if start_step <= current_step < end_step:
                span = end_step - start_step
                if span <= 0:
                    # duplicate steps, nothing to interpolate over
                    return end_lr / base_lr
                progress = (current_step - start_step) / span
                return (start_lr + progress * (end_lr - start_lr)) / base_lr

        return targets[-1][1] / base_lr

    return lr_lambda


def get_lr_scheduler(
        name: Optional[str],
        optimizer: torch.optim.Optimizer,
        **kwargs,
):
    # Piecewise linear scheduler driven by explicit (step, lr) targets.
    # The lr is linearly interpolated between consecutive targets and held
    # constant before the first / after the last one.
    #
    #   lr_scheduler: "step_targets"
    #   lr_scheduler_params:
    #     targets: [[0, 1e-10], [1000, 1e-3], [30000, 1e-3], [40000, 1e-5]]
    #
    # A single curve is applied to every param group. The absolute lrs above are
    # hit by the group with the largest base lr (the transformer); any other
    # group is scaled by the same factor, preserving its lr ratio.
    if name == "step_targets":
        targets = _parse_step_targets(kwargs.get('targets', None))

        base_lr = max(group.get("initial_lr", group["lr"]) for group in optimizer.param_groups)
        if base_lr <= 0:
            raise ValueError(
                f"step_targets needs a positive base learning rate to scale against, got {base_lr}"
            )

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, _build_step_targets_lambda(targets, base_lr)
        )

    if name == "cosine":
        if 'total_iters' in kwargs:
            kwargs['T_max'] = kwargs.pop('total_iters')
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **kwargs
        )
    elif name == "cosine_with_restarts":
        if 'total_iters' in kwargs:
            kwargs['T_0'] = kwargs.pop('total_iters')
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, **kwargs
        )
    elif name == "step":

        return torch.optim.lr_scheduler.StepLR(
            optimizer, **kwargs
        )
    elif name == "constant":
        if 'factor' not in kwargs:
            kwargs['factor'] = 1.0

        return torch.optim.lr_scheduler.ConstantLR(optimizer, **kwargs)
    elif name == "linear":

        return torch.optim.lr_scheduler.LinearLR(
            optimizer, **kwargs
        )
    elif name == 'constant_with_warmup':
        # see if num_warmup_steps is in kwargs
        if 'num_warmup_steps' not in kwargs:
            print(f"WARNING: num_warmup_steps not in kwargs. Using default value of 1000")
            kwargs['num_warmup_steps'] = 1000
        del kwargs['total_iters']
        return get_constant_schedule_with_warmup(optimizer, **kwargs)
    else:
        # try to use a diffusers scheduler
        print(f"Trying to use diffusers scheduler {name}")
        try:
            name = SchedulerType(name)
            schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
            return schedule_func(optimizer, **kwargs)
        except Exception as e:
            print(e)
            pass
        raise ValueError(
            "Scheduler must be cosine, cosine_with_restarts, step, linear or constant"
        )
