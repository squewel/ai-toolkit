import torch
from typing import Optional
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION, get_constant_schedule_with_warmup


def get_lr_scheduler(
        name: Optional[str],
        optimizer: torch.optim.Optimizer,
        **kwargs,
):

    # --- Custom Piecewise Linear Scheduler ---
        # scheduler config
        # self.train_config.lr_scheduler = "step_targets"
        # # "targets" is the key we will look for in the kwargs
        # self.train_config.lr_scheduler_params = {
        #     "targets": [[0, 1e-10], [1000, 1e-4], [2000, 1e-5]]
        # }
    if name == "step_targets":
        targets = kwargs.get('targets', None)
        if targets is None:
            raise ValueError("lr_scheduler_params must contain a 'targets' key for step_targets scheduler.")
        
        # Sort targets by step to ensure correct interpolation order
        # targets structure: [[step, lr], [step, lr], ...]
        targets = sorted(targets, key=lambda x: x[0])
        
        # We need the optimizer's initial LR to calculate the Lambda factor.
        # LambdaLR does: lr = initial_lr * factor. 
        # So: factor = desired_lr / initial_lr.
        
        # Attempt to get initial_lr, defaulting to current lr if not yet set
        base_lr = optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"])
        
        def lr_lambda(current_step):
            # 1. Before the first defined step, use the first target's LR
            if current_step < targets[0][0]:
                return targets[0][1] / base_lr
            
            # 2. After the last defined step, use the last target's LR
            if current_step >= targets[-1][0]:
                return targets[-1][1] / base_lr
            
            # 3. Find the interval [t_start, t_end] where current_step lies
            for i in range(len(targets) - 1):
                start_step, start_lr = targets[i]
                end_step, end_lr = targets[i+1]
                
                if start_step <= current_step < end_step:
                    # Calculate progress (0.0 to 1.0) within this interval
                    progress = (current_step - start_step) / (end_step - start_step)
                    # Linear interpolation
                    target_lr = start_lr + progress * (end_lr - start_lr)
                    return target_lr / base_lr
            
            return targets[-1][1] / base_lr

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Existing Schedulers ---
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
