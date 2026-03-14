import torch
from typing import Optional
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION, get_constant_schedule_with_warmup


def get_lr_scheduler(
        name: Optional[str],
        optimizer: torch.optim.Optimizer,
        **kwargs,
):

    # --- Custom Piecewise Linear Scheduler ---
        # lr_scheduler: "step_targets"
        # lr_scheduler_params:
        #   targets: [[0, 1e-10], [1000, 1e-3], [30000, 1e-3], [40000, 1e-5]]
        #   te_targets: [[0, 1e-11], [500, 1e-4],[15000, 1e-4], [25000, 1e-5], [40000, 1e-6]] # optional, if off and TE training on = sale by text_encoder_lr/lr. text_encoder_lr needs to be < lr 
    if name == "step_targets":
        targets = kwargs.get('targets', None)
        te_targets = kwargs.get('te_targets', None) # Optional separate TE targets
        
        if targets is None:
            raise ValueError("lr_scheduler_params must contain a 'targets' key for step_targets scheduler.")
        
        # Sort targets by step to ensure correct interpolation order
        targets = sorted(targets, key=lambda x: x[0])
        if te_targets is not None:
            te_targets = sorted(te_targets, key=lambda x: x[0])
            
        # Identify the main base LR (this will safely find the 1e-3 from your DiT)
        main_base_lr = max(group.get("initial_lr", group["lr"]) for group in optimizer.param_groups)
        
        # Factory function to create an isolated lambda for a specific target curve
        def create_lambda(target_list, base_lr_for_calc):
            def lr_lambda(current_step):
                # 1. Before the first defined step
                if current_step < target_list[0][0]:
                    return target_list[0][1] / base_lr_for_calc
                
                # 2. After the last defined step
                if current_step >= target_list[-1][0]:
                    return target_list[-1][1] / base_lr_for_calc
                
                # 3. Find the interval and interpolate
                for i in range(len(target_list) - 1):
                    start_step, start_lr = target_list[i]
                    end_step, end_lr = target_list[i+1]
                    
                    if start_step <= current_step < end_step:
                        progress = (current_step - start_step) / (end_step - start_step)
                        target_lr = start_lr + progress * (end_lr - start_lr)
                        return target_lr / base_lr_for_calc
                
                return target_list[-1][1] / base_lr_for_calc
            return lr_lambda

        # If te_targets is provided, we assign separate curves
        if te_targets is not None:
            lambdas =[]
            for group in optimizer.param_groups:
                group_base_lr = group.get("initial_lr", group["lr"])
                
                # Safest heuristic: If this group's initial LR is smaller than the DiT's (e.g., 1e-4 < 1e-3),
                # it is the Text Encoder. Assign it the te_targets.
                if group_base_lr < main_base_lr:
                    lambdas.append(create_lambda(te_targets, group_base_lr))
                else:
                    # Otherwise, it's the DiT (or another main component), give it the standard targets.
                    lambdas.append(create_lambda(targets, group_base_lr))
                    
            # PyTorch accepts a list of lambdas, mapping them 1:1 to the parameter groups
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
        
        else:
            # Fallback (Original working behavior)
            # We use the DiT's base LR to calculate the factor.
            # PyTorch applies this single factor to all groups, perfectly preserving your ratio.
            single_lambda = create_lambda(targets, main_base_lr)
            return torch.optim.lr_scheduler.LambdaLR(optimizer, single_lambda)

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
