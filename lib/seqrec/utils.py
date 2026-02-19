import os
import math
import tqdm
import time
from collections import defaultdict
       
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def validate(model, dataloader, steps=None, verbose=False):
    steps = steps or len(dataloader)

    torch.cuda.synchronize()
    t0 = time.time() 
    dataloader = iter(dataloader)
    model.eval()
    batch = next(dataloader)
    accumulated_metrics = defaultdict(int)
    with torch.inference_mode(True):
        for step in tqdm.tqdm(range(steps), total=steps, disable=not verbose):
            with torch.autocast('cuda', torch.bfloat16):
                _, metrics = model(batch, with_metrics=True)
            if step < steps - 1:
                batch = next(dataloader)
            for name, value in metrics.items():
                accumulated_metrics[name] += (value.detach().cpu() - accumulated_metrics[name]) / (step + 1)    
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    model.train()
    accumulated_metrics['time(s)'] = dt
    return accumulated_metrics


def train_loop(
        graph, 
        train_dataloader, 
        log_dir,
        step_optimizers_func, 
        num_iterations,
        grad_accum_steps,
        grad_clip=0.,
        valid_dataloaders=None, 
        eval_every=-1,
        custom_logging=None,
        checkpoint_dir=None,
        checkpoint_every=None,
        custom_validation=None
):
    graph.train()

    writer = SummaryWriter(log_dir=log_dir)
        
    train_dataloader_iterator = iter(train_dataloader)
    batch = next(train_dataloader_iterator)

    if checkpoint_every is not None:
        assert checkpoint_dir is not None
    if checkpoint_dir is not None and not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    tokens_passed = 0
    for step in tqdm.tqdm(range(num_iterations - 1), total=num_iterations - 1):
        last_step = step == num_iterations - 2
        if eval_every != -1 and (last_step or step % eval_every == 0):
            assert valid_dataloaders is not None or custom_validation is not None
            graph.eval()
            if valid_dataloaders is not None:
                for name, loader in valid_dataloaders.items():
                    metrics = validate(graph, loader)
                    for metric, value in metrics.items():
                        writer.add_scalar(f'{name}/{metric}', value, tokens_passed)
            if custom_validation is not None:
                for name, value in custom_validation().items():
                    writer.add_scalar(f'custom/{name}', value, tokens_passed)
                
            graph.train()

        torch.cuda.synchronize()
        train_step_t0 = time.time()

        dt = 0.
        train_loss = 0.
        curr_batch_tokens_passed = 0
        for _ in range(grad_accum_steps):
            curr_batch_tokens_passed += batch.size # inputs.numel()
            with torch.autocast('cuda', torch.bfloat16):
                loss = graph(batch, with_metrics=False)
            loss = loss / grad_accum_steps
            train_loss += loss.detach() # for logging
            loss.backward()

            t0 = time.time()
            batch = next(train_dataloader_iterator) # prefetch the next batch while the GPU is busy with forward/backward
            t1 = time.time()
            dt += (t1 - t0) / grad_accum_steps
        tokens_passed += curr_batch_tokens_passed
        writer.add_scalar('time/dataloading(s)', dt, tokens_passed)
        writer.add_scalar('train/loss', train_loss, tokens_passed)

        if custom_logging is not None:
            for name, value in custom_logging().items():
                writer.add_scalar(name, value, tokens_passed)
                
        if grad_clip > 0.:
            grad_norm = torch.nn.utils.clip_grad_norm_(graph.parameters(), grad_clip)
            writer.add_scalar('optim/grad_norm', grad_norm, tokens_passed)

        # step the optimizers
        step_optimizers_func(step)

        torch.cuda.synchronize()
        train_step_t1 = time.time()
        dt = train_step_t1 - train_step_t0
        writer.add_scalar('time/train_step_time(s)', dt, tokens_passed)
        writer.add_scalar('time/tokens_per_sec(k)', curr_batch_tokens_passed / dt // 1000, tokens_passed)

        if checkpoint_dir is not None and checkpoint_every is not None and step % checkpoint_every == 0:
            torch.save(graph.state_dict(), f'{checkpoint_dir}/{step}_{tokens_passed}.pkl')

    if checkpoint_dir is not None:
        torch.save(graph.state_dict(), f'{checkpoint_dir}/final.pkl')
        
    return step, tokens_passed


def get_cosine_scheduler(step, start, end, total_steps):
    if step >= total_steps:
        return end
    progress = step / total_steps
    if end < start:
        return end + 0.5 * (start - end) * (1. + math.cos(math.pi * progress))
    else:
        return start + 0.5 * (end - start) * (1. - math.cos(math.pi * progress))
    

def linear_decay(step, num_iterations, num_warmup_steps=0, warmdown_ratio=0.45, final_lr_frac=0.1):
    x = min(0.9999, step / num_iterations)
    assert 0 <= x < 1
    lr = 1.0
    if step < num_warmup_steps:
        return step / num_warmup_steps
    elif x >= 1 - warmdown_ratio:
        w = (1 - x) / warmdown_ratio
        lr = w * 1.0 + (1 - w) * final_lr_frac
    return lr


def get_muon_momentum(step: int, num_iterations, num_warmup_steps=300, num_cooldown_steps=50, momentum_min=0.85, momentum_max=0.95):
    # warmup phase: linearly increase momentum from min to max
    # cooldown phase: linearly decrease momentum from max to min
    momentum_cd_start = num_iterations - num_cooldown_steps
    if step < num_warmup_steps:
        frac = step / num_warmup_steps
        momentum = momentum_min + frac * (momentum_max - momentum_min)
    elif step > momentum_cd_start:
        frac = (step - momentum_cd_start) / num_cooldown_steps
        momentum = momentum_max - frac * (momentum_max - momentum_min)
    else:
        momentum = momentum_max
    return momentum


def step_optimizers(
        step: int, 
        optimizers, 
        model, 
        num_iterations, warmdown_ratio, final_lr_frac, 
        num_adam_warmup_steps=0,
        num_muon_warmup_steps=300, num_muon_cooldown_steps=50, momentum_min=0.85, momentum_max=0.95,
        hetero=True
):
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * linear_decay(step, num_iterations, num_adam_warmup_steps, warmdown_ratio, final_lr_frac)

    momentum = get_muon_momentum(step, num_iterations, num_muon_warmup_steps, num_muon_cooldown_steps, momentum_min, momentum_max)
    for group in optimizers[1].param_groups:
        group["momentum"] = momentum

    if hetero and step % 2 == 0:
        optimizers[1].step()
        optimizers[1].zero_grad(set_to_none=True)
    else:
        for optimizer in optimizers:
            optimizer.step()
        model.zero_grad(set_to_none=True)


def get_last_mask(cu_seqlens):
    last_mask = torch.zeros(cu_seqlens[-1].item(), dtype=torch.bool, device='cuda')
    last_mask[cu_seqlens[1:] - 1] = True
    return last_mask
