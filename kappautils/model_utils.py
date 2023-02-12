import torch
from .context_managers import temp_mode_change
from contextlib import contextmanager



def get_paramnames_with_no_gradient(model):
    return [name for name, param in model.named_parameters() if param.grad is None and param.requires_grad]


def get_output_shape_of_model(model, forward_fn, **forward_kwargs):
    with temp_mode_change(model=model, mode=False):
        # get outputshape from forward pass
        x = torch.ones(1, *model.input_shape, device=model.device)
        output = forward_fn(x, **forward_kwargs)
    return tuple(output.shape[1:])



@torch.no_grad()
def copy_params(source_model, target_model):
    # TODO i think inplace operations are okay here
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data = source_param.data


@torch.no_grad()
def update_ema(source_model, target_model, target_factor):
    # TODO i think inplace operations are okay here
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data = target_param.data * target_factor + source_param.data * (1. - target_factor)
    for target_buffer, source_buffer in zip(target_model.buffers(), source_model.buffers()):
        target_buffer.data.copy_(source_buffer.data)


def get_trainable_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_frozen_param_count(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


@torch.no_grad()
def backup_all_buffers(model):
    buffers = {}
    for name, buffer in model.named_buffers():
        buffers[name] = buffer.clone()
    return buffers

@torch.no_grad()
def restore_all_buffers(model, buffers):
    for name, buffer in model.named_buffers():
        buffer.data.copy_(buffers[name])
    return buffers

@contextmanager
def preserve_buffers(model):
    buffer_bkp = backup_all_buffers(model=model)
    yield
    restore_all_buffers(model=model, buffers=buffer_bkp)


