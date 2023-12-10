from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from igl import fast_winding_number_for_meshes, point_mesh_squared_distance, read_obj
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from threestudio.utils.typing import *

def custom_chunk_batch(func: Callable, chunk_size: int, depth: int, *args, **kwargs) -> Any:
    if chunk_size <= 0:
        return func(*args, **kwargs)
    B = None
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    assert (
        B is not None
    ), "No tensor found in args or kwargs, cannot determine batch size."
    out = defaultdict(list)
    out_type = None
    for i in range(0, B, chunk_size):
        out_chunk = func(
            *[
                arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg
                for arg in args
            ],
            **{
                k: arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg
                for k, arg in kwargs.items()
            },
        )
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(
                f"Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}."
            )
            exit(1)
            
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            out[k].append(v.reshape(depth, v.shape[0]//depth, -1).permute(1, 0, 2))

    if out_type is None:
        return None
    
    out_merged: Dict[Any, Optional[torch.Tensor]] = {}
    for k, v in out.items():
        if all([vv is None for vv in v]):
            # allow None in return value
            out_merged[k] = None
        elif all([isinstance(vv, torch.Tensor) for vv in v]):
            out_merged[k] = torch.cat(v, dim=0)
            out_merged[k] = out_merged[k].permute(1, 0, 2).flatten(0, 1)
        else:
            raise TypeError(
                f"Unsupported types in return value of func: {[type(vv) for vv in v if not isinstance(vv, torch.Tensor)]}"
            )

    if out_type is torch.Tensor:
        return out_merged[0]
    elif out_type in [tuple, list]:
        return out_type([out_merged[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out_merged