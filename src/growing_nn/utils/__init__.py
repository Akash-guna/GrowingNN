"""
growing_nn.utils
=================

Misc helpers used throughout the training scripts, split (from the
original monolithic ``utils.py``) by concern:

- :mod:`growing_nn.utils.distributed` - torch.distributed setup & helpers
- :mod:`growing_nn.utils.logging`     - ``SmoothedValue`` / ``MetricLogger``
- :mod:`growing_nn.utils.wandb_utils` - thin Weights & Biases wrappers

Everything is re-exported here so existing call sites that used to do
``import utils; utils.get_rank()`` etc. keep working with
``from growing_nn import utils; utils.get_rank()``.
"""

from growing_nn.utils.distributed import (
    setup_for_distributed,
    is_dist_avail_and_initialized,
    get_world_size,
    get_rank,
    is_main_process,
    save_on_master,
    init_distributed_mode,
)
from growing_nn.utils.logging import (
    SmoothedValue,
    MetricLogger,
    load_checkpoint_for_ema,
)
from growing_nn.utils.wandb_utils import init_wandb, log_wandb

__all__ = [
    "setup_for_distributed",
    "is_dist_avail_and_initialized",
    "get_world_size",
    "get_rank",
    "is_main_process",
    "save_on_master",
    "init_distributed_mode",
    "SmoothedValue",
    "MetricLogger",
    "load_checkpoint_for_ema",
    "init_wandb",
    "log_wandb",
]

# Backwards-compatible alias: the original codebase named this function
# `_load_checkpoint_for_ema` (leading underscore, "private").
_load_checkpoint_for_ema = load_checkpoint_for_ema
