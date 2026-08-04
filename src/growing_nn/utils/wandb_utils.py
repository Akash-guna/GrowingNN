"""
Thin Weights & Biases (wandb) helpers.

NOTE ON BEHAVIOR CHANGE
------------------------
The original ``main_growth_node_new_p.py`` called ``utils.init_wandb(...)``
and ``utils.log_wandb(...)``, but neither function was ever defined in the
original ``utils.py`` — those call sites would have raised
``AttributeError`` the moment ``--wandb`` was passed. This module supplies
the (previously missing) implementations so the growth training CLI
actually works when W&B logging is requested. If you were relying on
``--wandb`` silently failing, note that it will now log to Weights & Biases
for real.

Both functions are no-ops on non-main distributed ranks, consistent with
the rest of the logging/checkpointing code in this project.
"""
from typing import Any, Dict, Optional

from growing_nn.utils.distributed import is_main_process


def init_wandb(project: str, run_name: str, config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize a Weights & Biases run (main process only)."""
    if not is_main_process():
        return
    import wandb

    wandb.init(project=project, name=run_name, config=config or {})


def log_wandb(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log a dict of metrics to the current Weights & Biases run
    (main process only; no-op if a run hasn't been initialized).
    """
    if not is_main_process():
        return
    import wandb

    if wandb.run is None:
        return
    wandb.log(metrics, step=step)
