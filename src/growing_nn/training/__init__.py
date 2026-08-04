"""
growing_nn.training
====================

Generic (growth-agnostic) train/eval loop and the knowledge-distillation
loss, split out of the original top-level ``engine.py`` / ``losses.py``.
"""

from growing_nn.training.engine import train_one_epoch, evaluate
from growing_nn.training.losses import DistillationLoss

__all__ = ["train_one_epoch", "evaluate", "DistillationLoss"]
