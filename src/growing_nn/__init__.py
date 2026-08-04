"""
growing_nn
==========

A modularized version of the "GrowingNN" project: a DeiT/ViT training
pipeline (forked from Facebook's DeiT/timm) augmented with a "node
growth" mechanism that dynamically splits over-saturated Q/K/V and MLP
linear layers during training, both in depth (new sibling layers) and in
width (widening an existing layer and propagating the new width to
whatever consumes its output).

Sub-packages
------------
data       - dataset construction, transforms/augmentation, samplers
models     - the growth-friendly ``VisionTransformer`` backbone and DeiT
             builders on top of it
growth     - the growth/splitting machinery (GrowthModel, eigenvalue
             saturation analysis, quota-based layer selection, the
             depth/width growth operators, and the top-level schedule)
training   - the generic train/eval loop and the distillation loss
utils      - distributed-training helpers, logging/metrics, wandb helpers
cli        - the training command line entry point
             (``growing-nn-train-growth``)

This package depends on the (unmodified, pip-installable) ``timm``
library for its general training infrastructure (optimizers, schedulers,
Mixup, EMA, etc.); only the ``VisionTransformer`` backbone itself is
reimplemented locally, in :mod:`growing_nn.models.vit`, so its internal
layers can be split by the growth machinery.

See ``MIGRATION.md`` at the repository root for a full map from the
original flat-file scripts to these modules, plus notes on the handful of
pre-existing bugs that were fixed along the way (they are called out
explicitly, module by module, wherever they were fixed).
"""

__version__ = "0.1.0"
