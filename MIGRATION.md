# Migration guide: old scripts → `growing_nn` package

This refactor is based on **`GrowingNN-dev.zip`** (the "newer" source you
provided partway through the task), not the original `GrowingNN-master.zip`.
The two differ substantially — the dev version replaces the old
regression-based layer initialization (`regression_utils_add.py`,
`models_v2.py`) with a simpler direct-computation approach, adds a
custom growth-friendly `VisionTransformer` with separately-splittable
Q/K/V layers (`growth_vision_transformer.py`), and adds a checkpoint
visualization script (`vis_growth.py`). This document reflects the dev
source only.

## File → module map

| Old file (top-level script) | New module(s) |
|---|---|
| `GrowthNew.py` | `growing_nn/growth/block.py` |
| `growth_utils_node_new.py` | split across `growing_nn/growth/{layers,eigen,selection,mutate,schedule}.py` |
| `growth_vision_transformer.py` | `growing_nn/models/vit.py` |
| `models.py` | `growing_nn/models/deit.py` |
| `datasets.py` | `growing_nn/data/datasets.py` |
| `augment.py` | `growing_nn/data/augment.py` |
| `samplers.py` | `growing_nn/data/samplers.py` |
| `engine.py` | `growing_nn/training/engine.py` |
| `losses.py` | `growing_nn/training/losses.py` |
| `utils.py` | split across `growing_nn/utils/{distributed,logging,wandb_utils}.py` |
| `main_growth_node_new_p.py` | `growing_nn/cli/args.py` (parser) + `growing_nn/cli/train_growth.py` (script body) |
| `vis_growth.py` | `growing_nn/scripts/visualize_growth.py` |

### Why `growth_utils_node_new.py` was split five ways

That one file (746 lines) mixed five genuinely different concerns, so it
was split by responsibility rather than kept as one "growth utils"
grab-bag:

- **`layers.py`** — pure bookkeeping: finding `nn.Linear` layers inside a
  model and mapping them to `(block, sub_module, position)` coordinates,
  renumbering a `GrowthModel` architecture tree after a split. No
  tensors touched, no torch grad state involved.
- **`eigen.py`** — the per-neuron saturation *diagnostic*: computing and
  plotting minimum eigenvalues from gradient outer-products. This is
  read-only analysis of the model's current state.
- **`selection.py`** — the *decision* layer: given the eigenvalue
  analysis, decide which layers/neurons get split this step, subject to
  a parameter budget split across categories (MLP fc1/fc2, attention
  V/proj, attention Q/K).
- **`mutate.py`** — the *operators* that actually create new layers
  (depth growth via `create_new_layer_new`, width growth via
  `create_layer_width` / `create_width_growth` /
  `update_next_layer_weights`), including wiring new parameters into the
  live optimizer.
- **`schedule.py`** — top-level orchestration that calls the above in
  sequence once per growth step (`split_nodewise`), plus the one-time
  setup that wraps a fresh model for growth (`growth_wrapper`) and
  post-split cleanup (`remove_garbage`).

This mirrors a natural "read state → decide → act → orchestrate"
pipeline and makes each piece independently testable/reusable (e.g. you
can call `eigen.calc_all_eigs` for diagnostics without touching
`mutate.py` at all).

## Pre-existing bugs fixed during this refactor

These are **not** design changes — each one is a place where the
original script would raise an exception or silently no-op in a
supported configuration. Each fix is called out again, in-place, in a
docstring/comment at the site of the fix.

1. **`main_growth_node_new_p.py` imported a module that doesn't exist.**
   `import models_v2` at the top of the file — but `models_v2.py` is not
   present anywhere in `GrowingNN-dev`. This would raise
   `ModuleNotFoundError` before the script could even define `main()`.
   Fixed by dropping the import in `growing_nn/cli/train_growth.py`.

2. **`utils.init_wandb` / `utils.log_wandb` were called but never
   defined.** `main_growth_node_new_p.py` calls both whenever `--wandb`
   is set, but `utils.py` has no such functions — this would raise
   `AttributeError` the first time `--wandb <name>` was passed. Fixed by
   implementing them in `growing_nn/utils/wandb_utils.py` (thin wrappers
   around the `wandb` package, no-ops on non-main distributed ranks,
   consistent with the rest of the logging/checkpointing code).

3. **`find_split_layers_param_quota` required 5 positional args but was
   always called with 4.** The function signature was
   `(model, epoch, param_budget, percent, eigs)` with no default for
   `eigs`, but its only call site in `split_nodewise` passes just
   `(model, epoch, param_budget, percent)`. Combined with an internal
   `if eigs == None:` check (which only makes sense if `None` is a valid
   input), this was clearly meant to have a default. Fixed by giving
   `eigs` a default of `None` in
   `growing_nn/growth/selection.find_split_layers_param_quota`.

4. **`vis_growth.py` imports a module that doesn't exist.**
   `from growth_utils_node import get_all_linear_layers` — the actual
   module is `growth_utils_node_new.py`. Fixed by importing
   `get_all_linear_layers` from `growing_nn.growth` in
   `growing_nn/scripts/visualize_growth.py`.

## Known issue left *unfixed* (flagged, not guessed at)

`vis_growth.py` (now `growing_nn/scripts/visualize_growth.py`) reads
`block.attn.qkv` off saved checkpoints — a single combined QKV layer.
But the model architecture actually used in this codebase
(`growing_nn/models/vit.py`) has separate `block.attn.q`, `.k`, `.v`
layers; there is no `.qkv` attribute. This script will raise
`AttributeError` if actually run against checkpoints from this model.

This was **not** fixed, because fixing it requires a design decision
this refactor shouldn't make unilaterally: should Q/K/V growth be
visualized as one merged "QKV" category (summing across the three), or
as three separate categories? Either is a reasonable choice with
different plot layouts. The docstring in `visualize_growth.py` flags
this explicitly — let me know which you'd like and I can wire it up.

## What was intentionally left alone

- **`timm` itself** — out of scope per your instruction to modularize
  only the project-specific growth code. `growing_nn` depends on it as a
  normal (pinned) PyPI package; nothing in `timm`'s source was touched.
- **The growth *algorithm*** — selection heuristics, budget splitting,
  weight-initialization formulas for new layers, etc. are copied
  verbatim. If something there looks like a bug (e.g. the somewhat
  ad-hoc `v >= 60` saturation threshold, or the `base_lr` scaling of
  `param_budget` at split time), it's preserved as-is since changing
  training-affecting logic wasn't part of this request.
- **`args.unscale_lr` interacting with `base_lr`** — in
  `train_growth.py`, `base_lr` is always defined now (it used to only be
  defined inside the `if not args.unscale_lr:` block, which meant
  passing `--unscale-lr` would crash later at split time with
  `NameError: base_lr`). Giving it a default of `args.lr` before the
  conditional is the minimal fix to keep `--unscale-lr` from crashing;
  flagged here since it's a behavior-adjacent change, unlike the four
  fixes above which only affect execution paths that previously could
  not run at all.
