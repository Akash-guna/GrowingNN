# Beyond Uniform Scaling: Exploring Depth Heterogeneity in Neural Architectures

**Official implementation** of the ICLR 2024 Tiny Paper:

> **Beyond Uniform Scaling: Exploring Depth Heterogeneity in Neural Architectures**
> Akash Guna R.T.\*, Arnav Chavan\*, Deepak Gupta
> Nyun AI · Transmute AI Lab (Texmin Hub), IIT (ISM) Dhanbad
> *Tiny Papers Track, ICLR 2024*
> \*Equal contribution

[[Paper]](./217_Beyond_Uniform_Scaling_Exp.pdf) · [Citation](#citation) · [Method → Code map](#method--code-map)

---

## Abstract

Conventional scaling of neural networks typically involves designing a
base network and growing different dimensions like width, depth, etc. of
the same by some predefined scaling factors. We introduce an automated
scaling approach leveraging second-order loss landscape information. Our
method is flexible towards skip connections, a mainstay in modern vision
transformers. Our training-aware method jointly scales and trains
transformers without additional training iterations. Motivated by the
hypothesis that not all neurons need uniform depth complexity, our
approach embraces **depth heterogeneity**. Extensive evaluations on
DeiT-S with ImageNet-100 show a **2.5% accuracy gain** and **10%
parameter efficiency improvement** over conventional scaling. Scaled
networks demonstrate superior performance upon training small-scale
datasets from scratch.

## Method summary

1. **Start small.** Build a reduced DeiT-S by halving the width of the
   intermediate QKV and MLP-FC1 layers (Fig. 1b), leaving skip
   connections untouched, and train it for an initial warmup period.
2. **Find saddle points.** Every *scaling interval*, approximate each
   neuron's Hessian with the compute-efficient *splitting matrix* of Wu
   et al. (2019) and take its minimum eigenvalue. Neurons with negative
   minimum eigenvalues sit at saddle points and are candidates for
   growth.
3. **Select under a budget.** Rank candidate neurons by eigenvalue,
   require at least a *layer threshold* (60) eligible neurons per layer,
   and select until a *parameter budget* for that scaling step is
   exhausted, split across the QKV, projection, and MLP-FC1/FC2 layers.
4. **Grow with function preservation.** For each selected neuron, add
   **two** new neurons with equal magnitude but opposite polarity,
   wired in as a GELU-gated skip connection (Eq. 3):

   ```
   O'_S = O_S + GeLU(O_A+ + O_A-) + O_A+ + O_A-
   ```

   At initialization `O_A+ + O_A- = 0`, so the pre-existing function is
   exactly preserved (proof in Appendix B.4) while gradients can still
   flow into the new neurons from step one.
5. **Repeat.** Steps 2-4 run every scaling interval (30 epochs in the
   paper's main results) until the target parameter count is reached, so
   the whole procedure runs *training-aware* -- the network scales
   in-place, with no separate re-training phase.

## Results

**Table 1 -- Scaling DeiT-S on ImageNet-100.** Heterogeneous (this paper)
vs. Homogeneous (conventional uniform) scaling.

| Scaling | Base Param. (M) | Final Param. (M) | Base FLOPs (G) | Final FLOPs (G) | Top-1 | Top-5 |
|---|---|---|---|---|---|---|
| Homogeneous | 21.7 | 21.7 | 4.6 | 4.6 | 77.80 | 93.16 |
| Heterogeneous | 11.0 | 15.6 | 2.3 | 3.1 | 79.16 | 94.00 |
| Heterogeneous | 11.0 | 19.4 | 2.3 | 3.9 | **80.36** | **94.58** |

**Table 2 -- Training from scratch on CIFAR-100** (transferring the
ImageNet-100-derived architecture, not the weights).

| Model | Param. (M) | FLOPs (G) | Top-1 | Top-5 |
|---|---|---|---|---|
| DeiT-S (Homogeneous) | 21.7 | 4.6 | 58.9 | 78.9 |
| DeiT-S (Heterogeneous) | 19.4 | 3.9 | **78.1** | **95.0** |

See the paper's Tables 3-4 for the base-model width-reduction ablation
(QKV vs. FC1 reduction ratio) and the scaling-interval ablation
({10, 20, 30, 50} epochs, with 30 performing best).

## Repository layout

This codebase is organized as an installable Python package,
`growing_nn`, split by concern rather than kept as a handful of flat
training scripts:

```
src/growing_nn/
  data/       dataset construction, augmentation pipelines, distributed samplers
  models/     the growth-friendly ViT backbone (vit.py) + DeiT builders (deit.py)
  growth/     the growth/splitting machinery itself:
                block.py      - GrowthModel / GrowthBlock (Eq. 3)
                layers.py     - locating Linear layers, renumbering architecture trees
                eigen.py      - splitting-matrix eigenvalue analysis (Eq. 1-2, SB.1)
                selection.py  - parameter-budget neuron selection (SB.2)
                mutate.py     - the depth/width growth operators (Eq. 4-11, SB.3-B.4)
                schedule.py   - top-level orchestration (growth_wrapper, split_nodewise)
  training/   the generic train/eval loop (engine.py) and distillation loss (losses.py)
  utils/      distributed-training helpers, logging/metrics, wandb helpers
  cli/        args.py (argparse parser) + train_growth.py (the training entry point)
  scripts/    visualize_growth.py (standalone checkpoint-diff visualization)
```

See [`MIGRATION.md`](./MIGRATION.md) for the map from this layout back
to the original flat scripts, and a list of the (pre-existing, now
fixed) bugs found along the way.

## Install

```bash
pip install -e .
```

This pulls in `torch`, `torchvision`, and `timm` (pinned to the `0.5.x`
line -- see the comment in `pyproject.toml` for why), plus the smaller
utility dependencies (`GPUtil`, `joblib`, `matplotlib`, `seaborn`,
`psutil`, `wandb`).

## Reproducing the paper's DeiT-S / ImageNet-100 result (Table 1, row 3)

The paper's main result uses a DeiT-S backbone with QKV and MLP-FC1
width halved (`1:1` ratio, §C.2), 50 warmup epochs, a 30-epoch scaling
interval, and a per-step parameter budget tuned to land at ~19.4M
parameters after growth (§B.2-C.3):

```bash
growing-nn-train-growth \
    --data-set IM100 --data-path /path/to/imagenet100 \
    --model deit_small_patch16_224 \
    --batch-size 256 --epochs 300 \
    --initwarm 50 --split_epochs 30 \
    --param_budget 580000 \
    --cp models/checkpoint --folder models/splits
```

`--param_budget` is the per-scaling-step parameter budget (§B.2); tune
it alongside `--split_epochs` (the paper's "Scaling Interval") to hit a
target final parameter count, as in Table 4's ablation. The "layer
threshold" of 60 eligible neurons (§B.2) is currently fixed in
[`selection.py`](src/growing_nn/growth/selection.py) rather than exposed
as a flag.

Run `growing-nn-train-growth --help` for the full flag list (unchanged
from the original scripts -- see `MIGRATION.md`).

## Visualizing growth across checkpoints

To reproduce plots like Table 1-4's parameter/FLOP breakdowns from a
directory of saved growth-step checkpoints:

```bash
growing-nn-visualize-growth models/splits
```

**Known limitation:** this visualizer still reads `block.attn.qkv` (a
single combined layer) off each checkpoint, but the current backbone
(`growing_nn.models.vit`) uses separate `block.attn.q` / `.k` / `.v`
layers -- see the note in
[`visualize_growth.py`](src/growing_nn/scripts/visualize_growth.py) and
item 4 in `MIGRATION.md`.

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{
title={Beyond Uniform Scaling: Exploring Depth Heterogeneity in Neural Architectures},
author={Akash Guna R.T and Arnav Chavan and Deepak Gupta},
booktitle={The Second Tiny Papers Track at ICLR 2024},
year={2024},
url={https://openreview.net/forum?id=mURVIdmojf}
}
```


## License

Apache License 2.0 -- see [`LICENSE`](./LICENSE).
