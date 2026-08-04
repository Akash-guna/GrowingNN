"""
Top-level growth orchestration, called directly from the training loop:

- :func:`growth_wrapper` wraps every Q/K/V/proj/fc1/fc2 linear layer of a
  freshly-built model in a single-layer :class:`GrowthModel`, so later
  splits have a uniform representation to grow from.
- :func:`split_nodewise` runs one growth step: selects layers/neurons to
  split (:mod:`growing_nn.growth.selection`), applies the depth/width
  growth operators (:mod:`growing_nn.growth.mutate`), and rewires the
  model in place.
- :func:`remove_garbage` frees stale gradient tensors after a split.

Split out of the original top-level ``growth_utils_node_new.py`` with no
functional changes, other than giving ``find_split_layers_param_quota``'s
``eigs`` parameter a default (see the note on
:func:`growing_nn.growth.selection.find_split_layers_param_quota`, which
this module's ``split_nodewise`` relies on).
"""
import datetime
import gc
import time

import GPUtil
import torch

from growing_nn.growth.block import GrowthBlock, GrowthModel
from growing_nn.growth.eigen import calc_all_eigs  # noqa: F401  (re-exported for convenience)
from growing_nn.growth.layers import assign_model, get_all_linear_layers, get_all_linear_layers_transformer, ret_growth_model, return_arc_array
from growing_nn.growth.mutate import create_new_layer_new, create_width_growth, update_qk_model_width
from growing_nn.growth.selection import find_split_layers_param_quota


def convert(n):
    """Format a duration in seconds as ``H:MM:SS`` for log messages."""
    return str(datetime.timedelta(seconds=n))


def growth_wrapper(model):
    """Wrap every Q/K/V/proj/fc1/fc2 linear layer of every transformer
    block in ``model`` with a single-layer :class:`GrowthModel`, so that
    subsequent calls to :func:`split_nodewise` have a uniform
    representation to grow from.
    """
    for i in range(len(model.blocks)):
        model.blocks[i].attn.q = GrowthBlock(model.blocks[i].attn.q)
        print(f"Total Blocks: {len(model.blocks)} Block : {i} q")
        model.blocks[i].attn.k = GrowthBlock(model.blocks[i].attn.k)
        print(f"Total Blocks: {len(model.blocks)} Block : {i} k")
        model.blocks[i].attn.v = GrowthBlock(model.blocks[i].attn.v)
        print(f"Total Blocks: {len(model.blocks)} Block : {i} v")
        print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 1")
        model.blocks[i].attn.proj = GrowthBlock(model.blocks[i].attn.proj)
        print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 2")
        model.blocks[i].mlp.fc1 = GrowthBlock(model.blocks[i].mlp.fc1)
        print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 3")
        model.blocks[i].mlp.fc2 = GrowthBlock(model.blocks[i].mlp.fc2)
        print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 4")
    return model


def split_nodewise(
    model,
    optimizer,
    param_budget,
    epoch,
    percent=20,
    warmup=0,
    act_on=True,
    sel_layers_attn=None,
    sel_layers_data=None,
    neg_index_dic=None,
):
    """Run one growth step in place on ``model`` / ``optimizer``.

    If ``sel_layers_attn`` / ``sel_layers_data`` / ``neg_index_dic`` are
    not supplied, they are computed via
    :func:`growing_nn.growth.selection.find_split_layers_param_quota`.
    """
    start = time.time()
    if sel_layers_attn is None and sel_layers_data is None and neg_index_dic is None:
        sel_layers_attn, sel_layers_data, neg_index_dic = find_split_layers_param_quota(
            model, epoch, param_budget, percent
        )
    if len(sel_layers_data) == 0:
        return model
    cp1 = time.time()
    sum_reg = 0
    sum_max = 0
    l_neg = []
    for layer_data in sel_layers_data:
        s_max = time.time()
        neg_index = neg_index_dic[str(layer_data[0])]
        e_max = time.time()
        sum_max += e_max - s_max
        if len(neg_index) > layer_data[-1]:
            neg_index = neg_index[: layer_data[-1]]
        l_neg.append(len(neg_index))

        block = layer_data[2][0]
        growth_block = layer_data[2][1]
        block_model = ret_growth_model(model.blocks[block], growth_block)
        layers = get_all_linear_layers(block_model, typ="list")

        layer = layer_data[1]
        choices = [n for n in neg_index]
        choices.sort()
        s_layer = None
        for i, l in enumerate(layers):
            if l == layer_data[1]:
                layers.pop(i)
                s_layer = i
                break

        if growth_block == 2 and s_layer == 0:
            if len(choices) % 6 != 0:
                for i in range((len(choices) % 6)):
                    choices.pop(-1)
            model, layer, optimizer = create_width_growth(
                model, optimizer, layer, choices, block, growth_block, act_on, opposite=True, zeros=False
            )

        if growth_block == 4 and s_layer == 0:
            model, layer, optimizer = create_width_growth(
                model, optimizer, layer, choices, block, growth_block, act_on, opposite=True
            )

        new_layer, optimizer = create_new_layer_new(layer, choices, optimizer, 0.2)

        s_reg = time.time()
        lr = optimizer.param_groups[0]["lr"]
        e_reg = time.time()
        sum_reg += e_reg - s_reg

        layers = layers[:s_layer] + [layer, new_layer] + layers[s_layer:]
        layers = {str(i): layers[i] for i in range(len(layers))}
        _, _, architecture_array = return_arc_array(block_model.architecture_array, 0, s_layer, choices)
        model = assign_model(model, block, growth_block, GrowthModel(layers, architecture_array, act_on))
        cp2 = time.time()
        torch.cuda.empty_cache()

    for data in sel_layers_attn:
        q_data, k_data = data[0], data[1]
        model, optimizer = update_qk_model_width(model, optimizer, q_data, neg_index_dic, act_on, True)
        model, optimizer = update_qk_model_width(model, optimizer, k_data, neg_index_dic, act_on, False)

    print("Time for find splits :", convert(cp1 - start))
    print("Total Regression Time:", convert(sum_reg))
    print("Average Regression Time:", convert(sum_reg / len(sel_layers_data)))
    print("Total Max Layer Time:", convert(sum_max))
    print("Total Number of Selected Layers", len(sel_layers_data))
    print("Average Max Layer Time:", convert(sum_max / len(sel_layers_data)))
    print("Misc Time:", convert((cp2 - start) - (cp1 - start)))
    print("Total Time:", convert(cp2 - start))
    print("Negative Index Length:", l_neg)
    return model


def remove_garbage(model):
    """Free the gradient tensors of every linear layer in ``model``
    (called on non-main distributed ranks right after a split, whose
    stale gradients are no longer needed).
    """
    l, la = get_all_linear_layers_transformer(model)
    for i, layer in enumerate(l):
        print("Before Remove garbage")
        GPUtil.showUtilization()
        gradient = layer.weight.grad.cpu().detach().clone()
        del gradient
        gc.collect()
        torch.cuda.empty_cache()
        print("After Remove Garbage")
        GPUtil.showUtilization()
