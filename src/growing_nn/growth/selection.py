"""
Decides *which* layers (and which neurons within them) get split on a
given growth step, subject to a total parameter budget split across four
categories (attention Q/K, attention V+proj, MLP fc1, MLP fc2).

Split out of the original top-level ``growth_utils_node_new.py`` with no
functional changes.
"""
import GPUtil
import torch

from growing_nn.growth.eigen import calc_all_eigs, layer_negative, ret_flattened
from growing_nn.growth.layers import get_all_linear_layers, get_all_linear_layers_transformer, ret_growth_model


def get_num_layers_below(model, la):
    """Total output width of the sub-module immediately "below" (i.e. the
    next sub-module index for) the layer described by attribute tuple
    ``la = [block, sub_module, position]``. Used to account for the
    parameter cost of widening the next layer's inputs when a Q/K/V layer
    grows. Returns 0 for the last sub-module in a block (mlp.fc2).
    """
    if la[1] < 5:
        next_block = ret_growth_model(model.blocks[la[0]], la[1] + 1)
        layers_next = get_all_linear_layers(next_block, typ="list")
        s = sum([l.out_features for l in layers_next])
        return s
    else:
        return 0


def find_split_layers_param_quota(model, epoch, param_budget, percent, eigs=None):
    """Select layers/neurons to split for this growth step under a total
    parameter budget, split evenly across four categories: MLP fc1, MLP
    fc2, attention V+proj, and attention Q/K (which must grow together as
    matched pairs).

    Input:
        model: the (growth-wrapped) model to analyze.
        epoch: current epoch (unused for selection logic itself, kept for
            signature/log-message compatibility with the caller).
        param_budget: total number of new parameters allowed this step,
            split into quarters across the four categories above.
        percent: unused placeholder, kept for backwards-compatible call
            signature (the original code no longer uses a percentile
            cutoff — every negative-eigenvalue neuron is considered until
            the budget or layer counts run out).
        eigs: optionally, pre-computed eigenvalues (see
            :func:`growing_nn.growth.eigen.calc_all_eigs`) to reuse
            instead of recomputing them.

    Output:
        (sel_layer_data_attn, sel_layer_data, neg_index_dic)
            sel_layer_data_attn: list of matched [q_data, k_data] pairs to
                grow together.
            sel_layer_data: list of [layer_key, layer, layer_attr, count]
                entries for MLP fc1/fc2 and attention V/proj layers to
                grow.
            neg_index_dic: per-layer negative-eigenvalue neuron indices
                (see :func:`growing_nn.growth.eigen.layer_negative`).
    """
    l, la = get_all_linear_layers_transformer(model)
    if eigs is None:
        eigs = calc_all_eigs(l)
        torch.save(eigs, "eigs_qkv.pt")

    flat_eig, flat_eig_layer = ret_flattened(eigs)

    neg_index_dic = layer_negative(eigs)
    rank = torch.Tensor(flat_eig).argsort()
    limit = rank.shape[0]

    param_budget_left_mlp1 = int(param_budget / 4)
    param_budget_left_attn = int(param_budget / 4)
    param_budget_left_mlp2 = int(param_budget / 4)
    param_budget_left_proj = int(param_budget / 4)
    sel_count_attn = 0
    sel_count_mlp1 = 0
    sel_count_proj = 0
    sel_count_mlp2 = 0
    d = {}
    qk_blocks = {}
    sel_layer_nums = []
    sel_block_attn_nums = []
    qk_blocks_layer_nums = []
    for i in range(int(limit)):
        pos = rank[i]
        if flat_eig[pos] > 0:
            continue
        layer_num = flat_eig_layer[pos]
        # Number of Parameters caused by 1 neuron  = No. of Previous Layer Neurons * 1 + 1. The +1 is for bias

        if str(layer_num) in d:
            d[str(layer_num)] += 1
        else:
            d[str(layer_num)] = 1
        v = d[str(layer_num)]
        l_attribute = la[int(layer_num)]

        if l_attribute[1] in [0, 1]:
            if str(l_attribute[0]) in qk_blocks:
                qk_blocks[str(l_attribute[0])][l_attribute[1]] += 1
            else:
                qk_blocks[str(l_attribute[0])] = [0, 0]
                qk_blocks[str(l_attribute[0])][l_attribute[1]] += 1
        if v >= 60:
            num_nodes = get_num_layers_below(model, l_attribute)
            if (
                l_attribute[1] == 4
                and layer_num not in sel_layer_nums
                and sel_count_mlp1 <= 2
                and param_budget_left_mlp1 > 0
            ):
                sel_layer_nums.append(layer_num)
                param_budget_left_mlp1 -= ((l[int(layer_num)].in_features + 1) * v) * 4
                if l_attribute[2] == 0:
                    param_budget_left_mlp1 -= num_nodes * v * 2
                sel_count_mlp1 += 1
            elif l_attribute[1] == 4 and layer_num in sel_layer_nums and param_budget_left_mlp1 >= 0:
                param_budget_left_mlp1 -= (l[int(layer_num)].in_features + 1) * 4
                if l_attribute[2] == 0:
                    param_budget_left_mlp1 -= num_nodes * 2
            else:
                d[str(layer_num)] -= 1
            if (
                l_attribute[1] == 5
                and layer_num not in sel_layer_nums
                and sel_count_mlp2 <= 2
                and param_budget_left_mlp2 > 0
            ):
                sel_layer_nums.append(layer_num)
                param_budget_left_mlp2 -= ((l[int(layer_num)].in_features + 1) * v) * 4
                if l_attribute[2] == 0:
                    param_budget_left_mlp1 -= num_nodes * v * 2
                sel_count_mlp2 += 1
            elif l_attribute[1] == 5 and layer_num in sel_layer_nums and param_budget_left_mlp2 >= 0:
                param_budget_left_mlp2 -= (l[int(layer_num)].in_features + 1) * 4
                if l_attribute[2] == 0:
                    param_budget_left_mlp1 -= num_nodes * 2
            else:
                d[str(layer_num)] -= 1

            if (
                l_attribute[1] == 2
                and layer_num not in sel_layer_nums
                and sel_count_attn <= 6
                and param_budget_left_attn > 0
            ):
                sel_layer_nums.append(layer_num)
                param_budget_left_attn -= ((l[int(layer_num)].in_features + 1) * v) * 2
                sel_count_attn += 1
            elif l_attribute[1] == 2 and layer_num in sel_layer_nums and param_budget_left_attn >= 0:
                param_budget_left_attn -= (l[int(layer_num)].in_features + 1) * 2
            else:
                d[str(layer_num)] -= 1

            if (
                l_attribute[1] == 3
                and layer_num not in sel_layer_nums
                and sel_count_proj <= 2
                and param_budget_left_proj > 0
            ):
                sel_layer_nums.append(layer_num)
                param_budget_left_proj -= ((l[int(layer_num)].in_features + 1) * v) * 2
                sel_count_proj += 1
            elif l_attribute[1] == 3 and layer_num in sel_layer_nums and param_budget_left_proj >= 0:
                param_budget_left_proj -= (l[int(layer_num)].in_features + 1) * 2
            else:
                d[str(layer_num)] -= 1

            if (
                l_attribute[1] in [0, 1]
                and layer_num not in sel_layer_nums
                and sel_count_attn <= 6
                and param_budget_left_attn > 0
            ):
                sel_layer_nums.append(layer_num)
                param_budget_left_attn -= ((l[int(layer_num)].in_features + 1) * v) * 2
                sel_count_proj += 1
            elif l_attribute[1] in [0, 1] and layer_num in sel_layer_nums and param_budget_left_proj >= 0:
                param_budget_left_attn -= (l[int(layer_num)].in_features + 1) * 2
            else:
                d[str(layer_num)] -= 1
            if (
                l_attribute[1] in [0, 1]
                and str(l_attribute[0]) not in sel_block_attn_nums
                and param_budget_left_attn > 0
            ):
                if qk_blocks[l_attribute[0]][0] >= 60 and qk_blocks[l_attribute[0]][1]:
                    sel_block_attn_nums.append(layer_num)
                    v_min = min(qk_blocks[str(l_attribute[0])])
                    # to make no divisibility problems appear when dividing into 6 heads
                    v_min -= v_min % 6
                    param_budget_left_attn -= (l[int(layer_num)].in_features + 1) * v_min * 2

                    qk_blocks_layer_nums[l_attribute[0]] = ["a", "a"]
                    qk_blocks_layer_nums[l_attribute[0]][l_attribute[1]] = str(layer_num)
            elif (
                l_attribute[1] in [0, 1]
                and str(l_attribute[0]) in sel_block_attn_nums
                and param_budget_left_attn > 0
            ):
                param_budget_left_attn -= (l[int(layer_num)].in_features + 1) * 2

        if (
            param_budget_left_mlp1 <= 0
            and param_budget_left_mlp2 <= 0
            and param_budget_left_proj <= 0
            and param_budget_left_attn <= 0
        ):
            print(param_budget_left_attn, param_budget_left_mlp1, param_budget_left_mlp2, param_budget_left_proj)
            break

    sel_layer_data = []
    GPUtil.showUtilization()

    c = 0
    for layer_num in sel_layer_nums:
        k = str(layer_num)
        v = d[k]
        sel_layer_data.append([k, l[int(k)], la[int(k)], v])
        c += 1
    del eigs, flat_eig, flat_eig_layer
    torch.cuda.empty_cache()

    sel_layer_data_attn = []
    for block in sel_block_attn_nums:
        v_data = qk_blocks[str(block)]
        layer_nums = qk_blocks_layer_nums[str(block)]
        q_ln, k_ln = layer_nums
        min_v = min(v_data)
        q_data = [str(q_ln), l[str(q_ln)], la[int(q_ln)], min_v]
        k_data = [str(k_ln), l[str(k_ln)], la[int(k_ln)], min_v]
        sel_layer_data_attn.append([q_data, k_data])

    return sel_layer_data_attn, sel_layer_data, neg_index_dic
