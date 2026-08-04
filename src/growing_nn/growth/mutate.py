"""
Low-level operators that actually mutate layers: creating a new
"depth-growth" layer (a fresh linear layer, appended as a sibling branch
of an existing one, wired together by :class:`~growing_nn.growth.block.GrowthModel`),
and "width growth" (widening an existing Q/K/V layer and propagating the
new width into whatever consumes its output).

Split out of the original top-level ``growth_utils_node_new.py`` with no
functional changes.
"""
import torch
import torch.nn as nn

from growing_nn.growth.block import GrowthModel
from growing_nn.growth.layers import assign_model, get_all_linear_layers, ret_growth_model, return_arc_array


def create_new_layer_new(layer, choices, optimizer, reduction_factor=0.2):
    """Creates a New Layer for depth addition. New layer contains equal
    positive weights and negative weights with a reduction factor.
    (PW = reduction_factor * W, NW = -1 * PW)

    Inputs:
        layer: Layer which we want to increase the depth of.
        choices: List of selected neuron positions to increase depth.
        optimizer: add the new weights and bias to the optimizer.
        reduction_factor: multiplicative factor applied to new positive
            and negative weights to prevent an exact copy of the selected
            neuron.

    Outputs:
        new_layer: newly created layer.
        optimizer: updated optimizer.
    """
    # A single new layer for both positive and negative weights input -> same input as previous layer. output -> num(positve+negative weights)
    new_layer = nn.Linear(layer.in_features, 2 * len(choices))
    # Layer -> Weight (nn.Parameter) -> Tensor (nn.Parameter.data)
    layer_weight = layer.weight.data
    layer_bias = layer.bias.data
    # new_bias = bias[selected_neurons] * reduction factor. suppose 5 neurons are selected out of 10 shape of bias = (1,5)
    bias = layer_bias[choices] * reduction_factor
    # new_weight = weight[selected_neurons] * reduction factor. suppose 5 neurons are selected out of 10 shape of bias = (5,X)
    weight = layer_weight[choices, :] * reduction_factor
    # Negative Weights and Bias.
    bias_neg = layer_bias[choices] * reduction_factor * -1
    weight_neg = layer_weight[choices, :] * reduction_factor * -1
    # Concatenate positive and negative weights
    weight_f = torch.cat([weight, weight_neg], axis=0)
    bias_f = torch.cat([bias, bias_neg], axis=0)
    # assign weights and bias to the weights and bias of new layer.
    weight_f.requires_grad = True
    new_layer.weight = nn.Parameter(weight_f)
    new_layer.weight.requires_grad = True
    bias_f.requires_grad = True
    new_layer.bias = nn.Parameter(bias_f)
    new_layer.bias.requires_grad = True
    # Append New Parameters to Param Group of optimizer. The bias has no weight decay(0) and weight has weight decay(1)
    optimizer.param_groups[0]["params"].append(new_layer.bias)
    optimizer.param_groups[1]["params"].append(new_layer.weight)
    return new_layer, optimizer


def create_layer_width(layer, choices, optimizer, reduction_factor=0.2, opposite=True):
    """Creates a New Layer with increased width.

    Inputs:
        layer: Layer which we want to increase the width of.
        choices: List of selected neuron positions to increase width.
        optimizer: add the new weights and bias to the optimizer.
        reduction_factor: multiplicative factor applied to new weights to
            prevent an exact copy of a neuron.
        opposite: toggled when a mixture of positive and negative weights
            are needed.

    Outputs:
        new_layer: newly created layer.
        optimizer: updated optimizer.
    """
    # New number of neurons = number of old neurons + 2* number of selected neurons to grow
    new_width = layer.out_features + 2 * len(choices)
    # creating new layer
    new_layer = nn.Linear(layer.in_features, new_width)
    # New Weights (positively weighted). multiplied with reduction factor to prevent same copy of selected neurons.
    weight_p = layer.weight.data[choices, :] * reduction_factor
    # New Weights (negatively weighted if opposite=True). multiplied with reduction factor to prevent same copy of selected neurons.
    if opposite:
        weight_n = layer.weight.data[choices, :] * reduction_factor * -1
        bias_n = layer.bias.data[choices] * reduction_factor * -1
    else:
        weight_n = layer.weight.data[choices, :] * reduction_factor
        bias_n = layer.bias.data[choices] * reduction_factor
    # Appending both weights, similar operation for bias
    new_weight = torch.cat([layer.weight.data, weight_p, weight_n], axis=0)
    bias_p = layer.bias.data[choices] * reduction_factor
    new_bias = torch.cat([layer.bias.data, bias_p, bias_n], axis=0)
    # Assigning new values to weights and bias
    new_weight.requires_grad = True
    new_layer.weight = nn.Parameter(new_weight)
    new_layer.weight.requires_grad = True
    new_bias.requires_grad = True
    new_layer.bias = nn.Parameter(new_bias)
    new_layer.bias.requires_grad = True
    # Adding new weights and bias to param group
    optimizer.param_groups[0]["params"].append(new_layer.bias)
    optimizer.param_groups[1]["params"].append(new_layer.weight)
    return new_layer, optimizer


def update_next_layer_weights(layer_list, choices, optimizer, zeros=False):
    """When updating a layer's width, the child layers of that layer
    should have an increased number of inputs. This function takes a list
    of children of the width-grown layer and increases their input width.

    Inputs:
        layer_list: list of all child layers of a width-grown layer.
        choices: list of selected neuron positions.
        optimizer: to add new weights to the optimizer.
        zeros: for a V -> proj layer the new weights have to be zero. If
            True, those weights will be zero; otherwise they're
            initialized with random weights.

    Outputs:
        new_layer_list: list of updated children layers.
        optimizer: updated optimizer.
    """
    new_layer_list = []
    for layer in layer_list:
        # new Layer created with updated inputs
        new_layer = nn.Linear(layer.in_features + 2 * len(choices), layer.out_features)  # (12,10)
        # weight is transposed so that we have rows as weights for each input
        weight = layer.weight.data.T  # (10,10)
        # A random weight of chosen weight dimensions.
        # No issue because, we have positive and negative weighted neurons as input so it would become zero.
        chosen_weights = torch.rand(weight[choices, :].shape).to(weight.device)  # (1,10)
        # Concatenating two equal copies for weights so they get cancelled when multiplied with positive and negative weights.
        new_weight = torch.cat([weight, chosen_weights, chosen_weights], axis=0)
        new_weight = new_weight.T  # (10,12)
        new_weight.requires_grad = True
        if zeros:
            # if zeros == True we would replace random weights with 0 weights.
            new_weight = torch.zeros(new_weight.shape, requires_grad=True, device=new_weight.device)
        # Updating parameters and updating optimizer
        new_layer.weight = nn.Parameter(new_weight)
        new_layer.weight.requires_grad = True
        new_layer.bias = layer.bias
        new_layer.bias.requires_grad = True
        new_layer_list.append(new_layer)
        optimizer.param_groups[0]["params"].append(new_layer.bias)
        optimizer.param_groups[1]["params"].append(new_layer.weight)
    return new_layer_list, optimizer


def create_width_growth(model, optimizer, layer, choices, block, growth_block, act_on, opposite=True, update=True, zeros=False):
    """Performs the width-growth operation.

    Input:
        model: model.
        optimizer: optimizer.
        layer: layer to grow the width of.
        choices: list of selected neuron positions.
        block: current transformer block number.
        growth_block: current growth-block sub-module index (see
            :func:`growing_nn.growth.layers.ret_growth_model`).
        act_on: whether to activate neurons.
        opposite: whether to have opposite-signed neuron weights.
        zeros: whether to have zero-weight neurons for the child layers.

    Output:
        model: updated with the children of the widened layer.
        layer: new layer after growing width.
        optimizer: updated optimizer.
    """
    layer, optimizer = create_layer_width(layer, choices, optimizer, opposite=opposite)
    next_block = ret_growth_model(model.blocks[block], growth_block + 1)
    layers_next = get_all_linear_layers(next_block, typ="list")
    if update:
        layers_next, optimizer = update_next_layer_weights(layers_next, choices, optimizer, zeros)
    layers_next = {str(i): layers_next[i] for i in range(len(layers_next))}
    gb = GrowthModel(layers_next, next_block.architecture_array, act_on)
    model = assign_model(model, block, growth_block + 1, gb)
    return model, layer, optimizer


def update_qk_model_width(model, optimizer, data, neg_index_dic, act_on, opposite):
    """Apply a width-growth step to one of a matched Q/K layer pair (see
    :func:`growing_nn.growth.selection.find_split_layers_param_quota`).
    """
    neg_index = neg_index_dic[str(data[0])]
    if len(neg_index) > data[-1]:
        neg_index = neg_index[: data[-1]]
    choices = [n for n in neg_index]
    choices.sort()
    block = data[2][0]
    growth_block = data[2][1]
    block_model = ret_growth_model(model.blocks[block], growth_block)
    layers = get_all_linear_layers(block_model, typ="list")
    s_layer = None
    for i, l in enumerate(layers):
        if l == data[1]:
            layers.pop(i)
            s_layer = i
            break
    layer = data[1]
    model, layer, optimizer = create_width_growth(
        model, optimizer, layer, choices, block, growth_block, act_on, opposite=opposite, update=False
    )
    layers = layers[:s_layer] + [layer] + layers[s_layer:]
    layers = {str(i): layers[i] for i in range(len(layers))}
    _, _, architecture_array = return_arc_array(block_model.architecture_array, 0, s_layer, choices)
    model = assign_model(model, block, growth_block, GrowthModel(layers, architecture_array, act_on))
    return model, optimizer
