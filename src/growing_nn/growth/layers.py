"""
Utilities for locating ``nn.Linear`` layers inside a (possibly already
grown) model, mapping them back to their (block, sub-module) coordinates,
and renumbering a :class:`~growing_nn.growth.block.GrowthModel`
architecture tree after a split.

Split out of the original top-level ``growth_utils_node_new.py``.
"""
import torch.nn as nn


def get_all_linear_layers(model, typ="dict"):
    """Gets all Linear layers in the model.

    Input: Model
    Output: A List (or Dict, keyed by discovery order) of all Linear
        Layers in the model.
    """
    if typ == "dict":
        children = [i for i in model.children()]
        linear_layers = {}
        l_c = 0
        c = 0
        l = len(children)

        while c < l:
            grandchildren = [i for i in children[c].children()]
            if grandchildren == []:
                if isinstance(children[c], nn.Linear):
                    linear_layers[str(l_c)] = children[c]
                    l_c += 1
            else:
                children += grandchildren
                l = len(children)
            c += 1
        return linear_layers
    else:
        children = [i for i in model.children()]
        linear_layers = []
        l_c = 0
        c = 0
        l = len(children)

        while c < l:
            grandchildren = [i for i in children[c].children()]
            if grandchildren == []:
                if isinstance(children[c], nn.Linear):
                    linear_layers.append(children[c])
                    l_c += 1
            else:
                children += grandchildren
                l = len(children)
            c += 1
        return linear_layers


def ret_growth_model(block, num):
    """Return the growth-wrapped sub-module of a transformer ``block`` by
    index: 0=Q, 1=K, 2=V, 3=attn.proj, 4=mlp.fc1, 5=mlp.fc2.
    """
    if num == 0:
        return block.attn.q
    elif num == 1:
        return block.attn.k
    elif num == 2:
        return block.attn.v
    elif num == 3:
        return block.attn.proj
    elif num == 4:
        return block.mlp.fc1
    else:
        return block.mlp.fc2


def assign_model(model, block, num, gb):
    """Assign a (newly grown) :class:`GrowthModel` ``gb`` back onto
    ``model.blocks[block]`` at the sub-module identified by ``num`` (see
    :func:`ret_growth_model` for the index convention).
    """
    if num == 0:
        model.blocks[block].attn.q = gb
    elif num == 1:
        model.blocks[block].attn.k = gb
    elif num == 2:
        model.blocks[block].attn.v = gb
    elif num == 3:
        model.blocks[block].attn.proj = gb
    elif num == 4:
        model.blocks[block].mlp.fc1 = gb
    elif num == 5:
        model.blocks[block].mlp.fc2 = gb
    else:
        raise ValueError(f"Num {num} not defined for assign_model")
    return model


def get_all_linear_layers_transformer(model):
    """Flatten every Linear layer across every transformer block into a
    single list ``l``, alongside a parallel list ``l_att`` of
    ``[block_index, sub_module_index, position_within_sub_module]``
    coordinates (using the same 0..5 sub-module convention as
    :func:`ret_growth_model`).
    """
    l = []
    l_att = []
    for i in range(len(model.blocks)):
        a = get_all_linear_layers(model.blocks[i].attn.q, typ="list")
        l += a
        a_att = [[i, 0, j] for j in range(len(a))]
        l_att += a_att

        b = get_all_linear_layers(model.blocks[i].attn.k, typ="list")
        l += b
        b_att = [[i, 1, j] for j in range(len(b))]
        l_att += b_att

        c = get_all_linear_layers(model.blocks[i].attn.v, typ="list")
        l += c
        c_att = [[i, 2, j] for j in range(len(c))]
        l_att += c_att

        d = get_all_linear_layers(model.blocks[i].attn.proj, typ="list")
        l += d
        d_att = [[i, 3, j] for j in range(len(d))]
        l_att += d_att

        e = get_all_linear_layers(model.blocks[i].mlp.fc1, typ="list")
        l += e
        e_att = [[i, 4, j] for j in range(len(d))]
        l_att += e_att

        f = get_all_linear_layers(model.blocks[i].mlp.fc2, typ="list")
        l += f
        f_att = [[i, 5, j] for j in range(len(d))]
        l_att += f_att
    return l, l_att


def return_arc_array(a_array, i_num, sel_layer, positional):
    """Renumber a :class:`GrowthModel` architecture tree, inserting a
    growth-block placeholder ``[[0, 0], positional]`` at ``sel_layer``.

    Returns ``(name_array, next_free_index, new_architecture_array)``.
    """

    def create_numbered_arc_array(arc_array, init_num):
        name_array = []
        arc_new_arr = []
        for i in range(len(arc_array)):
            if arc_array[i] == 0:
                name_array.append(init_num)
                if init_num != sel_layer:
                    arc_new_arr.append(0)
                else:
                    arc_new_arr.append([[0, 0], positional])
                init_num += 1
            else:
                named_child_array, init_num, narr = create_numbered_arc_array(arc_array[i][0], init_num)
                name_array.append([named_child_array, arc_array[i][1]])
                arc_new_arr.append([narr, arc_array[i][1]])
        return name_array, init_num, arc_new_arr

    return create_numbered_arc_array(a_array, i_num)
