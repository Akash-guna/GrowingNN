import GPUtil
import os
import torch
import torch.nn as nn
import numpy as np
from GrowthNew import GrowthModel, GrowthBlock
from torch.multiprocessing import set_start_method
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    set_start_method('spawn')
except RuntimeError:
    pass
from joblib import Parallel, delayed
import time
import datetime
import gc


def convert(n):
    return str(datetime.timedelta(seconds=n))


def get_all_linear_layers(model, typ="dict"):
    '''
    Gets all Linear layers in the model.
    Input: Model
    Output: A List of all Linear Layers in the model
    '''
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


def split_matrix(gradient, weight=None):
    ''' '''

    def second_order_derivative():
        return torch.ones(gradient.shape)

    sm = gradient.view(1, -1).cpu() * second_order_derivative().view(-1, 1)
    return sm


def calculate_min_eig(gradient):
    try:
        splitting = split_matrix(gradient)
        eig, _ = torch.linalg.eig(splitting.cpu())
        min = torch.min(eig.cpu().double())
        del eig, splitting
        gc.collect()
        torch.cuda.empty_cache()
    except:
        min = 0
    return min


def growth_wrapper(model):
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


def return_arc_array(a_array, i_num, sel_layer, positional):
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

def return_num_arc_array(a_array, i_num):
    def create_numbered_arc_array(arc_array, init_num):
        name_array = []
        for i in range(len(arc_array)):
            if arc_array[i] == 0:
                name_array.append(init_num)
                init_num += 1
            else:
                named_child_array, init_num, narr = create_numbered_arc_array(arc_array[i][0], init_num)
                name_array.append([named_child_array, arc_array[i][1]])
        return name_array, init_num

    return create_numbered_arc_array(a_array, i_num)
def get_all_linear_layers_transformer(model):
    l = []
    l_att = []
    for i in range(len(model.blocks)):
        a = get_all_linear_layers(model.blocks[i].attn.q, typ='list')
        l += a
        a_att = [[i, 0, j] for j in range(len(a))]
        l_att += a_att

        b = get_all_linear_layers(model.blocks[i].attn.k, typ='list')
        l += b
        b_att = [[i, 1, j] for j in range(len(b))]
        l_att += b_att
        c = get_all_linear_layers(model.blocks[i].attn.v, typ='list')
        l += c
        c_att = [[i, 2, j] for j in range(len(c))]
        l_att += c_att
        d = get_all_linear_layers(model.blocks[i].attn.proj, typ='list')
        l += d
        d_att = [[i, 3, j] for j in range(len(d))]
        l_att += d_att
        e = get_all_linear_layers(model.blocks[i].mlp.fc1, typ='list')
        l += e
        e_att = [[i, 4, j] for j in range(len(d))]
        l_att += e_att
        f = get_all_linear_layers(model.blocks[i].mlp.fc2, typ='list')
        l += f
        f_att = [[i, 5, j] for j in range(len(d))]
        l_att += f_att
    return l, l_att


def calc_all_eigs(layers):
    eigs = []
    for i, layer in enumerate(layers):
        try:
            gradient = layer.weight.grad.cpu().detach().clone()
        except:
            print(layer)
        s = gradient.shape[0]
        min_eigs = Parallel(n_jobs=24)(delayed(calculate_min_eig)(gradient[i]) for i in range(s))
        eigs.append(min_eigs)
        del gradient
        gc.collect()
        torch.cuda.empty_cache()
    return eigs


def ret_flattened(eig):
    flat_eig = []
    flat_eig_layer = []
    for i, e in enumerate(eig):
        flat_eig += e
        flat_eig_layer += [i for j in range(len(e))]
    return flat_eig, flat_eig_layer


def layer_negative(eigs):
    neg_index_dic = {}
    neg_eig_dic = {}
    for layer, eig in enumerate(eigs):
        for j, e in enumerate(eig):
            if e < 0:
                if str(layer) in neg_index_dic:
                    neg_index_dic[str(layer)].append(j)
                    neg_eig_dic[str(layer)].append(e)
                else:
                    neg_index_dic[str(layer)] = [j]
                    neg_eig_dic[str(layer)] = [e]

    for k in neg_index_dic.keys():
        sort_pos = np.argsort(np.array(neg_eig_dic[k]))
        neg_index_dic[k] = np.array(neg_index_dic[k])[sort_pos]
    return neg_index_dic


def get_num_layers_below(model, la):
    if la[1] < 5:
        next_block = ret_growth_model(model.blocks[la[0]], la[1] + 1)
        layers_next = get_all_linear_layers(next_block, typ="list")
        s = sum([l.out_features for l in layers_next])
        return s
    else:
        return 0


def find_split_layers_depth(model, epoch, param_budget, percent,layer_threshold=40):
    l, la = get_all_linear_layers_transformer(model)
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
    qk_blocks_layer_nums = {}
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
        if v >= layer_threshold:
            if (
                l_attribute[1] == 4
                and layer_num not in sel_layer_nums
                and sel_count_mlp1 <= 2
                and param_budget_left_mlp1 > 0
            ):
                sel_layer_nums.append(layer_num)
                param_budget_left_mlp1 -= ((l[int(layer_num)].in_features + 1) * v) * 2
                sel_count_mlp1 += 1
            elif l_attribute[1] == 4 and layer_num in sel_layer_nums and param_budget_left_mlp1 >= 0:
                param_budget_left_mlp1 -= (l[int(layer_num)].in_features + 1) * 2
            elif l_attribute[1] == 4:
                d[str(layer_num)] -= 1
            if (
                l_attribute[1] == 5
                and layer_num not in sel_layer_nums
                and sel_count_mlp2 <= 2
                and param_budget_left_mlp2 > 0
            ):
                sel_layer_nums.append(layer_num)
                param_budget_left_mlp2 -= ((l[int(layer_num)].in_features + 1) * v) * 4
                sel_count_mlp2 += 1
            elif l_attribute[1] == 5 and layer_num in sel_layer_nums and param_budget_left_mlp2 >= 0:
                param_budget_left_mlp2 -= (l[int(layer_num)].in_features + 1) * 4
            elif l_attribute[1] == 5:
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
            elif l_attribute[1] == 3:
                d[str(layer_num)] -= 1

            if (
                l_attribute[1] in [0, 1, 2]
                and layer_num not in sel_layer_nums
                and sel_count_attn <= 6
                and param_budget_left_attn > 0
            ):
                sel_layer_nums.append(layer_num)
                param_budget_left_attn -= ((l[int(layer_num)].in_features + 1) * v) * 2
                sel_count_attn += 1
            elif l_attribute[1] in [0, 1,2] and layer_num in sel_layer_nums and param_budget_left_proj >= 0:
                param_budget_left_attn -= (l[int(layer_num)].in_features + 1) * 2
            elif l_attribute[1] in [0, 1,2]:
                d[str(layer_num)] -= 1
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

    return sel_layer_data, neg_index_dic

def find_split_layers_width():
    l, la = get_all_linear_layers_transformer(model)
    eigs = calc_all_eigs(l)
    torch.save(eigs, "eigs_qkv.pt")
    flat_eig, flat_eig_layer = ret_flattened(eigs)

    neg_index_dic = layer_negative(eigs)
    rank = torch.Tensor(flat_eig).argsort()
    limit = rank.shape[0]
    sel_layer_nums=[]
    param_budget_left_mlp1 = int(param_budget / 4)
    param_budget_left_attn = int(param_budget / 4)
    param_budget_left_mlp2 = int(param_budget / 2)


    qk_blocks = {}
    proj_fc2 ={}
    d = {}
    for i in range(int(limit)):
            pos = rank[i]
            if flat_eig[pos] > 0:
                continue
            layer_num = flat_eig_layer[pos]

            if l_attribute[2]==0 and l_attribute[1] in [0,1]:
                if str(l_attribute[0]) not in qk_blocks.keys():
                    qk_blocks[str(l_attribute[0])] =[0,0]
                qk_blocks[str(l_attribute[0])][str(l_attribute[1])] = layer_num
              
            if l_attribute[2]==0 and l_attribute[1] ==3:
                if str(l_attribute[0]+1) not in proj_fc2.keys():
                    proj_fc2[l_attribute[0]+1] =[0,0]
                proj_fc2[l_attribute[0]+1][0] = layer_num
            
            if l_attribute[2]==0 and l_attribute[1] ==5:
                if str(l_attribute[0]) not in proj_fc2.keys():
                    proj_fc2[l_attribute[0]] =[0,0]
                proj_fc2[l_attribute[0]][1] = layer_num

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
            layer = l[int(layer_num)]

            if l_attribute[2]==0 and v >= layer_threshold:
                if l_attribute[1] == 4 and layer_num not in sel_layer_nums and  param_budget_left_mlp1 > 0:
                    below_neurons = get_num_layers_below(model, l_attribute)
                    param_budget_left_mlp1 -= (layer.in_features * below_neurons * v + v)*2
                    sel_layer_nums.append(layer_num)
                elif l_attribute[1] == 4 and layer_num in sel_layer_nums and  param_budget_left_mlp1 > 0:
                    param_budget_left_mlp1 -= (layer.in_features * below_neurons +1)*2
                elif l_attribute[1] == 4:
                    d[str(layer_num)]-=1
                
                if l_attribute[1] == 5 and layer_num not in sel_layer_nums and  param_budget_left_mlp2 > 0:
                    below_neurons = get_num_layers_below(model, l_attribute)
                    param_budget_left_mlp1 -= (layer.in_features * below_neurons * v + v)*2 *4
                    sel_layer_nums.append(layer_num)
                elif l_attribute[1] == 5 and layer_num in sel_layer_nums and  param_budget_left_mlp1 > 0:
                    param_budget_left_mlp1 -= (layer.in_features * below_neurons +1)*2 *4
                elif l_attribute[1] == 5:
                    d[str(layer_num)]-=1

                if l_attribute[1]==2 and layer_num not in sel_layer_nums and  param_budget_left_attn > 0:
                    below_neurons = get_num_layers_below(model, l_attribute)
                    param_budget_left_attn -= (layer.in_features * below_neurons * v + v)*2
                    sel_layer_nums.append(layer_num)
                elif l_attribute[1] == 4 and layer_num in sel_layer_nums and  param_budget_left_attn > 0:
                    param_budget_left_attn -= (layer.in_features * below_neurons +1)*2
                elif l_attribute[1] == 2:
                    d[str(layer_num)]-=1
                
                if l_attribute[1] in [0,1] and layer_num not in sel_layer_nums and  param_budget_left_attn > 0:
                    param_budget_left_attn - = (layer.in_features * v + v)*4 #four since we add neurons to q and k for each selected q or k
                    sel_layer_nums.append(layer_num)
                elif  l_attribute[1] in [0,1] and layer_num in sel_layer_nums and  param_budget_left_attn > 0:
                    param_budget_left_attn - = (layer.in_features +1)*4
                elif l_attribute[1] in [0,1]:
                    d[str(layer_num)]-=1
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
        if la[int(k)][1]==0:
            sel_layer_data.append([k, l[int(k)], la[int(k)], v])
            nl = qk_blocks(la[int(k)][1])
            sel_layer_data.append([nl, l[int(nl)], la[int(nl)], v])
        elif la[int(k)][1]==1:
            sel_layer_data.append([k, l[int(k)], la[int(k)], v])
            nl = qk_blocks(la[int(k)][0])
            sel_layer_data.append([nl, l[int(nl)], la[int(nl)], v])
        elif la[int(k)][1] == 5:
            sel_layer_data.append([k, l[int(k)], la[int(k)], v])
            nl = proj_fc2[int(k)][0]
            sel_layer_data.append([nl, l[int(nl)], la[int(nl)], v])

        else:      
            sel_layer_data.append([k, l[int(k)], la[int(k)], v])
        c += 1
    del eigs, flat_eig, flat_eig_layer
    torch.cuda.empty_cache()

    return sel_layer_data, neg_index_dic,qk_blocks
            
def ret_growth_model(block, num):
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
    elif num == 5:
        return block.mlp.fc2


def assign_model(model, block, num, gb):
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


def split_nodewise(
    model,
    optimizer,
    param_budget,
    epoch,
    width =False,
    percent=20,
    warmup=0,
    act_on=True,
    width = False,
    sel_layers_attn=None,
    sel_layers_data=None,
    neg_index_dic=None,
):
    start = time.time()
    if sel_layers_attn == None and sel_layers_data == None and neg_index_dic == None:
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
        if width == True:
             if growth_block ==0 :
                model, optimizer = update_qk_model_width(model, optimizer, layer_data, neg_index_dic, act_on, True)
            
            if growth_block==1:
                model, optimizer = update_qk_model_width(model, optimizer, layer_data, neg_index_dic, act_on, False)

            if growth_block == 2:
                if len(choices) % 6 != 0:
                    for i in range((len(choices) % 6)):
                        choices.pop(-1)
                model, layer, optimizer = create_width_growth(
                    model, optimizer, layer, choices, block, growth_block, act_on, opposite=True, zeros=False
                )

            if growth_block in [3,4,5]:
                model, layer, optimizer = create_width_growth(
                    model, optimizer, layer, choices, block, growth_block, act_on, opposite=True
                )
           
            
        
        else:
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


def update_qk_model_width(model, optimizer, data, neg_index_dic, act_on, opposite):
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


def create_width_growth(
    model, optimizer, layer, choices, block, growth_block, act_on, opposite=True, update=True, zeros=False
):
    '''
    Function that performs width growth operation.

    Input:
    model -> model
    optimizer -> optimizer
    layer -> layer to grow width
    choices -> list of selected neuron positions
    block -> current transformer block number
    growth_block -> current growthblock number
    act_on -> whether to activate neurons.
    opposite -> whether to have opposite neuron weights
    zeros -> whether to have zero neuron weights for child layers.

    Output:
    model -> Model updated with children of selected layers
    layer -> new_layer after growing width
    optimizer -> updated optimizer
    '''
    layer, optimizer = create_layer_width(layer, choices, optimizer, opposite=opposite)
    if growth_block !=5:
        next_block = ret_growth_model(model.blocks[block], growth_block + 1)
        layers_next = get_all_linear_layers(next_block, typ="list")
        if update == True:
            layers_next, optimizer = update_next_layer_weights(layers_next, choices, optimizer, zeros)
        layers_next = {str(i): layers_next[i] for i in range(len(layers_next))}
        gb = GrowthModel(layers_next, next_block.architecture_array, act_on)
        model = assign_model(model, block, growth_block + 1, gb)
    elif growth_block ==5:
        for ln in range(3):
            next_block = ret_growth_model(model.blocks[block+1], ln)
            layers_next = get_all_linear_layers(next_block, typ="list")
            if update == True:
                layers_next, optimizer = update_next_layer_weights(layers_next, choices, optimizer, zeros)
            layers_next = {str(i): layers_next[i] for i in range(len(layers_next))}
            gb = GrowthModel(layers_next, next_block.architecture_array, act_on)
            model = assign_model(model, block, growth_block + 1, gb)
    return model, layer, optimizer


def update_next_layer_weights(layer_list, choices, optimizer, zeros=False):
    '''
    When updating a layer's width, the child layers of that layer should have increased inputs.
    This function takes a list of children of the width-grown layer and increases thier input.

    Inputs:
    Layer_list -> list of all child layers to a wdth grown layer.
    choices -> list of selected neuron positions
    optimizer -> to added new weight to optimizer
    zeros -> for V-> proj layer the weights has to be zero. If true those weights would be zero. Else would be initialized with random weights

    Outputs:
    new_layer_list -> list of updated children layers
    optimizer -> updated optimizer
    '''
    new_layer_list = []
    for layer in layer_list:
        # new Layer created with updated inputs
        new_layer = nn.Linear(layer.in_features + 2 * len(choices), layer.out_features)  # (12,10)
        # weight is transposed so that we have rows as weights for each input
        weight = layer.weight.data.T  # (10,10)
        # A random weight of chosen weight dimensions.
        # No issue because, we have positve and negative weighted neurons as input so it would become zero.
        chosen_weights = torch.rand(weight[choices, :].shape).to(weight.device)  # (1,10)
        # Concatenating two equal copies for weights so they get cancelled when multiplied with positive and negative weights.
        new_weight = torch.cat([weight, chosen_weights, chosen_weights], axis=0)
        new_weight = new_weight.T  # (10,12)
        new_weight.requires_grad = True
        if zeros == True:
            # if zeros == True we would replace random weights with 0 weights.
            new_weight = torch.zeros(new_weight.shape, requires_grad=True, device=new_weight.device)
        # Updating  parameters and updating optimizer
        new_layer.weight = nn.Parameter(new_weight)
        new_layer.weight.requires_grad = True
        new_layer.bias = layer.bias
        new_layer.bias.requires_grad = True
        new_layer_list.append(new_layer)
        optimizer.param_groups[0]["params"].append(new_layer.bias)
        optimizer.param_groups[1]["params"].append(new_layer.weight)
    return new_layer_list, optimizer


def create_layer_width(layer, choices, optimizer, reduction_factor=0.2, opposite=True):
    '''
    Creates a New Layer with increased width.
    Inputs:
        layer -> Layer which we want to inc. depth
        choices -> List of selected neuron positions to increase depth.
        optimizer -> add the new weights and bias to the optimizer.
        reduction_factor -> multiplicative factor applied to new weights to prevent same a neuron copy.
        opposite weights -> toggled when a mixture of positive and negative weights are needed
    Outputs:
        new_layer -> newly created layer
        optimizer -> Updated Optimizer
    '''
    # New number of neurons = number of old neurons + 2* number of selected neurons to grow
    new_width = layer.out_features + 2 * len(choices)
    # creating new layer
    new_layer = nn.Linear(layer.in_features, new_width)
    # New Weights (positively weighted). multiplied with reduction factor to prevent same copy of selected neurons.
    weight_p = layer.weight.data[choices, :] * reduction_factor
    # New Weights (negatively weighted if opposite= True). multiplied with reduction factor to prevent same copy of selected neurons.
    if opposite == True:
        weight_n = layer.weight.data[choices, :] * reduction_factor * -1
        bias_n = layer.bias.data[choices] * reduction_factor * -1
    else:
        weight_n = layer.weight.data[choices, :] * reduction_factor
        bias_n = layer.bias.data[choices] * reduction_factor
    # Appending both weights , similar operation for bias
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


def create_new_layer_new(layer, choices, optimizer, reduction_factor=0.2):
    '''
    Creates a New Layer for depth addition. New Layer contains Equal Positive Weights and Negative weights with a reduction factor.
    (PW = reduction_factor * W, NW = -1 * PW)
    Inputs:
        layer -> Layer which we want to inc. depth
        choices -> List of selected neuron positions to increase depth.
        optimizer -> add the new weights and bias to the optimizer.
        reduction_factor -> multiplicative factor applied to new positive and negative weights to prevent same a neuron copy.

    Outputs:
        new_layer -> newly created layer
        optimizer -> Updated Optimizer
    '''
    # A single new layer for both positive and negative weights input -> same input as previous layer. output -> num(positve+negative weights)
    new_layer = nn.Linear(layer.in_features, 2 * len(choices))
    # Layer -> Weight (nn.Paramer) -> Tensor (nn.Parameter.data)
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
    # weight = nn.Parameter(Tensor)
    new_layer.weight = nn.Parameter(weight_f)
    # Double checking that requires_grad=True
    new_layer.weight.requires_grad = True
    bias_f.requires_grad = True
    new_layer.bias = nn.Parameter(bias_f)
    new_layer.bias.requires_grad = True
    # Append New Parameters to Param Group of optimizer. The bias has no weight decay(0) and weight has weight decay(1)
    optimizer.param_groups[0]["params"].append(new_layer.bias)
    optimizer.param_groups[1]["params"].append(new_layer.weight)
    return new_layer, optimizer


def calculate_eig(model, epoch):
    '''
    Calculate eigenvalues and plot them.Used in main.py
    '''
    l, la = get_all_linear_layers_transformer(model)
    eigs = calc_all_eigs(l)

    os.makedirs(f"eig/{epoch}", exist_ok=True)
    plot_eig(eigs, epoch)


def plot_eig(eig, epoch):
    '''
    Plot Minimum eigenvalues for an epoch
    '''
    for k in range(len(eig)):
        y = eig[k]
        cn = 0
        for aa in y:
            if aa < 0:
                cn += 1
        x = [i for i in range(len(y))]
        plt.scatter(x, y)
        plt.title(f"Layer {k} count_neg ={cn}")
        plt.savefig(f"eig/{epoch}/{k}.jpg")
        plt.clf()


def remove_garbage(model):
    '''
    Remove not needed gradients and activaions
    '''
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
