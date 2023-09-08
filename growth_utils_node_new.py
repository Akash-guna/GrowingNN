import GPUtil
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from GrowthNew import GrowthModel,GrowthBlock
from regression_utils_add import return_layers,return_layers_data
from torch.multiprocessing import Pool, Process, set_start_method
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
     set_start_method('spawn')
except RuntimeError:
    pass
from joblib import Parallel,delayed
import time
import datetime
import gc
def convert(n):
    return str(datetime.timedelta(seconds = n))
BATCH_SIZE=128

def get_all_linear_layers(model,typ ="dict"):
    '''
    Gets all Linear layers in the model.
    Input: Model
    Output: A List of all Linear Layers in the model
    '''
    if typ=="dict":
        children = [i for i in model.children()]
        linear_layers ={}
        l_c=0
        c=0
        l=len(children)

        while c<l:
            grandchildren = [i for i in children[c].children()]
            if grandchildren == []:
                if isinstance(children[c],nn.Linear):
                    linear_layers[str(l_c)]=children[c]
                    l_c+=1

            else:
                children+=grandchildren
                l = len(children)
            c+=1
        return linear_layers
    else:
        children = [i for i in model.children()]
        linear_layers =[]
        l_c=0
        c=0
        l=len(children)

        while c<l:
            grandchildren = [i for i in children[c].children()]
            if grandchildren == []:
                if isinstance(children[c],nn.Linear):
                    linear_layers.append(children[c])
                    l_c+=1

            else:
                children+=grandchildren
                l = len(children)
            c+=1
        return linear_layers
#linear_array=get_all_linear_layers(model)

def ret_new_weights(weights,choices):
  new_weights=[]
  for c in choices:

    new_weights.append(weights[c].tolist())
  new_weights = torch.Tensor(new_weights)
  old_weights= weights.tolist()
  for c in sorted(choices,reverse=True):
    #print(c)
    old_weights.pop(c)
  old_weights = torch.Tensor(old_weights)
  return torch.Tensor(new_weights),torch.Tensor(old_weights)

def ret_new_bias(bias,choices):
  new_bias=[]
  for c in choices:

    new_bias.append(bias[c].tolist())
  new_bias = torch.Tensor(new_bias)
  old_bias= bias.tolist()
  for c in sorted(choices,reverse=True):
    #print(c)
    old_bias.pop(c)
  old_bias = torch.Tensor(old_bias)
  return torch.Tensor(new_bias),torch.Tensor(old_bias)

def split_matrix(gradient,weight=None):
    def second_order_derivative():
        return torch.ones(gradient.shape)
    #print(gradient.view(1,-1).shape,second_order_derivative().view(-1,1).shape)
    sm= gradient.view(1,-1).cpu() *second_order_derivative().view(-1,1)
    return sm

def calculate_min_eig(gradient):
    try:
      splitting = split_matrix(gradient)
      #print(splitting.shape)
      eig = splitting.eig()
      min =  torch.min(eig.eigenvalues.cpu().double())
      del eig,splitting
      gc.collect()
      torch.cuda.empty_cache()
    except:
      min=0
    return min

def calculate_min_eig_p(i,gradient):
    g = gradient[i]
    splitting = split_matrix(g)
    #print(splitting.shape)
    eig = splitting.eig()
    min =torch.min(eig.eigenvalues.cpu().double())
    del eig
    gc.collect()
    return min

def calculate_max_layer(layers):
    cur_min_eig=0
    sel_layer=None
    all_layer_neg_index=[]
    for i,layer in enumerate(layers):
        gradient = layer.weight.grad
        layer_min_eig=None
        layer_neg_index=[]
        for n in range(gradient.shape[0]):
            g=gradient[n]
            min_eig = calculate_min_eig(g)
            if min_eig<0:
                layer_neg_index.append([n,min_eig])
            if layer_min_eig ==None:
                layer_min_eig = min_eig
            elif layer_min_eig > min_eig:
                layer_min_eig = min_eig
            else:
                pass
        #print(layer_min_eig)
        if layer_min_eig.tolist()< cur_min_eig:
            cur_min_eig = layer_min_eig
            sel_layer = i
        all_layer_neg_index.append(layer_neg_index)
    if sel_layer != None:
        return sel_layer,layers[sel_layer],sorted(all_layer_neg_index[sel_layer],key = lambda tuple: tuple[1].tolist())
    else:
        return None,None,None

def c_index(nodes,choices):
    chosen_index=[0]*nodes
    for c in choices:
        chosen_index[c]=1
    return chosen_index

def train(net,loaders,input_size=784,EPOCHS=5,loss_func = nn.CrossEntropyLoss(),opt =optim.Adam,evaluate=True,verbose=True ):
    if verbose==True:
        print(net)
#     loss_func = nn.CrossEntropyLoss()
    optimizer = opt(net.parameters(), lr =0.001)
    loss_arr=[]
    train_acc_arr=[]
    test_acc_arr=[]

    net = net
    for epoch in range(EPOCHS):
        for data in loaders["train"]:
            X, y = data
            if X.shape[0] !=BATCH_SIZE:
                break
            #print(X.shape,y.shape)
            net.zero_grad()
            output = net(X.view(-1, input_size))
            #print(output.shape)
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()
            #print(net.layer_dict['0'].weight.grad.shape)
        correct = 0
        total = 0
        if evaluate ==True:
            net.eval()
            with torch.no_grad():
                for data in loaders["train"]:
                        X, y = data
                        output = net(X.view(-1, input_size))

                        for idx, i in enumerate(output):
                            if torch.argmax(i) == y[idx]:
                                correct += 1
                            total +=1


            correct_t = 0
            total_t = 0

            with torch.no_grad():
                for data in loaders["test"]:
                        X, y = data
                        output = net(X.view(-1, input_size))

                        for idx, i in enumerate(output):
                            if torch.argmax(i) == y[idx]:
                                correct_t += 1
                            total_t +=1
            if verbose==True:
                print(f"Epoch ={epoch} Loss = {loss}, Train Acc={round(correct/total, 3)} test acc ={round(correct_t/total_t, 3)} ")
            net.train()
            train_acc_arr.append(round(correct/total, 3))
            test_acc_arr.append(round(correct_t/total_t, 3))
        else:
            net.eval()
            loss_test=[]
            with torch.no_grad():
                for data in loaders["test"]:
                        X, y = data
                        output = net(X.view(-1, input_size))
                        loss_test.append(loss_func(output,y))
            loss_test = torch.Tensor(loss_test)
            avg_test_loss = torch.mean(loss_test)
            if verbose==True:
                print(f"Epoch ={epoch} Loss = {loss}, Test Loss = {avg_test_loss} ")
        loss_arr.append(loss)

    return (loss_arr,train_acc_arr,test_acc_arr),net

def load_data(func = datasets.MNIST):
    train_data = func(
    root = 'data',
    train = True,
    transform = T.Compose([T.ToTensor()]),
    download = True,
    )
    test_data = func(
    root = 'data',
    train = False,
    transform =T.Compose([T.ToTensor()]),
    )
    loaders = {
        'train' : torch.utils.data.DataLoader(train_data,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=1),

        'test'  : torch.utils.data.DataLoader(test_data,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=1),
    }
    return loaders

def permutation(choices,out_features):
    not_chosen = [i for i in range(out_features) if i not in choices]
    total_numbers =[i for i in range(out_features)]
    permutation=[]
    nc=0
    c=0
    for t in total_numbers:
        if t in choices:
            permutation.append(len(not_chosen)+c)
            c+=1
        else:
            permutation.append(nc)
            nc+=1
    return torch.Tensor(permutation).long()

def GrowthStep(model,follow_neg=True,nodes=5,act_on=True):
    layers =get_all_linear_layers(model,"list")
    s_layer,layer,neg_index=calculate_max_layer(layers)
    if follow_neg==False:
        neg_index = neg_index[:nodes]
    if len(neg_index) > layer.out_features/2:
        neg_index = neg_index[:len(neg_index)//2]

    choices=[n[0] for n in neg_index]
    choices.sort()

    #print(choices)
    chosen_index= c_index(layer.out_features,choices)
    perm = permutation(choices,layer.out_features)
    #print(s_layer)
    layer=layers.pop(s_layer)
    nw,ow = ret_new_weights(layer.weight,choices)
    nb,ob = ret_new_bias(layer.bias,choices)
    ow.requires_grad = True
    ob.requires_grad = True
    old_layer = nn.Linear(layer.in_features,layer.out_features-len(neg_index))
    #print(ow.shape,old_layer)
    old_layer.weight= nn.Parameter(ow)
    old_layer.bias = nn.Parameter(ob)
    #print(ow.requires_grad,ob.requires_grad)
    new_layer,feature_bottleneck,old_split,skip_fc,int_metrics = return_layers(layer,choices,num_nodes= len(neg_index)+10,samp_size=10000,reg_epochs=100,verbose=True)
    #new_layer,old_split,int_metrics = return_layers_data(loaders,model,layer,choices,num_nodes= len(neg_index)+10,samp_size=10000,reg_epochs=100,verbose=True)
    layers = layers[:s_layer]+[old_layer,new_layer,feature_bottleneck,old_split,skip_fc]+layers[s_layer:]
    layers= {str(i): layers[i] for i in range(len(layers))}
    #print(layers)
    _,_,architecture_array = return_arc_array(model.architecture_array,0,s_layer,perm)
    model1 = GrowthModel(layers,architecture_array,act_on)
    return model1, nw,ow,nb,ob,perm,int_metrics

def growth_wrapper(model):
    for i in range(len(model.blocks)):
        #print(f"Block {i} QKV {model.blocks[i].attn.qkv}")
        model.blocks[i].attn.qkv = GrowthBlock(model.blocks[i].attn.qkv)
        #print(f"Block {i} proj {model.blocks[i].attn.proj}")
        print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 1")
        model.blocks[i].attn.proj = GrowthBlock(model.blocks[i].attn.proj)
        print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 2")
        #print(f"Block {i} MLP fc1 {model.blocks[i].mlp.fc1}")
        model.blocks[i].mlp.fc1 = GrowthBlock(model.blocks[i].mlp.fc1)
        print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 3")
        #print(f"Block {i} MLP fc2 {model.blocks[i].mlp.fc2}")
        model.blocks[i].mlp.fc2 = GrowthBlock(model.blocks[i].mlp.fc2)
        print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 4")
    return model

def split(model):
    for i in range(len(model.blocks)):
        #print(f"Block {i} QKV {model.blocks[i].attn.qkv}")
        #model.blocks[i].attn.qkv = GrowthStep(model.blocks[i].attn.qkv,act_on=False)[0].cuda()
        print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 1")
        #print(f"Block {i} proj {model.blocks[i].attn.proj}")
        #model.blocks[i].attn.proj = GrowthStep(model.blocks[i].attn.proj,act_on=False)[0].cuda()
        print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 2")
        #print(f"Block {i} MLP fc1 {model.blocks[i].mlp.fc1}")
        model.blocks[i].mlp.fc1 = GrowthStep(model.blocks[i].mlp.fc1,act_on=False)[0].cuda()
        print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 3")
        #print(f"Block {i} MLP fc2 {model.blocks[i].mlp.fc2}")
        model.blocks[i].mlp.fc2 = GrowthStep(model.blocks[i].mlp.fc2,act_on=False)[0].cuda()
        print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 4")

    return model

#def split_one(model,i):
    #print(f"Block {i} QKV {model.blocks[i].attn.qkv}")
    #model.blocks[i].attn.qkv = GrowthStep(model.blocks[i].attn.qkv,act_on=False)[0].cuda()
    #print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 1")
    #print(f"Block {i} proj {model.blocks[i].attn.proj}")
    #model.blocks[i].attn.proj = GrowthStep(model.blocks[i].attn.proj,act_on=False)[0].cuda()
    #print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 2")
    #print(f"Block {i} MLP fc1 {model.blocks[i].mlp.fc1}")
    #model.blocks[i].mlp.fc1 = GrowthStep(model.blocks[i].mlp.fc1,act_on=False)[0].cuda()
    #print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 3")
    #print(f"Block {i} MLP fc2 {model.blocks[i].mlp.fc2}")
    #model.blocks[i].mlp.fc2 = GrowthStep(model.blocks[i].mlp.fc2,act_on=False)[0].cuda()
    #print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 4")

    #return model

def return_arc_array(a_array,i_num,sel_layer,positional):
    def create_numbered_arc_array(arc_array,init_num):
        name_array=[]
        arc_new_arr=[]
        for i in range(len(arc_array)):
            if arc_array[i] ==0:
                name_array.append(init_num)
                if init_num != sel_layer:
                    arc_new_arr.append(0)
                else:
                    arc_new_arr.append([[0,0],positional])
                init_num+=1
            else:
                named_child_array,init_num,narr = create_numbered_arc_array(arc_array[i][0],init_num)
                #print(arc_array[i][1])
                name_array.append([named_child_array,arc_array[i][1]])
                arc_new_arr.append([narr,arc_array[i][1]])
        return name_array,init_num,arc_new_arr
    return create_numbered_arc_array(a_array,i_num)

def get_all_linear_layers_transformer(model):
    l=[]
    l_att =[]
    for i in range(len(model.blocks)):
        #print(f"Block {i} QKV {model.blocks[i].attn.qkv}")
        a=get_all_linear_layers(model.blocks[i].attn.qkv,typ='list')
        l+=a
        a_att = [[i,0,j] for j in range(len(a))]
        l_att+=a_att
        b=get_all_linear_layers(model.blocks[i].attn.proj,typ='list')
        l+=b
        b_att = [[i,1,j] for j in range(len(b))]
        l_att+=b_att
        c=get_all_linear_layers(model.blocks[i].mlp.fc1,typ='list')
        l+=c
        c_att = [[i,2,j] for j in range(len(c))]
        l_att+=c_att
        d=get_all_linear_layers(model.blocks[i].mlp.fc2,typ='list')
        l+=d
        d_att = [[i,3,j] for j in range(len(d))]
        l_att+=d_att
    return l,l_att

def calc_all_eigs(layers):
    eigs=[]
    for i,layer in enumerate(layers):
        print("Before Calculation of eig")
        GPUtil.showUtilization()

        gradient = layer.weight.grad.cpu().detach().clone()
        #print(gradient.shape)
        #min_eigs=[]
        # for n in range(gradient.shape[0]):
        #     g=gradient[n]
        #     #if n==0:
        #         #print(g.shape)
        #     min_eigs.append(calculate_min_eig(g))
        s=gradient.shape[0]
        min_eigs = Parallel(n_jobs =6)(delayed(calculate_min_eig)(gradient[i]) for i in range(s))
        # gradient_list= [gradient[i] for i in range(s)]
        # count=0
        # while count<s:

        #     multi_pool = Pool(processes=5)
        #     min_eigs = multi_pool.map(calculate_min_eig,gradient_list[count:count])
        #     multi_pool.close()
        #     multi_pool.join()
        #     if count+10<s:
        #         count+=5
        #     else:


        #print(min_eigs[-1].shape)
        eigs.append(min_eigs)
        print("After Calculation of eig")
        GPUtil.showUtilization()
        del gradient
        gc.collect()
        torch.cuda.empty_cache()
        #print(len(min_eigs))
    return eigs

def ret_flattened(eig):
    flat_eig=[]
    flat_eig_layer=[]
    for i,e in enumerate(eig):
        flat_eig+=e
        flat_eig_layer+= [i for j in range(len(e))]
    return flat_eig,flat_eig_layer
# def layer_negative(flat_eigs,flat_eig_layer):
#     neg_index_dic={}
#     for i,(eig,layer) in enumerate(zip(flat_eigs,flat_eig_layer)):
#         if eig<0:
#             try:
#                 neg_index_dic[str(layer)].append(i)
#             except:
#                 neg_index_dic[str(layer)]=[i]
#     return neg_index_dic
def layer_negative(eigs):
    neg_index_dic = {}
    neg_eig_dic ={}
    for layer, eig in enumerate(eigs):
        for j, e in enumerate(eig):
            if e < 0:
                try:
                    neg_index_dic[str(layer)].append(j)
                    neg_eig_dic[str(layer)].append(e)
                except:
                    neg_index_dic[str(layer)] = [j]
                    neg_eig_dic[str(layer)]=[e]

    for k in neg_index_dic.keys():
        sort_pos = np.argsort(np.array(neg_eig_dic[k]))
        neg_index_dic[k] = np.array(neg_index_dic[k])[sort_pos]
    return neg_index_dic

def ret_eig_plot_data(model,percent=20):
    l,la=get_all_linear_layers_transformer(model)
    eigs = calc_all_eigs(l)
    flat_eig,flat_eig_layer = ret_flattened(eigs)
    rank = torch.Tensor(flat_eig).argsort()
    limit = (rank.shape[0]/100)*percent
    eig={}
    for i in range(int(limit)):
        pos = rank[i]
        if flat_eig[pos]>0:
            continue
        try:
            eig[str(la[flat_eig_layer[pos]][0])].append(flat_eig[pos])
        except:
            eig[str(la[flat_eig_layer[pos]][0])]=[flat_eig[pos]]
    print(eig.keys())
    plot_eig(eig)

def find_split_layers(model,percent=20,layer_percent=20,num_layers=5):
    l,la=get_all_linear_layers_transformer(model)
    eigs = calc_all_eigs(l)
    flat_eig,flat_eig_layer = ret_flattened(eigs)
    neg_index_dic = layer_negative(eigs)
    rank = torch.Tensor(flat_eig).argsort()
    limit = (rank.shape[0]/100)*percent
    d={}
    for i in range(int(limit)):
        pos = rank[i]
        if flat_eig[pos]>0:
            continue
        layer_num = flat_eig_layer[pos]
        try:
            d[str(layer_num)]+=1
        except:
            d[str(layer_num)]=1
    sel_layer_data=[]
    sorted_keys = sorted(d,reverse=True)
    c=0                                                                                                                                                                                                                                                                                                                                                                                   
    for k in sorted_keys:
        v = d[k]
        layer = l[int(k)]
        if v >= layer.out_features * (layer_percent/100):
            sel_layer_data.append([k,l[int(k)],la[int(k)],v])
        c+=1
        if len(sel_layer_data)==num_layers:
            break
    #print(sel_layer_data)
    return sel_layer_data,neg_index_dic

def find_split_layers_new(model,percent=20,layer_percent=20,num_layers=5):
    l,la=get_all_linear_layers_transformer(model)
    eigs = calc_all_eigs(l)
    flat_eig,flat_eig_layer = ret_flattened(eigs)
    neg_index_dic = layer_negative(eigs)
    rank = torch.Tensor(flat_eig).argsort()
    limit = (rank.shape[0]/100)*percent
    d={}
    sel_layer_nums=[]
    for i in range(int(limit)):
        pos = rank[i]
        if flat_eig[pos]>0:
            continue
        layer_num = flat_eig_layer[pos]
        try:
            d[str(layer_num)]+=1
        except:
            d[str(layer_num)]=1
        v = d[str(layer_num)]
        layer = l[layer_num]
        if v >= layer.out_features * (layer_percent/100) and v > 60 and layer_num not in sel_layer_nums:
            sel_layer_nums.append(layer_num)
        if len(sel_layer_nums)==num_layers:
            break
    sel_layer_data=[]
    sorted_keys = sorted(d,reverse=True)
    c=0                                                                                                                                                                                                                                                                                                                                                                                   
    for layer_num in sel_layer_nums:
        k = str(layer_num)
        v = d[k]
        sel_layer_data.append([k,l[int(k)],la[int(k)],v])
        c+=1
    print(c, sel_layer_nums)
    #print(sel_layer_data)
    return sel_layer_data,neg_index_dic
def find_split_layers_param(model, param_budget, percent=20):
    l, la = get_all_linear_layers_transformer(model)
    print("Inside find Split after getlinear")
    GPUtil.showUtilization()

    eigs = calc_all_eigs(l)
    print("Inside find Split after eig")
    GPUtil.showUtilization()

    flat_eig, flat_eig_layer = ret_flattened(eigs)
    print("Inside find Split after eig flatten")
    GPUtil.showUtilization()

    neg_index_dic = layer_negative(eigs)
    print("Inside find Split afterlayer_negative")
    GPUtil.showUtilization()

    rank = torch.Tensor(flat_eig).argsort()
    limit = (rank.shape[0] / 100) * percent
    print("Inside find Split after rank,limit")
    GPUtil.showUtilization()

    param_budget_left = param_budget
    d = {}
    sel_layer_nums = []

    for i in range(int(limit)):
        pos = rank[i]
        if flat_eig[pos] > 0:
            continue
        layer_num = flat_eig_layer[pos]
        # Number of Parameters caused by 1 neuron  = No. of Previous Layer Neuro                                                                                                             ns * 1 + 1. The +1 is for bias
        #param_budget_left -= l[int(layer_num)].in_features + 1

        try:
            d[str(layer_num)] += 1
        except:
            d[str(layer_num)] = 1
        v = d[str(layer_num)]
        layer =l[layer_num]
        if v >= layer.out_features *0.5:
            if layer_num not in sel_layer_nums:
                sel_layer_nums.append(layer_num)
                param_budget_left -= (l[int(layer_num)].in_features + 1)*v
            else:
                param_budget_left -= l[int(layer_num)].in_features + 1
        if param_budget_left <= 0:
            break

    #sel_layer_nums = [int(i) for i in list(d.keys()) if d[i]>20]
    sel_layer_data = []
    print("After sel layer data formation")
    GPUtil.showUtilization()

    c = 0
    for layer_num in sel_layer_nums:
        k = str(layer_num)
        v = d[k]
        sel_layer_data.append([k, l[int(k)], la[int(k)], v])
        c += 1
    print(c, sel_layer_nums)
    # print(sel_layer_data)
    del eigs,flat_eig,flat_eig_layer
    torch.cuda.empty_cache()
    return sel_layer_data, neg_index_dic


def ret_growth_model(block, num):
    if num==0:
        return block.attn.qkv
    elif num==1:
        return block.attn.proj
    elif num==2:
        return block.mlp.fc1
    else:
        return block.mlp.fc2

def assign_model(model,block,num,gb):
    if num==0:
        model.blocks[block].attn.qkv =gb
    elif num==1:
        model.blocks[block].attn.proj = gb
    elif num==2:
        model.blocks[block].mlp.fc1 = gb
    else:
        model.blocks[block].mlp.fc2 = gb
    return model
def not_trainable(gm):
    keys = gm.layer_dict.keys()
    for k in keys:
        for p in gm.layer_dict[k].parameters():
            p.requires_grad=False
    return gm

def trainable(gm):
    keys = gm.layer_dict.keys()
    for k in keys:
        for p in gm.layer_dict[k].parameters():
            p.requires_grad=True
    return gm
def set_all_not_trainable(model):
    for b in range(len(model.blocks)):
        model.blocks[b].attn.qkv = not_trainable(model.blocks[b].attn.qkv)
        model.blocks[b].attn.proj = not_trainable(model.blocks[b].attn.proj)
        model.blocks[b].mlp.fc1 = not_trainable(model.blocks[b].mlp.fc1)
        model.blocks[b].mlp.fc2 = not_trainable(model.blocks[b].mlp.fc2)
    return model
def set_all_trainable(model):
    for b in range(len(model.blocks)):
        model.blocks[b].attn.qkv = trainable(model.blocks[b].attn.qkv)
        model.blocks[b].attn.proj = trainable(model.blocks[b].attn.proj)
        model.blocks[b].mlp.fc1 = trainable(model.blocks[b].mlp.fc1)
        model.blocks[b].mlp.fc2 = trainable(model.blocks[b].mlp.fc2)
    return model
def print_all_requires_grad(model):
    all_linear = get_all_linear_layers_transformer(model)[0]
    rg = [[param.requires_grad for param in linear.parameters()]for linear in all_linear]
    print(rg)
def split_nodewise(model, param_budget,percent=20, warmup=0, act_on=True):
    start = time.time()
    print("Inside Split_Nodewise")
    GPUtil.showUtilization()

    if warmup != 0:
        set_all_not_trainable(model)
        print(model)
    print_all_requires_grad(model)
    sel_layers_data, neg_index_dic = find_split_layers_param(model, param_budget,percent)
    print("After find_layers")
    GPUtil.showUtilization()

    cp1 = time.time()
    sum_reg = 0
    sum_max = 0
    l_neg = []
    for layer_data in sel_layers_data:
        s_max = time.time()
        # _,_,neg_index=calculate_max_layer([layer_data[1]])
        neg_index = neg_index_dic[str(layer_data[0])]
        e_max = time.time()
        sum_max += e_max - s_max
        if len(neg_index) > layer_data[-1]:
            neg_index = neg_index[: layer_data[-1]]
        l_neg.append(len(neg_index))

        block = layer_data[2][0]
        growth_block = layer_data[2][1]
        block_model = ret_growth_model(model.blocks[block], growth_block)

        # print(block_model)
        layers = get_all_linear_layers(block_model, typ="list")
        print(layers)
        layer = layer_data[1]
        choices = [n for n in neg_index]
        choices.sort()
        perm = permutation(choices, layer_data[1].out_features)
        #print(layer_data)
        s_layer = None
        for i, l in enumerate(layers):
            if l == layer_data[1]:
                layers.pop(i)
                s_layer = i
                break
        # print(s_layer)
        new_layer = create_new_layer(layer, choices)
        #print("After Layer Creation")
        #GPUtil.showUtilization()

        # nw,ow = ret_new_weights(layer.weight,choices)
        # nb,ob = ret_new_bias(layer.bias,choices)
        # ow.requires_grad = True
        # ob.requires_grad = True
        # old_layer = nn.Linear(layer.in_features,layer.out_features-len(neg_index))
        # #print(ow.shape,old_layer)
        # old_layer.weight= nn.Parameter(ow)
        # old_layer.bias = nn.Parameter(ob)
        # print(ow.requires_grad,ob.requires_grad)
        s_reg = time.time()
        # new_layer,feature_bottleneck,old_split,skip_fc,int_metrics = return_layers(layer,choices,num_nodes= len(neg_index)+10,samp_size=5000,reg_epochs=100,verbose=False)
        # new_layer,old_split,int_metrics = return_layers_data(loaders,model,layer,layers_data[2],s_layer,choices,num_nodes= len(neg_index)+10,samp_size=10000,reg_epochs=100,verbose=True)

        e_reg = time.time()
        sum_reg += e_reg - s_reg
        #print(layers)
        # print(layers[:s_layer],layers[s_layer:])
        layers = layers[:s_layer] + [layer, new_layer] + layers[s_layer:]
        layers = {str(i): layers[i] for i in range(len(layers))}
        #print(layers)
        _, _, architecture_array = return_arc_array(block_model.architecture_array, 0, s_layer, choices)
        #print("Before Assign model")
        #GPUtil.showUtilization()
        model = assign_model(model, block, growth_block, GrowthModel(layers, architecture_array, act_on))
        print_all_requires_grad(model)
        cp2 = time.time()
        #print("End Split_Nodewise")
        #GPUtil.showUtilization()
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

#def create_new_layer(layer,choices,reduction_factor = 0.01):
#    new_layer = nn.Linear(layer.in_features,len(choices))
#    weight= torch.zeros(new_layer.weight.shape)
#    weight.requires_grad=True
#    new_layer.weight = nn.Parameter(weight)
#    bias = torch.zeros(new_layer.bias.shape)
#    bias.requires_grad=True
#    new_layer.bias = nn.Parameter(bias)
#    return new_layer
def create_new_layer(layer,choices,reduction_factor = 0.5):
    new_layer = nn.Linear(layer.in_features,len(choices))

    layer_weight = layer.weight.data
    layer_bias =layer.bias.data

    bias = layer_bias[choices]*0.5 * -1
    weight= layer_weight[choices,:]*0.5 * -1

    weight.requires_grad=True
    new_layer.weight = nn.Parameter(weight)
    new_layer.weight.requires_grad=True


    bias.requires_grad=True
    new_layer.bias = nn.Parameter(bias)
    new_layer.bias.requires_grad=True

    layer_weight[choices,:] = layer_weight[choices,:]* (0.5)
    layer.weight = nn.Parameter(layer_weight)
    layer.weight.requires_grad =True

    layer_bias[choices] = layer_bias[choices]*(0.5)
    layer.bias = nn.Parameter(layer_bias)
    layer.bias.requires_grad=True
    print(new_layer.weight)
    print(layer.weight)
    print(new_layer.weight+layer.weight[choices,:])
    return layer,new_layer

def plot_eig(eig):
    for k in eig.keys():
        y= eig[k]
        x = [i for i in range(len(y))]
        plt.scatter(x,y)
        plt.plot(x,[0 for i in range(1000)],linewidth=3,color='red')
        plt.title(f"Block {k}")
        plt.savefig(f"eig/{k}.jpg")

def remove_garbage(model):
    l, la = get_all_linear_layers_transformer(model)
    for i,layer in enumerate(l):
        print("Before Remove garbage")
        GPUtil.showUtilization()
        gradient = layer.weight.grad.cpu().detach().clone()
        del gradient
        gc.collect()
        torch.cuda.empty_cache()
        print("After Remove Garbage")
        GPUtil.showUtilization()
