import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from Growth import GrowthModel,GrowthBlock
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
    
def get_all_linear_layers_transformer(model):
    '''
    Get All Linear Layers from a model as a List
    '''
    l=[]
    l_att =[]
    for i in range(len(model.blocks)):
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

def ret_new_weights(weights,choices):
  '''
  Returning New Weights for Old Layer after splitting
  input:
  weight: Previous Weight Matrix
  choices: Selected Neurons of the layer to split

  Output:
  new_weight: New Weights for Old Split (Not Used)
  old_weight: New Weights for Old Layer (Used)
  '''
  new_weights=[]
  for c in choices:

    new_weights.append(weights[c].tolist())
  new_weights = torch.Tensor(new_weights)
  old_weights= weights.tolist()
  for c in sorted(choices,reverse=True):
    old_weights.pop(c)
  old_weights = torch.Tensor(old_weights)
  return torch.Tensor(new_weights),torch.Tensor(old_weights)

def ret_new_bias(bias,choices):
  '''
  Returning New bias for Old Layer after splitting
  input:
  bias: Previous bias Matrix
  choices: Selected Neurons of the layer to split

  Output:
  new_bias: New biass for Old Split (Not Used)
  old_bias: New biass for Old Layer (Used)
  '''
  new_bias=[]
  for c in choices:

    new_bias.append(bias[c].tolist())
  new_bias = torch.Tensor(new_bias)
  old_bias= bias.tolist()
  for c in sorted(choices,reverse=True):
    old_bias.pop(c)
  old_bias = torch.Tensor(old_bias)
  return torch.Tensor(new_bias),torch.Tensor(old_bias)

def split_matrix(gradient,weight=None):
    '''
    Creation of Splitting Matrix

    Input:
    gradient : gradient of weight
    Output:
    Splitting Matrix
    '''
    def second_order_derivative():
        '''
        second order derivative with respect to input = 1 for the gradient of a single neuron (Cross Verified with Official Implemetation)
        '''
        return torch.ones(gradient.shape)
    sm= gradient.view(1,-1).cpu() *second_order_derivative().view(-1,1)
    return sm

def calculate_min_eig(gradient):
    '''
    Function To calculate the minimum eigen value from the splitting matrix
    '''
    splitting = split_matrix(gradient)
    eig = splitting.eig()
    
    return torch.min(eig.eigenvalues.cpu().double())

def permutation(choices,out_features):
    '''
    Calculates the Permutation. 
    Eg a layer is split [1,3,5]-> not split [2,4]->split at one stage we need to concatenate [1,3,5,2,4]. 
    To order this we use a permutation [0,3,1,4,2]. 

    Input:
    choices : Chosen Neuron Indices of layer to split
    out_features: Number of Neurons in the layer

    Output:
    Permutation
    '''
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



def growth_wrapper(model):
    '''
    Wraps each Layer in a Deit Torch Model with a Growth Block  
    '''
    for i in range(len(model.blocks)):
        model.blocks[i].attn.qkv = GrowthBlock(model.blocks[i].attn.qkv)
        print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 1")
        model.blocks[i].attn.proj = GrowthBlock(model.blocks[i].attn.proj)
        print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 2")
        model.blocks[i].mlp.fc1 = GrowthBlock(model.blocks[i].mlp.fc1)
        print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 3")
        model.blocks[i].mlp.fc2 = GrowthBlock(model.blocks[i].mlp.fc2)
        print(f"Total Blocks: {len(model.blocks)} Block : {i} Count = 4")
    return model



def calc_all_eigs(layers):
    '''
    Calculate EigenValues Parallely for all layers
    Input :
    Layers: A List of all linear layers
    Output:
    eigs : A List of eigen values of each linear layer

    '''
    eigs=[]
    for i,layer in enumerate(layers):
        gradient = layer.weight.grad
        s=gradient.shape[0]
        min_eigs = Parallel(n_jobs =6)(delayed(calculate_min_eig)(gradient[i]) for i in range(s))
        eigs.append(min_eigs)
    return eigs

def ret_flattened(eig):
    '''
    Return Eigen Value as a 1-d array

    Input:
    eig : A list of eigen values for each layer

    Output:
    flat_eig : Flattened eigen value array
    flat_eig_layer : An array that maps each value in flag_eig to the layer it was calculated for (can be used to reconstruct eig from flat_eig)
    '''
    flat_eig=[]
    flat_eig_layer=[]
    for i,e in enumerate(eig):
        flat_eig+=e
        flat_eig_layer+= [i for j in range(len(e))]
    return flat_eig,flat_eig_layer

def layer_negative(eigs):
    '''
    Constructs a List of all Negative Eigen Values Position for each layer

    Input:
    eigs: eigen value list for each layer
    Output:
    neg_index : A Dictionary mapping a layer to negative eigen value positons
    '''
    neg_index_dic={}
    for layer,eig in enumerate(eigs):
        for j,e in enumerate(eig):
            if e<0:
                try:
                    neg_index_dic[str(layer)].append(j)
                except:         
                    neg_index_dic[str(layer)]=[j]
    return neg_index_dic


    


def find_split_layers_new(model,percent=20,layer_percent=20,num_layers=5):
    '''
    The Function to find which layers to split

    Input:
    model -> Model
    Percent -> Top Percent (P)
    Layer Percent -> Threshold for selecting Layers (K)
    num Layer -> Number of Layers to select

    Output:
    sel_layer_data -> Selected Layer Data  for each selected layer [ Layer Number , Layer, Layer Attribute, Number of Negative Eigen Values to Select ]
    negative_layer_dic -> the negative_layer_dic from layer_negative ()
    '''
    # Get All Linear Layers and Its Attributes
    l,la=get_all_linear_layers_transformer(model)
    # Calculate Eigen Values
    eigs = calc_all_eigs(l)
    # Flatten Eigen Values and get its layer mapping
    flat_eig,flat_eig_layer = ret_flattened(eigs)
    # Get all negative index positions per layer
    neg_index_dic = layer_negative(eigs)
    # Argsort the eigen values to find each eigen value's rank
    rank = torch.Tensor(flat_eig).argsort()
    #Find the limit given by P%
    limit = (rank.shape[0]/100)*percent
    d={}
    sel_layer_nums=[]

    for i in range(int(limit)):
        pos = rank[i]
        #If eigen value is positive ignoe
        if flat_eig[pos]>0:
            continue
        
        layer_num = flat_eig_layer[pos]
        try:
            # d contains count of negative eigenvlaues per layer
            d[str(layer_num)]+=1
        except:         
            d[str(layer_num)]=1
        v = d[str(layer_num)]
        layer = l[layer_num]
        # After increasing count we test whether the layer has enough eigenvalues to split if yes we add info to sel_layer_nums else we continue the process
        if v >= layer.out_features * (layer_percent/100) and layer_num not in sel_layer_nums:
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
    return sel_layer_data,neg_index_dic

def ret_growth_model(block, num):
    '''
    Util Function to return a layer from the given num
    '''
    if num==0:
        return block.attn.qkv
    elif num==1:
        return block.attn.proj
    elif num==2:
        return block.mlp.fc1
    else:
        return block.mlp.fc2

def assign_model(model,block,num,gb):
    '''
    Assigning a Growth Block to a given position

    '''
    if num==0:
        model.blocks[block].attn.qkv =gb
    elif num==1:
        model.blocks[block].attn.proj = gb
    elif num==2:
        model.blocks[block].mlp.fc1 = gb
    else:
        model.blocks[block].mlp.fc2 = gb
    return model


def split_nodewise(model,percent=20,layer_percent=20,num_layers=10,act_on=True):
    start = time.time()
    sel_layers_data,neg_index_dic=find_split_layers_new(model,percent,layer_percent,num_layers)
    cp1= time.time()
    sum_reg=0
    sum_max=0
    l_neg=[]
    for layer_data in sel_layers_data:
        s_max = time.time()
        neg_index = neg_index_dic[str(layer_data[0])]
        e_max = time.time()
        sum_max += e_max-s_max
        if len(neg_index)>layer_data[-1]:
            neg_index = neg_index[:layer_data[-1]]
        l_neg.append(len(neg_index))
      
        block = layer_data[2][0]
        growth_block =layer_data[2][1]
        block_model= ret_growth_model(model.blocks[block],growth_block)
        layers = get_all_linear_layers(block_model,typ="list")
        layer=layer_data[1]
        if len(neg_index) > layer.out_features/2:
            neg_index = neg_index[:len(neg_index)//2]
        choices=[n for n in neg_index]
        choices.sort()
        perm = permutation(choices,layer_data[1].out_features)
        s_layer=None
        for i,l in enumerate(layers):
            if l==layer_data[1]:
                layers.pop(i)
                s_layer=i
                break        
        nw,ow = ret_new_weights(layer.weight,choices)
        nb,ob = ret_new_bias(layer.bias,choices)
        ow.requires_grad = True
        ob.requires_grad = True
        old_layer = nn.Linear(layer.in_features,layer.out_features-len(neg_index))
        old_layer.weight= nn.Parameter(ow)
        old_layer.bias = nn.Parameter(ob)
        s_reg = time.time()
        new_layer,feature_bottleneck,old_split,skip_fc,int_metrics = return_layers(layer,choices,num_nodes= len(neg_index)+10,samp_size=5000,reg_epochs=100,verbose=False)
        e_reg = time.time()
        sum_reg+= (e_reg-s_reg)

        layers = layers[:s_layer]+[old_layer,new_layer,feature_bottleneck,old_split,skip_fc]+layers[s_layer:]
        layers= {str(i): layers[i] for i in range(len(layers))}
        _,_,architecture_array = return_arc_array(block_model.architecture_array,0,s_layer,perm)
        model = assign_model(model,block,growth_block,GrowthModel(layers,architecture_array,act_on))
        cp2 = time.time()

    print("Time for find splits :", convert(cp1-start))
    print("Total Regression Time:", convert(sum_reg))
    print("Average Regression Time:", convert(sum_reg/len(sel_layers_data)))
    print("Total Max Layer Time:", convert(sum_max))
    print("Total Number of Selected Layers",len(sel_layers_data))
    print("Average Max Layer Time:", convert(sum_max/len(sel_layers_data)))
    print("Misc Time:", convert((cp2-start)-(cp1-start)))
    print("Total Time:", convert(cp2-start))
    print("Negative Index Length:",l_neg)   
    return model


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
                    arc_new_arr.append([[0,0,0,0,0],positional])
                init_num+=1
            else:
                named_child_array,init_num,narr = create_numbered_arc_array(arc_array[i][0],init_num)
                name_array.append([named_child_array,arc_array[i][1]])
                arc_new_arr.append([narr,arc_array[i][1]])
        return name_array,init_num,arc_new_arr
    return create_numbered_arc_array(a_array,i_num)
       
def plot_eig(eig):
    for k in eig.keys():
        y= eig[k]
        x = [i for i in range(len(y))]
        plt.scatter(x,y)
        plt.plot(x,[0 for i in range(1000)],linewidth=3,color='red')
        plt.title(f"Block {k}")
        plt.savefig(f"eig/{k}.jpg")

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
    plot_eig(eig)