import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.checkpoint as checkpoint



class SharedSaveTensor():
  
  # tensor_dic with base64 string representation of tensor as key and tensor as value
  created_objects=0
  base64_tensor={}
  def __init__(self,sum=None,index=None):
    self.sum = sum
    self.index=index
    SharedSaveTensor.created_objects+=1
    
  
  def search_for_tensor(self,tensor):
    if self.sum in SharedSaveTensor.base64_tensor.keys():
      arr = SharedSaveTensor.base64_tensor[self.sum]
      for i in range(len(arr)):
        if torch.equal(arr[i], tensor):
          self.index =i
          return i
      return "NO_TENSOR"
    else:
      return "NO_KEY"


  def store_tensor(self,tensor):
    self.sum = str(torch.mean(torch.tensor([1.,1.,1.])).item())
    if self.search_for_tensor(tensor)  == "NO_KEY":
      SharedSaveTensor.base64_tensor[self.sum]=[tensor]
      self.index = 0
    elif self.search_for_tensor(tensor) =="NO_TENSOR":
      SharedSaveTensor.base64_tensor[self.sum].append(tensor)
      self.index = len(SharedSaveTensor.base64_tensor[self.sum])-1
    else:
      pass
    return [self.sum,self.index]
  def load_tensor(self,):
    return SharedSaveTensor.base64_tensor[self.sum][self.index]

  def __del__(self):

      SharedSaveTensor.created_objects -=1
      #print(SharedSaveTensor.created_objects)
      if SharedSaveTensor.created_objects ==0:
        SharedSaveTensor.base64_tensor = {}
def pack_hook(tensor):
    save_tensor_obj = SharedSaveTensor()
    var=save_tensor_obj.store_tensor(tensor)
    return save_tensor_obj

def unpack_hook(save_tensor_obj):
  #save_tensor_obj =  SharedSaveTensor(var[0],var[1])
  tensor = save_tensor_obj.load_tensor()
  #print(tensor)
  # if tensor ==None:
  #   raise Exception("Tensor was not stored properly, returned none on loading")
  return tensor






class GrowthModel(nn.Module):
    '''
    The Class to Create The Growing NN Block.

    Input:
        layer_array: A List containing all dense layers ordered from top to bottom

        act_on: Specifies whether to have an activation (used to turn of activation if the block is followed by a GelU in Deit)

        architecture_array: A Tree with 5 children per parent (each child denote one layer [old_layer,new_layer,feature_bottleneck,old_split,skip]) with value of each child could be a tree or 0. 0 denotes a leaf node which helps to add the layer to the model
                          Eg [0,0,0] -> a FeedForward Network with 3 Linear Layers.
                             [0,[[0,0],perm],0] -> Linear Layer -> Growth Block -> Linear Layer

                             perm -> is the order to shuffle to the concatentaion of old_split and split neurons to get the pre-split order. [1,3,4]-> not split, [0,2] -> split [1,3,4,0,2] -> after concat, perm= [3,0,4,1,2], after shuffle = [0,1,2,3,4]
                             Perm is added to each level of the tree (The Level of the tree is a growth block)

    Working: Layer array contains the layer, architecture array maps the layers to its corresponding position in a Neural Network.

    Output: A GrowthModel object (a Model)
    '''
    def __init__(self,layer_array,architecture_array,act_on=True):
        super(GrowthModel,self).__init__()
        self.layer_dict= nn.ModuleDict(layer_array)
        self.architecture_array = architecture_array
        self.layer_count=0
        self.act_on= act_on
        self.gelu = nn.GELU()
    def module(self,arc_array,x):
            """
            Growth Block Definition. Recursively Calls itself for branching.
            Input :
                arc_array : architecture array -> Main Call. a child architecture array  of parent if recursively called
                x : input (from forward())
            """
            #The architecture_array here has a fixed size of 5 [a,a] a = another arcitecture_array | 0
            architecture_array=arc_array[0]
            perm = arc_array[1]
            #print(torch.sum(x))
            
            if architecture_array[0]==0:
                #old_layer
                x1 = self.layer_dict[str(self.layer_count)](x)
                
                #print(torch.sum(x1))
                self.layer_count+=1
            else:
                #old_layer
                x1 = self.module(architecture_array[0],x)
            if architecture_array[1]==0:
                #new_layer
                x2 = self.layer_dict[str(self.layer_count)](x)
                
                self.layer_count+=1
            else:
                #new_layer
                x2 = self.module(architecture_array[1],x)
            #if architecture_array[2]==0:
                #new_layer
            #    x3 = self.layer_dict[str(self.layer_count)](x)
                
            #    self.layer_count+=1
            #else:
                #new_layer
            #    x3 = self.module(architecture_array[2],x)
            mid=int(x2.shape[-1]/2)
            if len(x2.shape)==2:
                #Handling Normal Linear Layers
                #x1[:,perm] += x2
                gelu_in = x2[:,:mid]+ x2[:,mid:]
                #print(gelu_in)
                x1[:,perm] += self.gelu(gelu_in) + x2[:,:mid]+ x2[:,mid:]
            else:
                #Handling QKV Layers , Stacked Linear Layers in Transformers
                #x1[:,:,perm]+=x2
                gelu_in = x2[:,:,:mid]+ x2[:,:,mid:]
                #print(gelu_in)
                x1[:,:,perm] += self.gelu(gelu_in) + x2[:,:,:mid]+ x2[:,:,mid:]
            #if final==False:
            #    x1 = self.gelu(x1)
            return x1
    def forward(self,x,):
        self.layer_count=0
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            for i in range(len(self.architecture_array)-1):
                # If a linear Layer add to model graph
                if self.architecture_array[i]== 0:
                    x = self.gelu(self.layer_dict[str(self.layer_count)](x))
                    self.layer_count+=1
                else:
                    # Else call the module() to handle growth
                    x = self.gelu(self.module(self.architecture_array[i],x))
                    
            #For final layer
            if self.architecture_array[len(self.architecture_array)-1]== 0:
                if self.act_on:
                    
                    x = self.gelu(self.layer_dict[str(self.layer_count)](x))

                else:
                    x = self.layer_dict[str(self.layer_count)](x)

            else:
                x =  self.module(self.architecture_array[len(self.architecture_array)-1],x)
        self.layer_count=0
        return x

def GrowthBlock(linear,act_on=False):
    '''
    Wraps a linear layer in a linear model with a GrowthModel Class.
    '''
    layer_array ={'0':linear}
    #Since single linear layer arc arr = [0]
    arc_array = [0]
    gb = GrowthModel(layer_array,arc_array,act_on)
    return gb


