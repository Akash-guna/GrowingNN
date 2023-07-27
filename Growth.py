import torch
import torch.nn as nn
import torch.nn.functional as F
class GrowthModel(nn.Module):
    '''
    The Class to Create The Growing NN Block.
    
    Input:
        layer_array: A List containing all dense layers ordered from top to bottom
        
        act_on: Specifies whether to have an activation (used to turn of activation if the block is followed by a GelU in Deit)
        
        architecture_array: A Tree with 5 children per parent (each child denote one layer [old_layer,new_layer,feature_bottleneck,old_split,skip]) with value of each child could be a tree or 0. 0 denotes a leaf node which helps to add the layer to the model
                          Eg [0,0,0] -> a FeedForward Network with 3 Linear Layers.  
                             [0,[[0,0,0,0,0],perm],0] -> Linear Layer -> Growth Block -> Linear Layer
                             
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
    def module(self,arc_array,x,final=False):
            """
            Growth Block Definition. Recursively Calls itself for branching.
            Input :
                arc_array : architecture array -> Main Call. a child architecture array  of parent if recursively called
                x : input (from forward())
                final : True if its the final block (usually for transformer)
            """
            #The architecture_array here has a fixed size of 5 [a,a,a,a,a] a = another arcitecture_array | 0
            architecture_array=arc_array[0]
            perm = arc_array[1]
            if architecture_array[0]==0:
                #old_layer
                x1 = self.layer_dict[str(self.layer_count)](x)
                self.layer_count+=1
            else:
                #old_layer
                x1 = self.module(architecture_array[0],x)
            if architecture_array[1]==0:
                #new_layer
                x2 = F.prelu(self.layer_dict[str(self.layer_count)](x),weight=torch.Tensor([0.25]).cuda())
                self.layer_count+=1
            else:
                #new_layer
                x2 = F.prelu(self.module(architecture_array[1],x),weight=torch.Tensor([0.25]).cuda())
            
            if architecture_array[2]==0:
                #feature_bottleneck
                x3 = F.prelu(self.layer_dict[str(self.layer_count)](x2),weight=torch.Tensor([0.25]).cuda()) 
                self.layer_count+=1
            else:
                #feature_bottleneck
                x3 = F.prelu(self.module(architecture_array[2],x2),weight=torch.Tensor([0.25]).cuda())
            if architecture_array[3]==0:
                #old_split
                x4 = self.layer_dict[str(self.layer_count)](x3)
                self.layer_count+=1
            else:
                #old_split
                x4 = self.module(architecture_array[3],x3)
            
            if architecture_array[4]==0:
                #skip
                x5 = self.layer_dict[str(self.layer_count)](x2)
                self.layer_count+=1
            else:
                #skip
                x5 = self.module(architecture_array[4],x2)
            # Skip connection
            skip = x4 + x5
            
            if len(skip.shape)==2:
                #Handling Normal Linear Layers

                # concat old_split and after skip connection output
                x = torch.cat([x1,skip],axis=1)
                # Shuffle as per Perm to get pre-split order
                x = x[:,perm].view(x.shape[0],x.shape[1])
            
            else:
                #Handling QKV Layers , Stacked Linear Layers in Transformers
                # concat old_split and after skip connection output
                x = torch.cat([x1,skip],axis=2)
                # Shuffle as per Perm to get pre-split order
                x = x[:,:,perm].view(x.shape[0],x.shape[1],x.shape[2])
            
            if final==False:
                x = F.prelu(x,weight=torch.Tensor([0.25]).cuda())
            return x
                

    def forward(self,x,):
        self.layer_count=0
        
        for i in range(len(self.architecture_array)-1):
            # If a linear Layer add to model graph
            if self.architecture_array[i]== 0:
                x = F.prelu(self.layer_dict[str(self.layer_count)](x),weight=torch.Tensor([0.25]).cuda())
                self.layer_count+=1
            else:
                # Else call the module() to handle growth
                x = self.module(self.architecture_array[i],x)
        #For final layer
        if self.architecture_array[len(self.architecture_array)-1]== 0:
            if self.act_on:

                x = F.prelu(self.layer_dict[str(self.layer_count)](x),weight=torch.Tensor([0.25]).cuda())

            else:
                x = self.layer_dict[str(self.layer_count)](x)

        else:
            x =  self.module(self.architecture_array[len(self.architecture_array)-1],x,final=True)
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