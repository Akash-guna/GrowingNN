import torch
import torch.nn as nn
import torch.nn.functional as F
class GrowthModel(nn.Module):
    def __init__(self,layer_array,architecture_array):
        super(GrowthModel,self).__init__()
        self.layer_dict= nn.ModuleDict(layer_array)
        self.architecture_array = architecture_array
        self.layer_count=0
        
    def module(self,arc_array,x):
         #The architecture_array here has a fixed size of 3
            architecture_array=arc_array[0]
            perm = arc_array[1]
            #print([i for i in range(len(chosen_indices)) if chosen_indices[i]==1])
            if architecture_array[0]==0:
                #old_layer
                x1 = self.layer_dict[str(self.layer_count)](x)
                #print(0,self.layer_dict[str(self.layer_count)])
                #print("old Layer",torch.sum(x1))
                self.layer_count+=1
            else:
                #old_layer
                x1 = self.module(architecture_array[0],x)
                #print("old Layer",torch.sum(x1))
            if architecture_array[1]==0:
                #new_layer
                #print(1,self.layer_dict[str(self.layer_count)])
                x2 = F.prelu(self.layer_dict[str(self.layer_count)](x),weight=w) # since in layer array new layer comes before old split
                #print("old Split",torch.sum(x2))
                self.layer_count+=1
            else:
                #new_layer
                x2 = F.prelu(self.module(architecture_array[1],x),weight=w)
            
            if architecture_array[2]==0:
                #feature_bottleneck
                #print(2,self.layer_dict[str(self.layer_count)])
                x3 = F.prelu(self.layer_dict[str(self.layer_count)](x2),weight=w) # since in layer array new layer comes before old split
                #print("old Split",torch.sum(x2))
                self.layer_count+=1
            else:
                #feature_bottleneck
                x3 = F.prelu(self.module(architecture_array[2],x2),weight=w)
            
                #print("old Split",torch.sum(x2))
            if architecture_array[3]==0:
                #old_split
                #print(3,self.layer_dict[str(self.layer_count)])
                x4 = self.layer_dict[str(self.layer_count)](x3)
                #print("New Layer",torch.sum(x3))
                self.layer_count+=1
            else:
                #old_split
                x4 = self.module(architecture_array[3],x3)
            
            if architecture_array[4]==0:
                #old_split
                #print(3,self.layer_dict[str(self.layer_count)])
                x5 = self.layer_dict[str(self.layer_count)](x2)
                #print("New Layer",torch.sum(x3))
                self.layer_count+=1
            else:
                #old_split
                x5 = self.module(architecture_array[4],x2)
                #print("New Layer",torch.sum(x3))
            #x = F.relu(self.conc(x3,x1,chosen_indices))
            skip = x4 + x5
            x = F.prelu(torch.cat([x1,x4],axis=1),weight=w)
            #print("Before",x.shape)
            #print(perm.shape)
            x = x[:,perm].view(x.shape[0],x.shape[1])
            #print(x.shape)
            
            #print(x3)
            #print("After Module",torch.sum(x))
            return x
                

    def forward(self,x):
        #print(self.layer_dict)
        #print(x.shape)
        self.layer_count=0
        
        for i in range(len(self.architecture_array)-1):
            if self.architecture_array[i]== 0:
                #print(self.layer_count)
                x = F.prelu(self.layer_dict[str(self.layer_count)](x),weight=w)
                #print("Linear Layer",torch.sum(x))
                #print(self.layer_dict[str(self.layer_count)])
                #print(x)
                self.layer_count+=1
            else:
                x = self.module(self.architecture_array[i],x)
                #print("Main Block Module",torch.sum(x))
               
                #print(x)
        x = F.softmax(self.layer_dict[str(self.layer_count)](x))
        #print("Final Output",torch.sum(x))
        return x  

def GrowthBlock(linear):
    layer_array ={'0':linear}
    arc_array = [0]
    gb = GrowthModel(layer_array,arc_array)
    return gb