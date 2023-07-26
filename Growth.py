import torch
import torch.nn as nn
import torch.nn.functional as F
class GrowthModel(nn.Module):
    def __init__(self,layer_array,architecture_array,act_on=True):
        super(GrowthModel,self).__init__()
        self.layer_dict= nn.ModuleDict(layer_array)
        self.architecture_array = architecture_array
        self.layer_count=0
        self.act_on= act_on
    def module(self,arc_array,x,final=False):
         #The architecture_array here has a fixed size of 3
            
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
                x2 = F.prelu(self.layer_dict[str(self.layer_count)](x),weight=torch.Tensor([0.25]).cuda()) # since in layer array new layer comes before old split
                self.layer_count+=1
            else:
                #new_layer
                x2 = F.prelu(self.module(architecture_array[1],x),weight=torch.Tensor([0.25]).cuda())
            
            if architecture_array[2]==0:
                #feature_bottleneck
                x3 = F.prelu(self.layer_dict[str(self.layer_count)](x2),weight=torch.Tensor([0.25]).cuda()) # since in layer array new layer comes before old split
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
            skip = x4 + x5
            if len(skip.shape)==2:
                x = torch.cat([x1,skip],axis=1)
                x = x[:,perm].view(x.shape[0],x.shape[1])
            
            else:
                x = torch.cat([x1,skip],axis=2)
                x = x[:,:,perm].view(x.shape[0],x.shape[1],x.shape[2])
            
            if final==False:
                x = F.prelu(x,weight=torch.Tensor([0.25]).cuda())
            return x
                

    def forward(self,x,):
        self.layer_count=0
        
        for i in range(len(self.architecture_array)-1):
            if self.architecture_array[i]== 0:
                x = F.prelu(self.layer_dict[str(self.layer_count)](x),weight=torch.Tensor([0.25]).cuda())
                self.layer_count+=1
            else:
                x = self.module(self.architecture_array[i],x)

        if self.architecture_array[len(self.architecture_array)-1]== 0:
            if self.act_on:

                x = F.prelu(self.layer_dict[str(self.layer_count)](x),weight=torch.Tensor([0.25]).cuda())

            else:
                x = self.layer_dict[str(self.layer_count)](x)

        else:
            x =  self.module(self.architecture_array[len(self.architecture_array)-1],x,final=True)
        return x  

def GrowthBlock(linear,act_on=False):
    layer_array ={'0':linear}
    arc_array = [0]
    gb = GrowthModel(layer_array,arc_array,act_on)
    return gb