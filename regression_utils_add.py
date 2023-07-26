import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset,DataLoader
#from growth_utils import train
from sklearn.model_selection import train_test_split

def train(net,loaders,input_size=784,EPOCHS=5,loss_func = nn.CrossEntropyLoss(),opt =optim.Adam,evaluate=True,verbose=True ):
    if verbose==True:
        print(net)
#     loss_func = nn.CrossEntropyLoss()
    optimizer = opt(net.parameters(), lr =0.001)
    loss_arr=[]
    train_acc_arr=[]
    test_acc_arr=[]
    
    net = net.cuda()
    for epoch in range(EPOCHS):
        for data in loaders["train"]:
            X, y = data
            X=X.cuda()
            y = y.cuda()
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
                        X=X.cuda()
                        y = y.cuda()
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
                        X=X.cuda()
                        y = y.cuda()
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
                        X=X.cuda()
                        y = y.cuda()
                        output = net(X.view(-1, input_size))
                        loss_test.append(loss_func(output,y))
            loss_test = torch.Tensor(loss_test)
            avg_test_loss = torch.mean(loss_test)
            if verbose==True:
                print(f"Epoch ={epoch} Loss = {loss}, Test Loss = {avg_test_loss} ")
        loss_arr.append(loss)

    return (loss_arr,train_acc_arr,test_acc_arr),net


class IntermediateRegressionModel(nn.Module):
    # Splitting Fc2
    def __init__(self,outputlayer,out_nodes,num_nodes):
        super(IntermediateRegressionModel,self).__init__()
        
        self.num_nodes= num_nodes
        self.new_layer = nn.Linear(outputlayer.in_features,self.num_nodes)
        self.fb = nn.Linear(self.num_nodes,int(self.num_nodes/2))
        self.skip_fc = nn.Linear(self.num_nodes,out_nodes )
        self.output_layer = nn.Linear(int(self.num_nodes/2),out_nodes)
    def forward(self,x):
        x1 = F.prelu(self.new_layer(x),weight=torch.Tensor([0.25]).cuda())
        x2 = F.prelu(self.fb(x1),weight=torch.Tensor([0.25]).cuda())
        skip = F.prelu(self.skip_fc(x1),weight=torch.Tensor([0.25]).cuda())
        x3 = self.output_layer(x2)
        x  = x3 + skip
        return x

class DataModel(nn.Module):
    def __init__(self,layer):
        super(DataModel,self).__init__()
        self.inp_features = layer.in_features
        self.layer = layer
    def forward(self,x):
        return self.layer(x)
    
class DataLoaderInterm(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __getitem__(self,idx):
        x_sample = self.x[idx]
        y_sample = self.y[idx]
        return (x_sample,y_sample)
    def __len__(self):
        return self.x.shape[0]

BATCH_SIZE=128
def load_data_interm(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    dl_interm_train = DataLoaderInterm(x_train,y_train)
    dl_interm_test = DataLoaderInterm(x_test,y_test)
    loaders = {
        'train' : torch.utils.data.DataLoader(dl_interm_train, 
                                              batch_size=BATCH_SIZE, 
                                              shuffle=True, 
                                              num_workers=0),

        'test'  : torch.utils.data.DataLoader(dl_interm_test, 
                                              batch_size=BATCH_SIZE, 
                                              shuffle=False, 
                                              num_workers=0),
    }
    return loaders

def generate_intermed_data(model,count,choices):
    model.eval()
    x=[]
    y=[]
    for i in range(count):
        rand = torch.rand((1,model.layer.in_features)).cuda()
        y_o= model(rand)
        x.append(torch.squeeze(rand,axis=0).cpu().detach().numpy().tolist())
        y.append(torch.squeeze(y_o)[choices].cpu().detach().numpy().tolist())
    model.train()
    x = torch.Tensor(x).cuda()
    y= torch.Tensor(y).cuda()
    return x,y

def return_layers(layer,choices,num_nodes=10,samp_size=10000,reg_epochs=100,verbose=False):
    intermed_model =IntermediateRegressionModel(layer,len(choices),num_nodes)
    data_model =DataModel(layer)
    x,y =generate_intermed_data(data_model,samp_size,choices)
    loaders=load_data_interm(x,y)
    metrics,intermed_model =train(intermed_model,loaders,EPOCHS=reg_epochs,input_size = layer.in_features, loss_func = nn.MSELoss(),evaluate=False,verbose=verbose)
    print(intermed_model.parameters)
    print("Intermediate Regrssion Loss = ",metrics[0][-1])
    return [intermed_model.new_layer,intermed_model.fb,intermed_model.output_layer,intermed_model.skip_fc,metrics]


def get_intermed_model_data(loaders,choices,model,sel_layer, samples=10000):
    activation={str(sel_layer):torch.Tensor([])}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name]=torch.concat([activation[name],output.detach()[:,choices]],axis=0)
        return hook
    x =torch.Tensor([])
    
    model.layer_dict[str(sel_layer)].register_forward_hook(get_activation(str(sel_layer)))
    for data in loaders["train"]:
        out = model(data[0].view(-1,3072))
        x = torch.concat([x,data[0]],axis=0)
        if x.shape[0]>=samples:
            break
    y = activation[str(sel_layer)]
    return x.cuda(),y.cuda()

def return_layers_data(loaders,model,layer,choices,num_nodes=10,samp_size=10000,reg_epochs=100,verbose=False):
    intermed_model =IntermediateRegressionModel(layer,len(choices),num_nodes)
    x,y=get_intermed_model_data(loaders,choices,model,'0',samples=samp_size)
    loaders_inter=load_data_interm(x,y)
    inter_met,intermed_model =train(intermed_model,loaders_inter,EPOCHS=reg_epochs,input_size = layer.in_features, loss_func = nn.MSELoss(),evaluate=False,verbose=verbose)
    return [intermed_model.new_layer,intermed_model.fb,intermed_model.output_layer,intermed_model.skip_fc,inter_met]