from glob import glob
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from growth_utils_node import get_all_linear_layers
import shutil
import warnings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='Visualization of layers')
parser.add_argument('folder',type =str , help='folder of models')
args = parser.parse_args()
try:
    os.remove(args.folder+"/0.pt")
except:
    pass

modelpaths = glob(args.folder+'/*.pt' )
modelpaths = [i for i in modelpaths if "chk" not in i]

num_models = len(modelpaths)
n=args.folder.split('\\')[-1]
viz_path = f"viz/{n}"
try:
    shutil.rmtree(viz_path)
except:
    pass

#shutil.copy('0.pt',os.path.join(args.folder,'0.pt'))

def plot_pie(count,labels,path):
    sns.set(font_scale = 1.2)
    plt.figure(figsize=(7,7))

    plt.pie(
        x=count, 
        labels=labels,
        colors=sns.color_palette('rocket'),
        startangle=90,

        explode=[ 0.05, 0.05, 0.05, 0.05]
    )

    hole = plt.Circle((0, 0), 0.65, facecolor='white')
    plt.gcf().gca().add_artist(hole)
    plt.title(f"{path.split('/')[-1].split('.')[0]}th  Split Distribution")
    #plt.show()
    plt.savefig(path)

def plot_bar(inc,labels,path):
    sns.set(font_scale = 1.2)
    plt.figure(figsize=(7,7))
    n=path.split('/')[-1].split('.')[0]
    plt.title(f"{n}th split Number of Increase")
    sns.barplot(x = 'label',
            y = 'inc',
            data = {'inc':inc,"label":labels},palette = sns.color_palette('rocket'))
    #plt.show()
    plt.savefig(path)

def plot_param(param,path,p):
    print(param)
    plt.clf()
    sns.barplot(x = 'block',
            y = 'parameters',
            data = {'block':[i for i in range(len(param))],"parameters":param},palette = sns.color_palette('rocket'))
    n=path.split('/')[-1].split('.')[0]
    plt.title(f"{n}th split parameter distribution - {p}")
    plt.xlabel("Blocks")
    plt.ylabel("Parameters")
    plt.savefig(path)


def check_split(bef,aft):
    b = len(get_all_linear_layers(bef,typ='list'))
    a = len(get_all_linear_layers(aft,typ='list'))
    if b==a:
        return 0
    else:
        print(b,a)
        return 1
def get_inc(bef,aft):
    b = len(get_all_linear_layers(bef,typ='list'))
    a = len(get_all_linear_layers(aft,typ='list'))
    return a-b
def get_parameter(bef,aft):
    b= get_all_linear_layers(bef,typ='list')
    a= get_all_linear_layers(aft,typ='list')
    
    b_parameters = sum(p.numel() for p in bef.parameters())
    a_parameters = sum(p.numel() for p in aft.parameters())
    print(b_parameters,a_parameters)
    return a_parameters-b_parameters
print(modelpaths)
for j in range(1,num_models):
    before_model = torch.load(modelpaths[j-1],map_location='cuda')["model"]
    after_model = torch.load(modelpaths[j],map_location='cuda')["model"]
    count=[0,0,0,0]
    inc_layers=[0,0,0,0]
    labels =["QKV","Proj","FC1","FC2"]
    parameters_total=[]
    parameters_qkv =[]
    parameters_proj=[]
    parameters_fc1=[]
    parameters_fc2 =[]
    for i in range(len(before_model.blocks)):
        total_param=0
        if check_split(before_model.blocks[i].attn.qkv,after_model.blocks[i].attn.qkv):
            count[0]+=1
            inc_layers[0]+= get_inc(before_model.blocks[i].attn.qkv,after_model.blocks[i].attn.qkv)
            param = get_parameter(before_model.blocks[i].attn.qkv,after_model.blocks[i].attn.qkv)
            parameters_qkv.append(param)
            total_param+=param
        else:
            parameters_qkv.append(0)
        if check_split(before_model.blocks[i].attn.proj,after_model.blocks[i].attn.proj):
            count[1]+=1
            inc_layers[1]+= get_inc(before_model.blocks[i].attn.proj,after_model.blocks[i].attn.proj)
            param = get_parameter(before_model.blocks[i].attn.proj,after_model.blocks[i].attn.proj)
            parameters_proj.append(param)
            total_param+=param
        else:
            parameters_proj.append(0)
        if check_split(before_model.blocks[i].mlp.fc1,after_model.blocks[i].mlp.fc1):
            count[2]+=1
            inc_layers[2]+= get_inc(before_model.blocks[i].mlp.fc1,after_model.blocks[i].mlp.fc1)
            param = get_parameter(before_model.blocks[i].mlp.fc1,after_model.blocks[i].mlp.fc1)
            parameters_fc1.append(param)
            total_param+=param
        else:
            parameters_fc1.append(0)
        if check_split(before_model.blocks[i].mlp.fc2,after_model.blocks[i].mlp.fc2):
            count[3]+=1
            inc_layers[3]+= get_inc(before_model.blocks[i].mlp.fc2,after_model.blocks[i].mlp.fc2)
            param = get_parameter(before_model.blocks[i].mlp.fc2,after_model.blocks[i].mlp.fc2)
            parameters_fc2.append(param)
            total_param+=param
        else:
            parameters_fc2.append(0)
        parameters_total.append(total_param)
    print(parameters_qkv)
    os.makedirs(viz_path+'/percent_diff',exist_ok=True)
    os.makedirs(viz_path+'/layer_increase',exist_ok=True)
    os.makedirs(viz_path+'/total_params',exist_ok=True)
    os.makedirs(viz_path+'/qkv_params',exist_ok=True)
    os.makedirs(viz_path+'/proj_params',exist_ok=True)
    os.makedirs(viz_path+'/fc1_params',exist_ok=True)
    os.makedirs(viz_path+'/fc2_params',exist_ok=True)   
    plot_pie(count,labels,viz_path+f'/percent_diff/{j-1}.jpg' )
    plot_bar(inc_layers,labels,viz_path+f'/layer_increase/{j-1}.jpg' )
    plot_param(parameters_total,viz_path+f'/total_params/{j-1}.png',"Total Parameters")
    plot_param(parameters_qkv,viz_path+f'/qkv_params/{j-1}.png',"QKV Parameters")
    plot_param(parameters_proj,viz_path+f'/proj_params/{j-1}.png',"Proj Parameters")
    plot_param(parameters_fc1,viz_path+f'/fc1_params/{j-1}.png',"FC1 Parameters")
    plot_param(parameters_fc2,viz_path+f'/fc2_params/{j-1}.png',"FC2 Parameters")
    print(total_param)
    print(sum(parameters_fc2))