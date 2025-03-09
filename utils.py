import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from copy import deepcopy

from dataclasses import dataclass, asdict
import json
from sklearn import metrics
import pandas as pd

@dataclass
class Params:
    num_sites: int = None
    num_rounds: int = None
    inner_epochs: int = None
    batch_size: int = None
    outer_lr: float = None
    weight_decay: float = None
    inner_lr: float = None

PARAMS_FILE = "params.json"

def save_params(run_path, params):
    with open(f'{run_path}/{PARAMS_FILE}', "w") as f:
        json.dump(asdict(params), f, indent=2)

def get_device():
    if torch.cuda.is_available():
       device=torch.device('cuda:0')
    else:
       device=torch.device('cpu')   
    return device




def prediction_binary(model,loader,loss_fn,device):
    P=[]
    L=[]
    model.eval()
    val_loss=0
    for i,batch in enumerate(loader):
        data,labels=batch
        data=data.to(torch.float32).to(device)
        labels=labels.to(torch.float32).to(device)
        
        pred=model(data)[:,0]
        loss=loss_fn(pred,labels)
        val_loss=val_loss+loss.item()

        P.append(pred.cpu().detach().numpy())
        L.append(labels.cpu().detach().numpy())
        
    val_loss=val_loss/len(loader)
    P=np.concatenate(P)  
    L=np.concatenate(L)
    auc=roc_auc_score(L,P)


    # apr = metrics.average_precision_score(L,P)
    precision, recall, _ = metrics.precision_recall_curve(L,P)
    apr = metrics.auc(recall,precision)

    return val_loss,auc,apr



def evaluate_models(client_id, Loaders, net, loss_fn, device, df, B,train_loss,path):
    ''' Given site i, and model net, evaluate the model peformance on the site's val set'''
   
    val_loss, val_auc, val_apr = prediction_binary(net, Loaders[client_id][1], loss_fn, device) 
    
    if val_auc> B[client_id]:
       B[client_id] = val_auc
       torch.save(net, f'./trained_models/'+path+'/node'+str(client_id)) 
    df = pd.concat([df, pd.DataFrame([{ 'Train_Loss': train_loss,'Val_Loss': val_loss, 'Val_AUC': val_auc,'Val_APR': val_apr}])], ignore_index=True)    
    return df, B[client_id] ,val_auc,val_apr    


def evaluate_models_test(client_id,Loaders, net, loss_fn, device):
    ''' Given site i, and model net, evaluate the model peformance on the site's val set'''
    val_loss, val_auc, val_apr = prediction_binary(net, Loaders[client_id][1], loss_fn, device) 
    return val_loss,val_auc,val_apr    




