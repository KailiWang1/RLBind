#-*- encoding:utf8 -*-
##### construct the model of binding sites prediction                    __written by kaili on 18th MAY 2021
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import torch as t
from torch.autograd import Variable 
import os
import time
import sys
from collections import OrderedDict
from utils import *
from utils import DefinedConfig
defconstant = DefinedConfig()

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        in_channel = 1
        out_channel = 128
        kernels = [17, 19, 21]     #### the kernels can be changed [17, 19, 21]
        features_dim =10
        window_size = defconstant.cutoff_seq_len
        local_window = 11    ## 7
        padding1 = (kernels[0]-1)//2
        class_num = 1

        self.conv1 = torch.nn.Sequential()
        self.conv1.add_module("conv1",torch.nn.Conv2d(in_channel,out_channel,padding= (padding1,0),kernel_size=(kernels[0],features_dim)))
        self.conv1.add_module("relu1",torch.nn.ReLU())
        self.conv1.add_module("pool2",torch.nn.MaxPool2d(kernel_size= (window_size,1),stride=1))
    def forward(self,x):
        features = self.conv1(x)
        shapes = features.data.shape
        features = features.view(shapes[0],shapes[1]*shapes[2]*shapes[3])
        return features
                    
class RLBind(torch.nn.Module):
    def __init__(self):
        super(RLBind, self).__init__()

        out_channel = 128
        kernels = [15, 19, 21]   #### the kernels can be changed   [17, 19, 21] 
        features_dim =10
        local_window = 11   ## 7
        class_num = 1
        self.dropout = 0.1 
        local_dim = local_window * features_dim   
        input_dim = local_dim + out_channel                
        self.Multi_CNN = nn.Sequential()
        self.Multi_CNN.add_module("layer_convs1",Net())

        self.DNN1 = nn.Sequential()
        self.DNN1.add_module("Dense1", torch.nn.Linear(input_dim,192))
        self.DNN1.add_module("Relu1", torch.nn.ReLU())
        self.dropout_layer = nn.Dropout(self.dropout)
        self.DNN2 = nn.Sequential()
        self.DNN2.add_module("Dense2", torch.nn.Linear(192,96))
        self.DNN2.add_module("Relu2", torch.nn.ReLU())
                              
        self.outLayer = nn.Sequential(
            torch.nn.Linear(96, class_num),
            torch.nn.Sigmoid())
        
    def forward(self,all_features,local_features): 
        features = self.Multi_CNN(all_features)
        shapes = features.data.shape
        shapes = local_features.data.shape
        local_features = local_features.view(shapes[0],shapes[1] * shapes[2] * shapes[3])
        features = t.cat((features, local_features),1)  
        features = self.DNN1(features)
#         features = self.dropout_layer(features)
        features = self.DNN2(features)
        features = self.outLayer(features)

        return features