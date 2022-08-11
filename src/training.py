import os
import sys
import math
import time
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data.sampler as sampler
import torch.optim as optim
from sklearn.model_selection import KFold

from RL_bind import RLBind
from dataset import DataSet
from evaluation import compute_roc, compute_mcc, micro_score, compute_performance
from utils import *
from utils import DefinedConfig
defconstant = DefinedConfig()

device = torch.device("cuda:0")

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def params_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m,nn.Linear):
        nn.init.normal_(m.weight.data, mean=0, std=min(1.0 / math.sqrt(m.weight.data.shape[-1]), 0.1))
        nn.init.constant_(m.bias, 0)
        
def t_epoch(model, load, optimizer, epoch, epochs, trainnum = None):
    print_freq= 10
    global threads
    model.train()
    losses = AverageMeter()
    
    for batch_indx, (global_feats, local_feats, labels, nucle_idx, rna_id) in enumerate(load): ### nucle_idx: sample id
        with torch.no_grad():
            if torch.cuda.is_available():
                global_vert = torch.autograd.Variable(global_feats.cuda().float())
                local_vert = torch.autograd.Variable(local_feats.cuda().float())
                label_vert = torch.autograd.Variable(labels.cuda().float())
            else:
                global_vert = torch.autograd.Variable(global_feats.float())
                local_vert = torch.autograd.Variable(local_feats.float())
                label_vert = torch.autograd.Variable(labels.float())
        
        batch_size = global_feats.size(0)
        output = model(global_vert, local_vert)
        output = torch.cat(tuple(output), 0)     
        loss = torch.nn.functional.binary_cross_entropy(output, label_vert).cuda()
        pred_value = output.ge(threads)
        pred_value = pred_value +0
        MiP, MiR, MiF, PNum, RNum = micro_score(pred_value.data.cpu().numpy(), label_vert.data.cpu().numpy())
        
        losses.update(loss.item(), batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_indx % print_freq == 0:
            performance = '\t'.join([
            'Epoch: %d/%d'% (epoch,epochs),
            'Iter: %d/%d'% (batch_indx, len(load)),
            'Loss: %0.4f'% (losses.avg),
            'MiP: %0.4f'% (MiP),
            'MiR: %0.4f'% (MiR),
            'MiF: %0.4f'% (MiF)   
            ])
    return losses.avg 

def v_epoch(model, load, is_test= True, validnum = None):
    print_freq = 1
    global threads
#     losses = 0
    losses = AverageMeter()
    model.eval()
    
    v_labels = []
    v_predicts = []
    nucle_nums = []
    rna_names = []
    
    for batch_indx, (global_feats, local_feats, labels, nucle_id, rna_id) in enumerate(load):
        with torch.no_grad():
            if torch.cuda.is_available():
                global_vert = torch.autograd.Variable(global_feats.cuda().float())
                local_vert = torch.autograd.Variable(local_feats.cuda().float())
                label_vert = torch.autograd.Variable(labels.cuda().float())
            else:
                globaa_vert = torch.autograd.Variable(global_feats.float())
                local_vert = torch.autograd.Variable(local_feats.float())
                label_vert = torch.autograd.Variable(labels.float())

        batch_size = global_feats.size(0)
        output = model(global_vert,local_vert)
        output = torch.cat(tuple(output), 0)

        loss = torch.nn.functional.binary_cross_entropy(output, label_vert).cuda()
        losses.update(loss.item(), batch_size)
        
        if batch_indx % print_freq == 0:
            performance = '\t'.join([
                'Test' if is_test else 'Valid',
                'Iter: [%d/%d]' % (batch_indx + 1, len(load)),
                'Loss %0.4f' % (losses.avg),
            ])

        v_labels.append(labels.numpy())
        v_predicts.append(output.data.cpu().numpy())
        nucle_nums.append(nucle_id.numpy())
        rna_names.append(rna_id)

    v_labels = np.concatenate(v_labels, axis=0)
    v_predicts = np.concatenate(v_predicts, axis=0)
    nucle_nums = np.concatenate(nucle_nums, axis=0)
    rna_names = np.concatenate(rna_names, axis=0) 
        
    auc = compute_roc(v_predicts, v_labels)
    p_max, r_max, t_max, predictions_max = compute_performance(v_predicts,v_labels)
    mcc = compute_mcc(predictions_max, v_labels)
    return losses.avg, p_max, r_max, auc, t_max, mcc, predictions_max, v_labels, v_predicts, nucle_nums, rna_names

def test_epoch(model, load, is_test= None):
    print_freq = 1
    global threads
    losses = AverageMeter()
    model.eval()
    
    v_labels = []
    v_predicts = []
    nucle_nums = []
    rna_names = []
    for batch_indx, (global_feats, local_feats, labels, nucle_id, rna_id) in enumerate(load):
        with torch.no_grad():
            if torch.cuda.is_available():
                global_vert = torch.autograd.Variable(global_feats.cuda().float())
                local_vert = torch.autograd.Variable(local_feats.cuda().float())
                label_vert = torch.autograd.Variable(labels.cuda().float())
            else:
                globaa_vert = torch.autograd.Variable(global_feats.float())
                local_vert = torch.autograd.Variable(local_feats.float())
                label_vert = torch.autograd.Variable(labels.float())
        batch_size = global_feats.size(0)
        output = model(global_vert,local_vert)
        output = torch.cat(tuple(output), 0)
        loss = torch.nn.functional.binary_cross_entropy(output, label_vert).cuda()
        losses.update(loss.item(), batch_size)
        
        if batch_indx % print_freq == 0:
            performance = '\t'.join([
                'Test19' if is_test else 'T19',
                'Iter: [%d/%d]' % (batch_indx + 1, len(load)),
                'Loss %0.4f' % (losses.avg),
            ])
        v_labels.append(labels.numpy())
        v_predicts.append(output.data.cpu().numpy())
        nucle_nums.append(nucle_id.numpy())
        rna_names.append(rna_id)
    v_labels = np.concatenate(v_labels, axis=0)
    v_predicts = np.concatenate(v_predicts, axis=0)
    nucle_nums = np.concatenate(nucle_nums, axis=0)
    rna_names = np.concatenate(rna_names, axis=0) 
        
    auc = compute_roc(v_predicts, v_labels)
    p_max, r_max, t_max, predictions_max = compute_performance(v_predicts,v_labels)
    mcc = compute_mcc(predictions_max, v_labels)
    return losses.avg, p_max, r_max, auc,t_max, mcc, predictions_max, v_labels, v_predicts, nucle_nums, rna_names
    
def train(model, train_dataset,test19, save=None, batch_size=32, train_number=6, epochs= 30, train_files=None, test19_files=None):
    cutoff_seq_len= defconstant.cutoff_seq_len
    global threads
    global splite_rate
    with open(train_files,'rb') as f_rna:
        rna_list = pickle.load(f_rna)
    # the test_index_files load
    with open(test19_files,'rb') as f_test19:
        test19_list = pickle.load(f_test19)
    nucleo_samples = len(rna_list)
    split_id = int(splite_rate * nucleo_samples)
    valid_id = nucleo_samples - split_id
    np.random.shuffle(rna_list)
    train_idx = rna_list[:split_id]     
    valid_idx = rna_list[split_id:]         
    train_samples = sampler.SubsetRandomSampler(train_idx)
    valid_samples = sampler.SubsetRandomSampler(valid_idx)
    ## load test set
    test19_samples = sampler.SubsetRandomSampler(test19_list)
    ## load dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_samples, pin_memory=(torch.cuda.is_available()), num_workers = 6, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_samples, pin_memory=(torch.cuda.is_available()), num_workers = 6, drop_last=False)
    test19_loader = torch.utils.data.DataLoader(test19, batch_size=batch_size, sampler=test19_samples, pin_memory=(torch.cuda.is_available()), num_workers = 6, drop_last=False)
    if torch.cuda.is_available():
        model = model.cuda()
    ## multi GPU
    model_wrapper = model
    optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=0.0001)
    best_p_max = 0.0
    best_r_max = 0.0
    for epoch in range(epochs):    ## a epoch of train and valid\
        t_loss = t_epoch(model = model_wrapper, load = train_loader, optimizer = optimizer, epoch = epoch, epochs = epochs, trainnum = split_id)
        v_loss, p_max, r_max, auc, t_max, mcc, v_predictions,v_labels, v_predicts, nucle_nums, rna_names= v_epoch(model= model_wrapper, load=valid_loader,is_test=(not valid_loader), validnum = valid_id)
### for the test set
        test19_loss, p_max19, r_max19, auc19, t_max19, mcc19, predictions19,v_labels19, v_predicts19, nucle_nums19, rna_names19 = test_epoch(model= model_wrapper, load=test19_loader,is_test=test19_loader)
        print('Epoch:%02d,Precision19:%0.4f,Recall19:%0.4f,AUC19:%0.4f,T_max19:%0.4f,MCC19:%0.4f\n'%((epoch+1),test19_loss,p_max19, r_max19, auc19,t_max19, mcc19))
 
        if p_max > best_p_max and r_max > best_r_max:
            best_p_max = p_max
            best_r_max = r_max
            threadhold = t_max
            print('The best precision and recall is %2d,%0.4f,%0.4f (threadhold is %0.4f)'%(cutoff_seq_len,best_p_max, best_r_max, threads))
            torch.save(model.state_dict(),os.path.join(save, './model.dat'))

if __name__ == '__main__':

    batch_size = defconstant.batch_size
    splite_rate =defconstant.splite_rate
    epochs = defconstant.epochs
    class_nums = defconstant.class_nums
    threads = 0.577
    
    for numbers in range(1,60):
        path_dir = '../results'
        train_set = ['T60']   ### The train and valid data set
        test19_set = ['T19']
        if not os.path.exists(path_dir):
            os.makedirs(path_dir) 
 ### Train files
        train_global_files = ['../data_cache/data_%s.pkl'%(name) for name in train_set]
        train_global_labels = ['../data_cache/label_%s.pkl'%(name) for name in train_set]
        train_local_files = ['../data_cache/mot11_%s.pkl'%(name) for name in train_set]
        train_all_nucleotide_files = '../data_cache/train_all.pkl'
        train_index_files = '../data_cache/train_index.pkl'
### Test19 files
        test19_global_files = ['../data_cache/data_%s.pkl'%(name) for name in test19_set]
        test19_global_labels = ['../data_cache/label_%s.pkl'%(name) for name in test19_set]
        test19_local_files = ['../data_cache/mot11_%s.pkl'%(name) for name in test19_set]
        test19_all_nucleotide_files = '../data_cache/test19_all.pkl'
        test19_index_files = '../data_cache/test19_index.pkl'
        
        train_data = DataSet(train_global_files, train_local_files, train_global_labels, train_all_nucleotide_files)
        test19_data = DataSet(test19_global_files, test19_local_files, test19_global_labels, test19_all_nucleotide_files)
        
        model = RLBind()
        model.apply(params_init)
         #### model training 
        train(model, train_data,test19_data,path_dir, batch_size, numbers, epochs, train_index_files,test19_index_files)
        print("Training_(%s/60) finished!"%(numbers))
