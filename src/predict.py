import os
import sys
import math
import time
import pickle
import glob
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data.sampler as sampler
import torch.optim as optim

from RL_bind import RLBind
from dataset import DataSet
from evaluation import compute_roc,compute_mcc, micro_score, compute_performance
from utils import *
from utils import DefinedConfig
defconstant = DefinedConfig()
    
def test(model, load):
    model.eval()
    global threads
    t_labels = []
    t_predicts = []
    nucle_nums = []
    rna_names = []
    for batch_indx, (global_feats, local_feats, labels, nucle_id, rna_id) in enumerate(load):
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
        output = model(global_vert,local_vert)
        output = torch.cat(tuple(output), 0)
        t_labels.append(labels.numpy())
        t_predicts.append(output.data.cpu().numpy())
        nucle_nums.append(nucle_id.numpy())
        rna_names.append(rna_id)

    t_labels = np.concatenate(t_labels, axis=0)
    t_predicts = np.concatenate(t_predicts, axis=0)
    nucle_nums = np.concatenate(nucle_nums, axis=0)
    rna_names = np.concatenate(rna_names, axis=0) 
        
    auc = compute_roc(t_predicts, t_labels)
    p_max, r_max, t_max, predictions_max = compute_performance(t_predicts,t_labels)
    mcc = compute_mcc(predictions_max, t_labels)
    return p_max, r_max, auc, t_max, mcc, predictions_max, t_labels, t_predicts, nucle_nums, rna_names

def predict(model_file,test_data,batch_size,test_index,save=None):
    global threads
    cutoff_seq_len= defconstant.cutoff_seq_len
    with open(test_index,'rb') as f_test:
        test_list = pickle.load(f_test)
    test_samples = sampler.SubsetRandomSampler(test_list)
    test_load = torch.utils.data.DataLoader(test_data, batch_size=batch_size, sampler=test_samples, pin_memory=(torch.cuda.is_available()), num_workers = 6, drop_last=False)
    model = RLBind()
    model.load_state_dict(torch.load(model_file))
    model = model.cuda()
    p_max,r_max,auc,t_max,mcc,predictions,t_labels,t_predicts,nucle_nums,rna_names=test(model,test_load)
    print('Precision:%0.4f,Recall:%0.4f,AUC:%0.4f,T_max:%0.4f,MCC:%0.4f,cutoff_len:%02d\n'%(p_max, r_max, auc, t_max, mcc, cutoff_seq_len))
#     print(t_labels,predictions, nucle_nums,rna_names)
    result_dict = {}
    for i in range(len(t_labels)):
        result_dict[nucle_nums[i]]=(rna_names[i],int(t_labels[i]),int(predictions[i]))
    keylist = sorted(result_dict)
    sort_dict = {}
    for j in keylist:
        sort_dict[j] = result_dict[j]
    head = ["RNA_name","Label","Predicted"]
    (pd.DataFrame.from_dict(data=sort_dict, orient='index').\
     to_csv(os.path.join(save, 'results.csv'),header=head))        
    
    return p_max, r_max, auc, mcc

if __name__ == '__main__':

    batch_size = defconstant.batch_size
    path_dir = '../results'
    test_set = ['T19']
    threads = 0.577
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
        
    test_global_files = ['../data_cache/data_%s.pkl'%(name) for name in test_set]
    test_global_labels = ['../data_cache/label_%s.pkl'%(name) for name in test_set]
    test_local_files = ['../data_cache/mot11_%s.pkl'%(name) for name in test_set]
    test_all_nucleotide_files = '../data_cache/test19_all.pkl'
    test_index_files = '../data_cache/test19_index.pkl'
    
    test_data = DataSet(test_global_files, test_local_files, test_global_labels, test_all_nucleotide_files)
    
    f_cont = glob.glob('%s/best_model.dat'%(path_dir))
    for f_i in f_cont:
        print(f_i)
        model_file = f_i
        p_max, r_max, auc, mcc = predict(model_file, test_data, batch_size, test_index_files, path_dir)
    print('p_max:%f, r_max:%f, auc:%f, mcc:%f'% (p_max, r_max, auc, mcc))