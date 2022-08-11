#-*- encoding:utf8 -*-
import os
import sys
import time
import pickle
import torch as t
import numpy as np
from torch.utils import data
from utils import *
from utils import DefinedConfig
defconstant = DefinedConfig()

class DataSet(data.Dataset):
    def __init__(self, train_global_files = None, train_local_files = None, train_labels = None, train_nucleotide_files = None):
        super(DataSet,self).__init__()

        for all_files in train_global_files:
            with open(all_files,'rb') as f_all:
                self.global_features = pickle.load(f_all)

        for label_files in train_labels:
            with open(label_files, 'rb') as f_lab:
                self.global_labels = pickle.load(f_lab)

        for local_files in train_local_files:
            with open(local_files, 'rb') as f_loc:
                self.local_features = pickle.load(f_loc)
        
        with open(train_nucleotide_files, 'rb') as f_indx:
            self.rna_list= pickle.load(f_indx)

        self.cutoff_seq_len = defconstant.cutoff_seq_len
        
    def __getitem__(self,index):
        nucle_nums, nucle_id, rna_nums, rna_id, nucle_length = self.rna_list[index]
        nucle_nums = int(nucle_nums)
        nucle_id = int(nucle_id)
        rna_nums = int(rna_nums)
        nucle_length = int(nucle_length)

        ## cutoff_global_features for a rna
        cutoff_global_feats = self.global_features[rna_nums][:self.cutoff_seq_len]
        nucle_len = len(cutoff_global_feats)
        while nucle_len < self.cutoff_seq_len:
            zero_vector = [0 for i in range(10)]
            cutoff_global_feats.append(zero_vector)
            nucle_len += 1
        
        ## local_features for a nucleotide
        local_feats = self.local_features[nucle_nums]
        ## the label of a nucleotide
        label = self.global_labels[rna_nums][nucle_id]   
        
        cutoff_global_feats = np.stack(cutoff_global_feats)
        cutoff_global_feats = cutoff_global_feats[np.newaxis,:,:]
        
        local_feats = np.stack(local_feats)
        local_feats = local_feats[np.newaxis,:,:]
        label = np.array(label, dtype=np.float32)

        return cutoff_global_feats, local_feats, label, index, rna_id
    
    def __len__(self):
        return len(self.rna_list)
