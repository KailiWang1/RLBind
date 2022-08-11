import sys
import math
import time
import pickle
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
        
class DefinedConfig(object):
    batch_size = 16
    splite_rate = 0.9
    epochs = 60
    class_nums = 1
    cutoff_seq_len = 64   ### 70 60 no; 
    threadhold = 0.6
    dropout = 0.1