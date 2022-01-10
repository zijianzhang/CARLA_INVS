#! /usr/bin/env python3
# amend relative import
import sys
from pathlib import Path
sys.path.append( Path(__file__).resolve().parent.parent.as_posix() ) #repo path
sys.path.append( Path(__file__).resolve().parent.as_posix() ) #file path
from params import *
# original import
import os
import subprocess as sp
import torch
import pdb
import copy
import time
from sys import argv
from averaging import average_weights
import numpy as np

ITER = int(argv[1])
train_file = Path(__file__, '..', 'tools', 'train.py').resolve().as_posix()
cfg_folder = Path(__file__, '..', 'tools', 'cfgs')

#------------------------------ Attention ------------------------------#
#       The following codes demonstrate federated learning              #
#       for sample dataset available here:                              #
#       https://cloud.189.cn/t/jQJvuimquaEj                             #
#-----------------------------------------------------------------------#

model713_file = MODEL_FOLDER / 'model713.pth'
model714_file = MODEL_FOLDER / 'model714.pth'
try:
    model713_file.unlink()
    model714_file.unlink()
except Exception as e: 
    pass

with workSpace(TRAIN_ROOT) as wrk: #chdir to 'tools' folder
    if ITER == 0: # no pretrain model
        sp.run(['bash', '-c', 'python3 train.py --cfg_file cfgs/tesla713.yaml --batch_size 1 --workers 0 --epochs 1 --root_dir %s'%ROOT_PATH])
        sp.run(['bash', '-c', 'python3 train.py --cfg_file cfgs/tesla714.yaml --batch_size 1 --workers 0 --epochs 1 --root_dir %s'%ROOT_PATH])

    if ITER >= 1: # train from last global model
        sp.run(['bash', '-c', 'python3 train.py --cfg_file cfgs/tesla713.yaml --batch_size 1 --workers 0 --epochs 1 --pretrained_model %s --root_dir %s'%(model713_file, ROOT_PATH)])
        sp.run(['bash', '-c', 'python3 train.py --cfg_file cfgs/tesla714.yaml --batch_size 1 --workers 0 --epochs 1 --pretrained_model %s --root_dir %s'%(model714_file, ROOT_PATH)])
    pass

  
# save model to dictionary
model_dict0 = torch.load( str(MODEL_FOLDER/'checkpoint_epoch_0.pth') ); params0 = model_dict0['model_state']
model_dict1 = torch.load( str(MODEL_FOLDER/'checkpoint_epoch_1.pth') ); params1 = model_dict1['model_state']

# append parameters into a list
w_locals = []
w_locals.append(params0)
w_locals.append(params1)

# compute perfect global model
params_avg = average_weights(w_locals)

# remove the model files
del(params0)
del(params1)
(MODEL_FOLDER/'checkpoint_epoch_0.pth').unlink()
(MODEL_FOLDER/'checkpoint_epoch_1.pth').unlink()

# save global model at each vehicle
torch.save({'model_state':params_avg,
            'optimizer_state': model_dict0['optimizer_state'],
            'accumulated_iter': model_dict0['accumulated_iter']
            },
            str(model713_file)
        )

torch.save({'model_state':params_avg,
            'optimizer_state': model_dict1['optimizer_state'],
            'accumulated_iter': model_dict1['accumulated_iter']
            },
            str(model714_file)
        )

# finish one FL iteration
print('---------------------------------FL Iteration %r is completed.'%argv[1])

