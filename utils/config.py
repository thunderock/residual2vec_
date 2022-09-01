# @Filename:    config.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/31/22 5:59 PM
import os
import logging
import psutil
from multiprocessing import cpu_count
import numpy as np
import torch

P = psutil.Process(os.getpid())
try:
    P.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        P.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass
CUDA = torch.cuda.is_available()
TRAIN_TEST_SPLIT = .2
RANDOM_SEED = 2022
DROPOUT = .2
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cpu")
ALPHA = .2
BATCH_SIZE = 128
NUM_WORKERS = 4
EPOCHS = 5
PREDICTION_THRESHOLD = .7
NUM_NEIGHBORS = 10
EMBEDDING_DIM = 128
LR = .01
if CUDA:
    torch.cuda.manual_seed(RANDOM_SEED)
    DEVICE = torch.device("cuda")
LOGGER = logging.getLogger(__name__)
LOGFORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
