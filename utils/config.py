# @Filename:    config.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/31/22 5:59 PM
import os
import logging
import psutil
from multiprocessing import cpu_count
import torch

def get_string_boolean(string):
    if string in ['True', 'true', 'TRUE', 'T', 't', '1']:
        return True
    elif string in ['False', 'false', 'FALSE', 'F', 'f', '0']:
        return False
    else:
        raise ValueError('String must be either True or False')

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
# torch.manual_seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cpu")
# ALPHA = .2
# BATCH_SIZE = 128
# NUM_WORKERS = 4
# EPOCHS = 5
PREDICTION_THRESHOLD = .7
# NUM_NEIGHBORS = 10
EMBEDDING_DIM = 128
# LR = .01
GPU_ID = 0
DISABLE_TQDM = os.environ.get("DISABLE_TQDM", 'False')
DISABLE_TQDM = get_string_boolean(DISABLE_TQDM)
DISABLE_WANDB = os.environ.get("DISABLE_WANDB", 'False')
DISABLE_WANDB = get_string_boolean(DISABLE_WANDB)
if CUDA:
    # torch.cuda.manual_seed(RANDOM_SEED)
    DEVICE = os.environ.get("SET_GPU", "cuda:0")
    # GPU_ID = int(DEVICE.split(":")[1])
    print(f"Using GPU: {DEVICE}")
R2V_TRAINING_EPOCHS = {
    'pokec': 1,
    'small_pokec': 4,
    'airport': 100,
    'polbook': 200,
    'polblog': 200
}
NUM_GNN_LAYERS = {
    'pokec': 5,
    'small_pokec': 5,
    'airport': 4,
    'polbook': 4,
    'polblog': 4
}

NUM_NEGATIVE_SAMPLING = {
    'pokec': 1,
    'small_pokec': 1,
    'airport': 4,
    'polbook': 10,
    'polblog': 10
}

NUM_THREADS = {
    'pokec': 40,
    'small_pokec': 20,
    'airport': 20,
    'polbook': 20,
    'polblog': 20
}

NUM_WORKERS = 20
LOGGER = logging.getLogger(__name__)
LOGFORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
