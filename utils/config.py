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
    'pokec': 3,
    
    'airport': 100 * 3,
    'polbook': 200 * 3,
    'polblog': 200 * 3, 
    'facebook': 25 * 3,
    'copenhagen': 200 * 3,
    "twitch": 25 * 3
}
NUM_GNN_LAYERS = {
    'pokec': 5,
    'small_pokec': 5,
    'facebook': 4,
    'airport': 3,
    'polbook': 3,
    'polblog': 3,
    'copenhagen': 3,
    'twitch': 4
}

NUM_NEGATIVE_SAMPLING = {
    'pokec': 1,
    'small_pokec': 1,
    'airport': 4,
    'polbook': 10,
    'polblog': 10,
    'copenhagen': 10,
    'facebook': 1,
    'twitch': 1
}

NUM_THREADS = {
    'pokec': 30,
    'small_pokec': 20,
    'airport': 20,
    'polbook': 20,
    'polblog': 20,
    'facebook': 20,
    'copenhagen': 20,
    'twitch': 20
}

TEST_SPLIT_FRAC = {
    'pokec': .5,
    'small_pokec': .5,
    'airport': .45,
    'polbook': .45,
    'polblog': .45,
    'facebook': .45,
    'copenhagen': .45,
    'twitch': .45
}

NUM_WORKERS = 20
LOGGER = logging.getLogger(__name__)
LOGFORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
