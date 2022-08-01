# @Filename:    config.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/31/22 5:59 PM
import torch

CUDA = torch.cuda.is_available()
TRAIN_TEST_SPLIT = .2
RANDOM_SEED = 2022
DROPOUT = .5
torch.manual_seed(RANDOM_SEED)
if CUDA:
    torch.cuda.manual_seed(RANDOM_SEED)

