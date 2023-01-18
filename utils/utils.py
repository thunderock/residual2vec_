# @Filename:    utils.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/31/22 5:32 PM
import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse
import torch


class CONSTANTS:
    NLL_LOSS = torch.nn.NLLLoss()
    BCE_LOSS = torch.nn.BCEWithLogitsLoss()


def sparse_to_torch_tensor(sp_mt):
    """
    converts a scipy sparse matrix to a torch tensor
    """
    sp_mt = sp_mt.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sp_mt.row, sp_mt.col))).long()
    values = torch.from_numpy(sp_mt.data)
    shape = torch.Size(sp_mt.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_row_wise(mx):
    """
    Normalize sparse matrix by row
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def check_if_sparse_symmetric(A):
    sym_err = A - A.T
    return np.all(np.abs(sym_err.data) < 1e-10)


def check_if_symmetric(m):
    """
    Check if a matrix is symmetric
    """
    if issparse(m):
        return check_if_sparse_symmetric(m)
    return np.allclose(m, m.T)