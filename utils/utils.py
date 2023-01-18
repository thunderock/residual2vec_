# @Filename:    utils.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/31/22 5:32 PM
import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse
import torch
import itertools
import pandas as pd
from snakemake.utils import Paramspace
import string
import itertools


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


# Utilities
def param2paramDataFrame(param_list):
    if isinstance(param_list, list) is False:
        param_list = [param_list]
    my_dict = {}
    cols = []
    for dic in param_list:
        my_dict.update(dic)
        cols += list(dic.keys())
    keys, values = zip(*my_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    df = pd.DataFrame(permutations_dicts)
    df = df[cols]
    return df


def to_grid_paramspace(param_list):
    df = param2paramDataFrame(param_list)
    return Paramspace(df, filename_params="*")


def to_union_paramspace(param_list):
    df = pd.concat([param2paramDataFrame(l) for l in param_list])
    return Paramspace(df, filename_params="*")


def constrain_by(str_list):
    return "(" + ")|(".join(str_list) + ")"

def partial_format(filename, **params):
    field_names = [v[1] for v in string.Formatter().parse(filename) if v[1] is not None]
    fields = {field_name:"{"+field_name+"}" for field_name in field_names}
    for k,v in params.items():
        fields[k] = v
    return filename.format(**fields)

def to_list_value(params):
    for k, v in params.items():
        if isinstance(v, list):
            continue
        else:
            params[k]=[v]
    return params

def _expand(filename, **params):
    params = to_list_value(params)
    retval_filename = []
    keys, values = zip(*params.items())
    for bundle in itertools.product(*values):
        d = dict(zip(keys, bundle))
        retval_filename.append(partial_format(filename, **d))
    return retval_filename

def expand(filename, *args, **params):
    retval = []
    if len(args) == 0:
        return _expand(filename, **params)
    for l in args:
        retval += _expand(filename, **l, **params)
    return retval
