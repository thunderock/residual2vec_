# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-02-02 11:59:32
# @Filepath: residual2vec/residual2vec_sgd.py
import random

import numpy as np
from numba import njit
from torch.optim import Adam
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from scipy import sparse
from residual2vec import utils
from residual2vec.random_walk_sampler import RandomWalkSampler
from residual2vec.word2vec import NegativeSampling
import wandb
from utils.config import DEVICE, DISABLE_TQDM, DISABLE_WANDB
import torch

class residual2vec_sgd:


    def __init__(
        self,
        noise_sampler,
        window_length=10,
        batch_size=256,
        num_walks=5,
        walk_length=40,
        p=1,
        q=1,
        cuda=DEVICE,
        buffer_size=100000,
        context_window_type="double",
        miniters=200,
    ):
        self.window_length = window_length
        self.sampler = noise_sampler
        self.cuda = cuda
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.miniters = miniters
        self.context_window_type = context_window_type

    # add feature matrix here
    def fit(self, adjmat=None):
        if not (sparse.isspmatrix(adjmat) or isinstance(adjmat, np.ndarray)):
            # dont need sampler in case of link prediction
            self.n_nodes = None
            return self
        # Convert to scipy.sparse.csr_matrix format
        adjmat = utils.to_adjacency_matrix(adjmat)

        # Set up the graph object for efficient sampling
        self.adjmat = adjmat
        self.n_nodes = adjmat.shape[0]
        self.sampler.fit(adjmat)
        return self

    def transform(self, model, dataloader: torch.utils.data.DataLoader, epochs=1, learning_rate=.001):
        """
        * model is the model to be used with the framework
        * x are the node features
        """

        neg_sampling = NegativeSampling(embedding=model)
        model.to(self.cuda)
        # Training
        optim = Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optim, base_lr=learning_rate * .5, max_lr=learning_rate*2, mode='exp_range', cycle_momentum=False, step_size_up=epochs // 10)

        # number of batches
        n_batches = len(dataloader)
        patience_threshold = n_batches // 2 # 50% of the batches
        print(f"Patience threshold: {patience_threshold}")
        if DISABLE_TQDM:
            epoch_range = trange(epochs, desc="training FINAL MODEL")
        else:
            epoch_range = range(epochs)
        for epoch in epoch_range:
            scheduler.step()
            agg_loss = 0
            break_loop = False
            patience = 0
            pbar = tqdm(dataloader, miniters=500, disable=DISABLE_TQDM)
            batch_num = 0
            for iword, owords, nwords in pbar:
                optim.zero_grad()
                for param in model.parameters():
                    param.grad = None
                loss = neg_sampling(iword, owords, nwords)
                if not torch.is_nonzero(loss):
                    patience += 1
                    if patience > patience_threshold:
                        break_loop = True
                        print("Early stopping {}, patience: {}".format(epoch, patience))
                        break
                loss.backward()
                optim.step()
                pbar.set_postfix(epoch=epoch)
                batch_num += 1
                agg_loss += loss.item()
            if not DISABLE_WANDB:
                wandb.log({"epoch": epoch, "loss": agg_loss / n_batches, "lr": scheduler.get_lr()[0]})
            if break_loop:
                break
        return self


class TripletSimpleDataset(Dataset):
    """Dataset for training word2vec with negative sampling."""

    def __init__(
        self,
        adjmat,
        num_walks,
        window_length,
        noise_sampler,
        padding_id,
        walk_length=40,
        p=1.0,
        q=1.0,
        context_window_type="double",
        buffer_size=100000,
    ):

        self.adjmat = adjmat

        self.num_walks = num_walks

        self.window_length = window_length
        self.noise_sampler = noise_sampler
        self.walk_length = walk_length
        self.padding_id = padding_id
        self.context_window_type = {"double": 0, "left": -1, "right": 1}[
            context_window_type
        ]
        self.rw_sampler = RandomWalkSampler(
            adjmat, walk_length=walk_length, p=p, q=q, padding_id=padding_id
        )
        self.node_order = np.random.choice(
            adjmat.shape[0], adjmat.shape[0], replace=False
        )
        self.n_nodes = adjmat.shape[0]

        self.ave_deg = adjmat.sum() / adjmat.shape[0]

        # Counter and Memory
        self.n_sampled = 0
        self.sample_id = 0
        self.scanned_node_id = 0
        self.buffer_size = buffer_size
        self.contexts = None
        self.centers = None
        self.random_contexts = None

        # Initialize
        self._generate_samples()

    def __len__(self):
        return self.n_nodes * self.num_walks * self.walk_length

    def __getitem__(self, idx):
        if self.sample_id == self.n_sampled:
            self._generate_samples()

        center = self.centers[self.sample_id]
        cont = self.contexts[self.sample_id].astype(np.int64)
        rand_cont = self.random_contexts[self.sample_id].astype(np.int64)

        self.sample_id += 1

        return center, cont, rand_cont

    def _generate_samples(self):
        next_scanned_node_id = np.minimum(
            self.scanned_node_id + self.buffer_size, self.n_nodes
        )
        walks = self.rw_sampler.sampling(
            self.node_order[self.scanned_node_id : next_scanned_node_id]
        )
        self.centers, self.contexts = _get_center_context(
            context_window_type=self.context_window_type,
            walks=walks,
            n_walks=walks.shape[0],
            walk_len=walks.shape[1],
            window_length=self.window_length,
            padding_id=self.padding_id,
        )
        self.random_contexts = self.noise_sampler.sampling(
            center_nodes=self.centers,
            context_nodes=self.contexts,
            padding_id=self.padding_id,
        )
        self.n_sampled = len(self.centers)
        self.scanned_node_id = next_scanned_node_id % self.n_nodes
        self.sample_id = 0


def _get_center_context(
    context_window_type, walks, n_walks, walk_len, window_length, padding_id
):
    """Get center and context pairs from a sequence
    window_type = {-1,0,1} specifies the type of context window.
    window_type = 0 specifies a context window of length window_length that extends both
    left and right of a center word. window_type = -1 and 1 specifies a context window
    that extends either left or right of a center word, respectively.
    """
    if context_window_type == 0:
        center, context = _get_center_double_context_windows(
            walks, n_walks, walk_len, window_length, padding_id
        )
    elif context_window_type == -1:
        center, context = _get_center_single_context_window(
            walks, n_walks, walk_len, window_length, padding_id, is_left_window=True
        )
    elif context_window_type == 1:
        center, context = _get_center_single_context_window(
            walks, n_walks, walk_len, window_length, padding_id, is_left_window=False
        )
    else:
        raise ValueError("Unknown window type")
    center = np.outer(center, np.ones(context.shape[1]))
    center, context = center.reshape(-1), context.reshape(-1)
    s = (center != padding_id) * (context != padding_id)
    center, context = center[s], context[s]
    order = np.arange(len(center))
    random.shuffle(order)
    return center[order].astype(int), context[order].astype(int)


@njit(nogil=True)
def _get_center_double_context_windows(
    walks, n_walks, walk_len, window_length, padding_id
):
    centers = padding_id * np.ones(n_walks * walk_len, dtype=np.int64)
    contexts = padding_id * np.ones(
        (n_walks * walk_len, 2 * window_length), dtype=np.int64
    )
    for t_walk in range(walk_len):
        start, end = n_walks * t_walk, n_walks * (t_walk + 1)
        centers[start:end] = walks[:, t_walk]

        for i in range(window_length):
            if t_walk - 1 - i < 0:
                break
            contexts[start:end, window_length - 1 - i] = walks[:, t_walk - 1 - i]

        for i in range(window_length):
            if t_walk + 1 + i >= walk_len:
                break
            contexts[start:end, window_length + i] = walks[:, t_walk + 1 + i]

    return centers, contexts


@njit(nogil=True)
def _get_center_single_context_window(
    walks, n_walks, walk_len, window_length, padding_id, is_left_window=True
):
    centers = padding_id * np.ones(n_walks * walk_len, dtype=np.int64)
    contexts = padding_id * np.ones((n_walks * walk_len, window_length), dtype=np.int64)
    for t_walk in range(walk_len):
        start, end = n_walks * t_walk, n_walks * (t_walk + 1)
        centers[start:end] = walks[:, t_walk]

        if is_left_window:
            for i in range(window_length):
                if t_walk - 1 - i < 0:
                    break
                contexts[start:end, window_length - 1 - i] = walks[:, t_walk - 1 - i]
        else:
            for i in range(window_length):
                if t_walk + 1 + i >= walk_len:
                    break
                contexts[start:end, i] = walks[:, t_walk + 1 + i]
    return centers, contexts


