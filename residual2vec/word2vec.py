import numpy as np
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, learn_outvec=True):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(
            self.vocab_size, self.embedding_size, padding_idx=padding_idx
        )
        self.ovectors = nn.Embedding(
            self.vocab_size, self.embedding_size, padding_idx=padding_idx
        )
        self.ivectors.weight = nn.Parameter(
            torch.cat(
                [
                    torch.zeros(1, self.embedding_size),
                    FloatTensor(self.vocab_size, self.embedding_size).uniform_(
                        -0.5 / self.embedding_size, 0.5 / self.embedding_size
                    ),
                ]
            )
        )
        self.ovectors.weight = nn.Parameter(
            torch.cat(
                [
                    torch.zeros(1, self.embedding_size),
                    FloatTensor(self.vocab_size, self.embedding_size).uniform_(
                        -0.5 / self.embedding_size, 0.5 / self.embedding_size
                    ),
                ]
            )
        )
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True
        self.learn_outvec = learn_outvec

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = LongTensor(data)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        if not self.learn_outvec:
            return self.forward_i(data)
        v = LongTensor(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)


class NegativeSampling(nn.Module):
    def __init__(self, embedding):
        super(NegativeSampling, self).__init__()
        self.embedding = embedding
        self.weights = None
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, iword, owords, nwords):
        # these can be tuple of tensors or single tensor
        ivectors = self.embedding.forward_i(iword)
        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords)
        # print("shapes: ", ivectors.shape, ovectors.shape, nvectors.shape)
        oloss = self.logsigmoid((ovectors * ivectors).sum(dim=1))
        nloss = self.logsigmoid((nvectors * ivectors).sum(dim=1).neg())
        return -(oloss + nloss).mean()