import os
from os.path import join as j


class FileResources(object):
    def __init__(self, root: str, crosswalk: bool, baseline: bool, model_name:str, basename: str='pokec'):
        self.root = root
        self.crosswalk = crosswalk
        self.baseline = baseline
        self.basename = basename
        assert model_name in ['gat', 'gcn']
        self.model_name = model_name

    @property
    def adj_path(self): return str(j(self.root, "{}_crosswalk_adj.npz".format(self.basename)))

    @property
    def node2vec_weights(self):
        if self.baseline:
            if self.crosswalk:
                return str(j(self.root, "{}_crosswalk_{}_nodevec.h5".format(self.basename, self.model_name)))
            else:
                return str(j(self.root, "{}_{}_node2vec.h5".format(self.basename, self.model_name)))
        else:
            if self.crosswalk:
                return str(j(self.root, "{}_crosswalk_{}_r2v_node2vec.h5".format(self.basename, self.model_name)))
            else:
                return str(j(self.root, "{}_{}_r2v_node2vec.h5".format(self.basename, self.model_name)))

    @property
    def model_weights(self):
        if self.baseline:
            if self.crosswalk:
                return str(j(self.root, "{}_crosswalk_{}_nodevec.h5".format(self.basename, self.model_name)))
            else:
                return str(j(self.root, "{}_{}_nodevec.h5".format(self.basename, self.model_name)))
        else:
            if self.crosswalk:
                return str(j(self.root, "{}_crosswalk_{}_r2v_nodevec.h5".format(self.basename, self.model_name)))
            else:
                return str(j(self.root, "{}_{}_r2v_nodevec.h5".format(self.basename, self.model_name)))

    @property
    def embs_file(self):
        if self.baseline:
            if self.crosswalk:
                return str(j(self.root, "{}_crosswalk_{}_node2vec_embs.npy".format(self.basename, self.model_name)))
            else:
                return str(j(self.root, "{}_{}_node2vec_embs.npy".format(self.basename, self.model_name)))
        else:
            if self.crosswalk:
                return str(j(self.root, "{}_crosswalk_{}_r2v_node2vec_embs.npy".format(self.basename, self.model_name)))
            else:
                return str(j(self.root, "{}_{}_r2v_node2vec_embs.npy".format(self.basename, self.model_name)))
