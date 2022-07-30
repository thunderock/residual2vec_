# @Filename:    score.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/29/22 9:46 PM
import numpy as np
import networkx as nx
from tqdm import trange

def statistical_parity(G, y):
    classes, counts = np.unique(y, return_counts=True)
    # check if G is graph
    assert isinstance(G, nx.classes.graph.Graph)
    # check if these are labels
    assert np.alltrue(y >= 0) and np.alltrue(y < counts.shape[0]) and y.dtype in [np.int64, np.int32]
    # class_counts = {i: j for i, j in zip(classes, counts)}

    score = {}

    for class_i in trange(len(classes)):
        class_i_vertices = set(np.where(y == class_i)[0])
        class_scores = np.zeros(len(class_i_vertices))
        # Statistical Parity
        # (typically defined in terms of two groups) requires the accep-
        # tance rates of the candidates from both groups to be equal
        for idx, i in enumerate(class_i_vertices):
            s, not_s, total = 0., 0., 0.
            for _, v in G.edges([i]):
                total += 1.
                if v in class_i_vertices:
                    s += 1
                else:
                    not_s += 1
            if total == 0.:
                continue
            p_y_s = s / total
            # or 1 - pys
            p_y_not_s = not_s / total
            class_scores[idx] = p_y_s - p_y_not_s
        # To capture the differences between
        # multiple groups, we calculate the variance between the ac-
        # ceptance (recommendation) rates of each group in GS
        score[class_i] = np.var(np.abs(class_scores))
        # print(f"Class {class_i} has score {score[class_i]}")
    return np.mean(list(score.values()))
