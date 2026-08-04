"""
Per-neuron gradient-Hessian-proxy eigenvalue analysis: the growth
mechanism decides *which* neurons in a layer are "saturated" (and
therefore worth splitting) by looking at the minimum eigenvalue of a
rank-1 outer-product proxy for each neuron's gradient. Negative minimum
eigenvalues flag neurons sitting at a saddle point.

Split out of the original top-level ``growth_utils_node_new.py``.
"""
import gc
import os

import numpy as np
import torch
from joblib import Parallel, delayed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

from growing_nn.growth.layers import get_all_linear_layers_transformer


def split_matrix(gradient, weight=None):
    """Build the rank-1 outer-product proxy matrix used to approximate a
    neuron's local curvature from its gradient.
    """

    def second_order_derivative():
        return torch.ones(gradient.shape)

    sm = gradient.view(1, -1).cpu() * second_order_derivative().view(-1, 1)
    return sm


def calculate_min_eig(gradient):
    """Minimum eigenvalue of the ``split_matrix`` proxy for one neuron's
    gradient. Returns ``0`` if the eigendecomposition fails.
    """
    try:
        splitting = split_matrix(gradient)
        eig, _ = torch.linalg.eig(splitting.cpu())
        min_eig = torch.min(eig.cpu().double())
        del eig, splitting
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        min_eig = 0
    return min_eig


def calc_all_eigs(layers, n_jobs=24):
    """Compute the per-neuron minimum eigenvalue (see
    :func:`calculate_min_eig`) for every neuron in every layer of
    ``layers``, in parallel via joblib.

    Returns a list (one entry per layer) of lists (one entry per neuron).
    """
    eigs = []
    for i, layer in enumerate(layers):
        gradient = layer.weight.grad.cpu().detach().clone()
        s = gradient.shape[0]
        min_eigs = Parallel(n_jobs=n_jobs)(delayed(calculate_min_eig)(gradient[i]) for i in range(s))
        eigs.append(min_eigs)
        del gradient
        gc.collect()
        torch.cuda.empty_cache()
    return eigs


def ret_flattened(eig):
    """Flatten the per-layer eigenvalue lists returned by
    :func:`calc_all_eigs` into a single list plus a parallel list of
    "which layer did this come from" indices.
    """
    flat_eig = []
    flat_eig_layer = []
    for i, e in enumerate(eig):
        flat_eig += e
        flat_eig_layer += [i for j in range(len(e))]
    return flat_eig, flat_eig_layer


def layer_negative(eigs):
    """For each layer, collect the neuron indices with a negative minimum
    eigenvalue, sorted from most-negative to least-negative.
    """
    neg_index_dic = {}
    neg_eig_dic = {}
    for layer, eig in enumerate(eigs):
        for j, e in enumerate(eig):
            if e < 0:
                if str(layer) in neg_index_dic:
                    neg_index_dic[str(layer)].append(j)
                    neg_eig_dic[str(layer)].append(e)
                else:
                    neg_index_dic[str(layer)] = [j]
                    neg_eig_dic[str(layer)] = [e]

    for k in neg_index_dic.keys():
        sort_pos = np.argsort(np.array(neg_eig_dic[k]))
        neg_index_dic[k] = np.array(neg_index_dic[k])[sort_pos]
    return neg_index_dic


def plot_eig(eig, epoch):
    """Plot the minimum eigenvalues for every layer at a given epoch to
    ``eig/{epoch}/{layer}.jpg``.
    """
    for k in range(len(eig)):
        y = eig[k]
        cn = 0
        for aa in y:
            if aa < 0:
                cn += 1
        x = [i for i in range(len(y))]
        plt.scatter(x, y)
        plt.title(f"Layer {k} count_neg ={cn}")
        plt.savefig(f"eig/{epoch}/{k}.jpg")
        plt.clf()


def calculate_eig(model, epoch):
    """Convenience wrapper: compute and plot all per-neuron eigenvalues
    for ``model`` at ``epoch``. Used from the training loop.
    """
    l, la = get_all_linear_layers_transformer(model)
    eigs = calc_all_eigs(l)

    os.makedirs(f"eig/{epoch}", exist_ok=True)
    plot_eig(eigs, epoch)
