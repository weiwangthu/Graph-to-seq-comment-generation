import numpy as np


def find_norm(syn0):
    syn0norm = (syn0 / np.sqrt((syn0 ** 2).sum(-1))[..., np.newaxis]).astype(np.float32)
    return syn0norm


def argsort(x, topn=None, reverse=False):
    """
    Return indices of the `topn` smallest elements in array `x`, in ascending order.
    If reverse is True, return the greatest elements instead, in descending order.
    """
    x = np.asarray(x)  # unify code path for when `x` is not a numpy array (list, tuple...)
    if topn is None:
        topn = x.size
    if topn <= 0:
        return []
    if reverse:
        x = -x
    if topn >= x.size or not hasattr(np, 'argpartition'):
        return np.argsort(x)[:topn]
    # numpy >= 1.8 has a fast partial argsort, use that!
    most_extreme = np.argpartition(x, topn)[:topn]
    return most_extreme.take(np.argsort(x.take(most_extreme)))  # resort topn into order


def find_similar(des_norm, vec_norm):
    dists = np.dot(des_norm, vec_norm)

    best = argsort(dists, reverse=True)

    dist_sort = np.sort(dists)[::-1]

    return dist_sort, best

