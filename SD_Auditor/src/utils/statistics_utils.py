import numpy as np


def sample_split_three(n):
    index = np.arange(n)
    np.random.shuffle(index)
    index1 = index[:n // 3]
    index2 = index[n // 3:2 * n // 3]
    index3 = index[2 * n // 3:]
    return index1, index2, index3

def sample_split_two(n):
    index = np.arange(n)
    np.random.shuffle(index)
    index1 = index[:n // 2]
    index2 = index[n // 2:]
    return index1, index2