import numpy as np
import matplotlib.mlab as mlab
from scipy.signal import hilbert


def gen_edges_corr(x, weighted=True):
    """
    Generate weighted(optional) edges based on channel correlation coefficient
    :param x: (T, C)
    :param weighted: True or False
    :return: edge_index: (2, num_edges)
             edge_weight:(num_edges, 1)
    """
    adj = np.corrcoef(x.T)
    adj[range(adj.shape[0]), range(adj.shape[0])] = 0
    avg = np.sum(adj) / (adj.shape[0] * adj.shape[0] - adj.shape[0])
    zeros_index = np.argwhere(adj <= avg)
    adj[zeros_index[:, 0], zeros_index[:, 1]] = 0

    edge_index = np.argwhere(adj != 0).T

    if weighted:
        edge_weight = adj[edge_index[0, :], edge_index[1, :]].reshape(-1, 1)
        return edge_index, edge_weight
    else:
        return edge_index


def gen_edges_wpli(x, fs=256, weighted=True):
    """
    Generate weighted(optional) edges based on weighted phase lax index
    :param x: (T, C)
    :param weighted: True or False
    :return: edge_index: (2, num_edges)
             edge_weight:(num_edges, 1)
    """
    x = x.T
    channels, samples = x.shape
    wpli = np.zeros((channels, channels))

    pairs = [(i, j) for i in range(channels) for j in range(channels)]

    for pair in pairs:
        ch1, ch2 = x[pair,]
        csdxy, _ = mlab.csd(ch1, ch2, Fs=fs, scale_by_freq=True,
                            sides='onesided')

        i_xy = np.imag(csdxy)
        num = np.nansum(np.abs(i_xy) * np.sign(i_xy))
        denom = np.nansum(np.abs(i_xy))

        wpli[pair] = np.abs(num / denom)

    adj = wpli
    adj[range(adj.shape[0]), range(adj.shape[0])] = 0
    avg = np.sum(adj) / (adj.shape[0] * adj.shape[0] - adj.shape[0])
    zeros_index = np.argwhere(adj <= avg)
    adj[zeros_index[:, 0], zeros_index[:, 1]] = 0

    edge_index = np.argwhere(adj != 0).T

    if weighted:
        edge_weight = adj[edge_index[0, :], edge_index[1, :]].reshape(-1, 1)
        return edge_index, edge_weight
    else:
        return edge_index


def gen_edges_plv(x, weighted=True):
    """
    Generate weighted(optional) edges based on phase locking value
    :param x: (T, C)
    :param weighted: True or False
    :return: edge_index: (2, num_edges)
             edge_weight:(num_edges, 1)
    """
    x = x.T
    channels, samples = x.shape
    x_h = hilbert(x)
    phase = np.unwrap(np.angle(x_h))

    Q = np.exp(1j * phase)
    Q = np.matrix(Q)
    W = np.abs(Q @ Q.conj().transpose()) / np.float32(samples)

    adj = W
    adj[range(adj.shape[0]), range(adj.shape[0])] = 0
    avg = np.sum(adj) / (adj.shape[0] * adj.shape[0] - adj.shape[0])
    zeros_index = np.argwhere(adj <= avg)
    adj[zeros_index[:, 0], zeros_index[:, 1]] = 0

    edge_index = np.argwhere(adj != 0).T

    if weighted:
        edge_weight = adj[edge_index[0, :], edge_index[1, :]].reshape(-1, 1)
        return edge_index, edge_weight
    else:
        return edge_index


def gen_edges_cg(x):
    """
    Generate edges based on complete graph
    :param x: (T, C)
    :return: edge_index: (2, C * C - C)
    """
    samples, channels = x.shape
    edge_index = [[i, j] for i in range(channels) for j in range(channels)
                  if i != j]
    edge_index = np.asarray(edge_index).T
    return edge_index


if __name__ == '__main__':
    data = np.load('../data/s12/x_2.npy')
    _x = data[0, range(0, 1024, 4), :]
    # np.random.seed(3)
    # x = np.random.rand(2, 4)
    # edge_index_corr = gen_edges_corr(_x)
    # edge_index_wpli = gen_edges_wpli(_x)
    # edge_index_plv = gen_edges_plv(_x)
    edge_index_cg = gen_edges_cg(_x)
    # print(edge_index_corr)
    # print(edge_index_wpli)
    # print(edge_index_plv)
    print(edge_index_cg, edge_index_cg.shape)
