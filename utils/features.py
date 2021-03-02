import warnings
import networkx as nx
import numpy as np
from pywt import wavedec
from scipy import signal
from dit import Distribution as Dist
from dit.other import cumulative_residual_entropy as cre
from sklearn import preprocessing
from utils.vg import visibility
from scipy.signal import welch
from scipy.integrate import simps


warnings.filterwarnings('ignore')


def sos_filter(data, f_1, f_2, fs=256):
    # data: (T, C)
    # wn = [f_1 * 2 / fs, f_2 * 2 / fs]
    sos = signal.butter(3, Wn=[f_1, f_2], btype='bandpass', output='sos', fs=fs)
    return signal.sosfilt(sos, data, axis=0)


def gen_hvg_edges(nodes):
    """
    Generate edges from the horizontal visibility graph.
    :param nodes: time series of the specific node
    :return: edges list
    """
    edges = []
    for i in range(len(nodes) - 1):
        a_idx, a_val = nodes[i]
        b_idx, b_val = nodes[i + 1]
        edges.append([a_idx, b_idx])

    for i in range(0, len(nodes) - 1):
        a_idx, a_val = nodes[i]
        for j in range(i + 2, len(nodes)):
            b_idx, b_val = nodes[j]
            visible = True
            for c_idx, c_val in nodes:
                if c_idx > a_idx and c_idx < b_idx:
                    if c_val > a_val or c_val > b_val:
                        visible = False
                        break
            if visible:
                edges.append([a_idx, b_idx])
    return edges


def gen_features_hvg(x):
    """
    Generate node features based on hvg of the node.
    :param x: (T, C)
    :return: normalized node features, shape: (C, F)
             C denotes the number of nodes,
             F denotes the number of node features.
    """
    features = []
    for i in range(x.shape[1]):
        # # use original hvg algorithm
        # time_series = x[:, i]
        # nodes = [(i, j) for i, j in zip(range(0, len(time_series)), time_series)]
        # edges = gen_hvg_edges(nodes=nodes)

        # use bst
        _edges = visibility(x[:, i])
        edges = [[i[0], j] for i in _edges for j in i[1]]

        hvg = nx.Graph()
        hvg.add_edges_from(edges)

        # use average and average clustering coefficient as node features
        deg_avg = np.average([j for i, j in hvg.degree])
        clus_coef_avg = np.average([j for i, j in nx.clustering(hvg).items()])

        features.append([deg_avg, clus_coef_avg])

    features = preprocessing.scale(np.array(features))
    return features


def gen_features_cre(x):
    """
    Generate node features based on cumulative residual entropy.
    The cumulative residual entropy is an alternative to the Shannon differential entropy
    with several desirable properties including non-negativity.
    :param x: (T, C)
    :return: normalized node features, shape: (C, F)
             C denotes the number of nodes,
             F denotes the number of node features.
    """
    samples, _ = x.shape
    prob = [1 / samples] * samples
    x = [tuple(i.tolist()) for i in x]
    dist = Dist(x, prob)
    features = cre(dist)
    features = preprocessing.scale(features)
    return features.reshape(-1, 1)


def gen_features_cre_group(x):
    """
    Generate node features based on band power spectral density.
    :param x: (T, C)
    :return: normalized node features, shape: (C, F)
             C denotes the number of nodes,
             F denotes the number of node features.
    """
    samples, _ = x.shape
    prob = [1 / samples] * samples

    x_delta = sos_filter(x, 1, 4)
    x_theta = sos_filter(x, 4, 8)
    x_alpha = sos_filter(x, 8, 12)
    x_beta = sos_filter(x, 12, 30)
    x_gamma = sos_filter(x, 30, 100)
    features = None
    for x_tmp in [x_delta, x_theta, x_alpha, x_beta, x_gamma]:
        x_tmp = [tuple(i.tolist()) for i in x_tmp]
        dist = Dist(x_tmp, prob)
        feature_bp = cre(dist).reshape(-1, 1)
        if features is None:
            features = feature_bp
        else:
            features = np.hstack([features, feature_bp])

    features = preprocessing.scale(features)
    return features


def gen_psd_bp(x, fs, band, relative=True):
    """
    Computer the average the power of the signal x in a specific
    frequency band. (https://raphaelvallat.com/bandpower.html)
    :param x: single channel EEG data
    :param fs: sampling frequency
    :param band: frequency band
    :param relative: True/False (return the relative psd or not)
    :return: absolute or relative psd
    """
    band = np.asarray(band)
    low, high = band

    nperseg = (2 / low) * fs
    freqs, psd = welch(x, fs, nperseg=nperseg)

    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    psd_bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        psd_bp /= simps(psd, dx=freq_res)
    return psd_bp


def gen_features_psd(x):
    """
    Generate node features based on power spectral density.
    :param x: (T, C)
    :return: normalized node features, shape: (C, F)
             C denotes the number of nodes,
             F denotes the number of node features.
    """
    features = []
    for i in range(x.shape[1]):
        tmp_psd = gen_psd_bp(x[:, i], fs=256, band=[1, 40])
        features.append([tmp_psd])

    features = preprocessing.scale(np.array(features))
    return features


def gen_features_psd_group(x):
    """
    Generate node features based on band power spectral density.
    :param x: (T, C)
    :return: normalized node features, shape: (C, F)
             C denotes the number of nodes,
             F denotes the number of node features.
    """
    features = []
    for i in range(x.shape[1]):
        psd_delta = gen_psd_bp(x[:, i], fs=256, band=[1, 4])
        psd_theta = gen_psd_bp(x[:, i], fs=256, band=[4, 8])
        psd_alpha = gen_psd_bp(x[:, i], fs=256, band=[8, 12])
        psd_beta = gen_psd_bp(x[:, i], fs=256, band=[12, 30])
        psd_gamma = gen_psd_bp(x[:, i], fs=256, band=[30, 45])
        features.append([psd_delta, psd_theta, psd_alpha, psd_beta, psd_gamma])

    features = np.array(features)
    features = preprocessing.scale(features)
    return features


def gen_features_wavelet(x, wavelet='db4', level=4, axis=0):
    """
    Generate node features based on wavelet decomposition.
    :param x: (T, C)
    :param wavelet: wavelet function (default: 'db4)
    :param level: decomposition level (default: 4)
    :param axis: axis over which to compute the DWT
    :return: normalized node features, shape: (C, F)
             C denotes the number of nodes,
             F denotes the number of node features.
    """

    coeffs = wavedec(x, wavelet=wavelet, level=level, axis=axis)
    # approximation coefficients of the forth level of decomposition
    cAi = coeffs[0]
    features = cAi.T
    features = preprocessing.scale(features)
    return features


def gen_features_wt_deg(x, wavelet='coif1', level=4, axis=0):
    """
    Generate node features based on degree after wavelet decomposition.
    :param x: (T, C)
    :param wavelet: wavelet function (default: 'db4)
    :param level: decomposition level (default: 4)
    :param axis: axis over which to compute the DWT
    :return: normalized node features, shape: (C, F)
             C denotes the number of nodes,
             F denotes the number of node features.
    """
    features = []
    coeffs = wavedec(x, wavelet=4, level=level, axis=axis)
    # approximation coefficients of the level-th decomposition
    cAi = coeffs[0]
    for ch in range(cAi.shape[1]):
        wt_series = cAi[:, ch]
        # wt_series = wt_series - min(wt_series) + 0.1

        _edges = visibility(wt_series)
        edges = [[i[0], j] for i in _edges for j in i[1]]
        hvg = nx.Graph()
        hvg.add_edges_from(edges)
        degree = [(i, j) for i, j in hvg.degree]
        degree.sort()
        features.append([j for i, j in degree])

    features = np.asarray(features)
    # features = preprocessing.scale(features)
    return features


def gen_features_raw(x):
    """
    Generate node features using raw data.
    :param x: (T, C)
    """
    # x = x[range(0, x.shape[0], 2), :]
    features = x.T
    return features


if __name__ == '__main__':
    data = np.load('../data/s12/x_2.npy')
    _x = data[0, range(0, 1024, 4), :]

    # time_series = x[range(0, 1024), 0]
    # nodes = [(i, j) for i, j in zip(range(0, len(time_series)), time_series)]
    # edges = gen_hvg_edges(nodes)
    # print(sorted(edges))
    # print(len(edges))

    # features = gen_features_hvg(_x)
    # features = gen_features_cre(_x)
    # features = gen_features_psd_group(_x)
    # features = gen_features_cre_group(_x)
    # features = gen_features_wavelet(_x)
    # features = gen_features_raw(_x)
    features = gen_features_wt_deg(_x)
    print(features.shape)
    print(features)
