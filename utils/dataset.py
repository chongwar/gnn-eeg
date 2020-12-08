from tqdm import trange
import torch
from torch_geometric.data import Data, DataLoader

from .edges import *
from .features import gen_features_hvg, \
    gen_features_cre, gen_features_cre_group, gen_features_psd_group


def gen_data_list(data, label, edge_type='corr', feature_type='psd_group'):
    """
    Generate graph data list from matrix data and label.
    :param data: training or testing data in matrix form, shape: (N, T, C)
    :param label: training or testing label in matrix form, shape: (N, )
    :return: training or testing data list,
             each item in this list is a torch_geometric.data.Data object.
    """
    data_list = []
    for trial in trange(data.shape[0]):
        trial_data = data[trial, ...]
        trial_label = label[trial]

        # generate edge index and node features
        if edge_type == 'corr':
            edge_index, edge_weight = gen_edges_corr(trial_data)
        elif edge_type == 'wpli':
            edge_index, edge_weight = gen_edges_wpli(trial_data)
        elif edge_type == 'plv':
            edge_index, edge_weight = gen_edges_plv(trial_data)
        elif edge_type == 'cg':
            edge_index = gen_edges_cg(trial_data)
            edge_weight = np.zeros((edge_index.shape[-1], 1))

        if feature_type == 'hvg':
            x = gen_features_hvg(trial_data)
        elif feature_type == 'cre':
            x = gen_features_cre(trial_data)
        elif feature_type == 'cre_group':
            x = gen_features_cre_group(trial_data)
        elif feature_type == 'psd_group':
            x = gen_features_psd_group(trial_data)

        edge_index = torch.from_numpy(edge_index).long()
        edge_weight = torch.from_numpy(edge_weight).float()
        x = torch.from_numpy(x).float()

        graph_data = Data(x=x, edge_index=edge_index,
                          y=trial_label, edge_attr=edge_weight)
        data_list.append(graph_data)
    return data_list


def gen_dataloader(data_list, batch_size=32, shuffle=True):
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    pass
