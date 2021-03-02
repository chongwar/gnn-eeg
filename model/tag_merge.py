import torch
from torch_geometric.nn import TAGConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import numpy as np

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TAGMerge(torch.nn.Module):
    """
    The topology adaptive graph convolutional networks operator from the
    "Topology Adaptive Graph Convolutional Networks"
    <https://arxiv.org/abs/1710.10370>
    """

    def __init__(self, num_features, channels=64):
        super(TAGMerge, self).__init__()
        self.num_channels = channels

        self.conv_prior_1 = TAGConv(num_features, 4)
        self.conv_prior_2 = TAGConv(4, 8)

        self.conv_learn_1 = TAGConv(num_features, 4)
        self.conv_learn_2 = TAGConv(4, 8)

        self.fc = torch.nn.Linear(8, 2)

        num_edges = channels * channels - channels
        self.edge_weight_learn = torch.nn.Parameter(torch.FloatTensor(num_edges, 1),
                                              requires_grad=True)

        self.edge_weight_learn.data.fill_(1)
        self.edge_index_learn = self.gen_edges_cg(self.num_channels)
        self.edge_index_learn = torch.from_numpy(self.edge_index_learn).long()

    def forward(self, data):
        x, edge_index_prior, batch, edge_weight_prior = data.x, data.edge_index, \
                                                        data.batch, data.edge_attr

        edge_weight_learn = self.edge_weight_learn
        edge_index_learn = self.edge_index_learn
        batch_size = x.shape[0] // self.num_channels

        _edge_weight_learn = edge_weight_learn
        for i in range(batch_size - 1):
            edge_weight_learn = torch.cat((edge_weight_learn, _edge_weight_learn), dim=0)
        edge_index_learn = edge_index_learn.repeat(1, batch_size).to(device)

        # calculate the prior graph representation
        x_prior = F.relu(self.conv_prior_1(x, edge_index_prior, edge_weight_prior))
        x_prior = F.relu(self.conv_prior_2(x_prior, edge_index_prior, edge_weight_prior))
        x_prior = gap(x_prior, batch)

        # calculate the learned graph representation
        x_learn = F.relu(self.conv_learn_1(x, edge_index_learn, edge_weight_learn))
        x_learn = F.relu(self.conv_learn_2(x_learn, edge_index_learn, edge_weight_learn))
        x_learn = gap(x_learn, batch)

        x = x_prior + x_learn
        # x = torch.cat([x_prior, x_learn], dim=1)

        x = self.fc(x)
        return x

    def gen_edges_cg(self, num_channels):
        """
        Generate edges based on complete graph
        :param num_channels: number of channels
        :return: edge_index for
        """
        edge_index = [[i, j] for i in range(num_channels) for j in range(num_channels)
                      if i != j]
        edge_index = np.asarray(edge_index).T
        return edge_index


if __name__ == '__main__':
    tag_learn = TAGMerge(5)
    print(tag_learn)
