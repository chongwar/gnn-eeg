import torch
from torch_geometric.nn import TAGConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import math


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / tensor.size(-2) + tensor.size(-1))
        tensor.data.uniform_(-stdv, stdv)


class TAGLearn(torch.nn.Module):
    """
    The topology adaptive graph convolutional networks operator from the
    "Topology Adaptive Graph Convolutional Networks"
    <https://arxiv.org/abs/1710.10370>
    """

    def __init__(self, num_features, channels=64):
        super(TAGLearn, self).__init__()
        self.conv1 = TAGConv(num_features, 8)
        self.conv2 = TAGConv(8, 8)

        self.fc = torch.nn.Linear(2 * 8, 2)
        # self.fc = torch.nn.Linear(8, 2)

        num_edges = channels * channels - channels
        self.edge_weight = torch.nn.Parameter(torch.FloatTensor(num_edges, 1),
                                              requires_grad=True)
        self.edge_weight.data.fill_(1)
        # glorot(self.edge_weight)
        # print(self.edge_weight)

    def forward(self, data):
        x, edge_index, batch, edge_weight = data.x, data.edge_index, \
                                            data.batch, self.edge_weight
        # print('\n', '-' * 25)
        # print('edge_index:', edge_index.shape, edge_index.requires_grad)
        # print('edge_weight:', edge_weight.shape, edge_weight.requires_grad)
        # print('*' * 25, '\n', edge_weight)
        _edge_weight = edge_weight
        # print('_edge_weight:', _edge_weight.shape, _edge_weight.requires_grad)
        for i in range(edge_index.shape[-1] // edge_weight.shape[0] - 1):
            edge_weight = torch.cat((edge_weight, _edge_weight), dim=0)
        # print('edge_weight:', edge_weight.shape, edge_weight.requires_grad, edge_weight)

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))

        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # x = gmp(x, batch)
        # x = gap(x, batch)

        # x = torch.sigmoid(self.fc(x)).squeeze(1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    tag_learn = TAGLearn(5)
    print(tag_learn)
