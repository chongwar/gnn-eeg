import torch
from torch_geometric.nn import TAGConv
from torch_geometric.nn import global_mean_pool as gap, \
    global_max_pool as gmp, global_sort_pool as gsp
import torch.nn.functional as F
import math


class TAGSortPool(torch.nn.Module):
    """
    The topology adaptive graph convolutional networks operator from the
    "Topology Adaptive Graph Convolutional Networks"
    <https://arxiv.org/abs/1710.10370>
    """

    def __init__(self, num_features, channels=64):
        super(TAGSortPool, self).__init__()
        self.k = 12
        self.conv1 = TAGConv(num_features, 4)
        self.conv2 = TAGConv(4, 4)
        self.conv1d = torch.nn.Conv1d(4, 4, self.k)
        self.fc = torch.nn.Linear(4, 2)

        num_edges = channels * channels - channels
        self.edge_weight = torch.nn.Parameter(torch.FloatTensor(num_edges, 1),
                                              requires_grad=True)
        self.edge_weight.data.fill_(1)

    def forward(self, data):
        x, edge_index, batch, edge_weight = data.x, data.edge_index, \
                                            data.batch, self.edge_weight
        _edge_weight = edge_weight
        for i in range(edge_index.shape[-1] // edge_weight.shape[0] - 1):
            edge_weight = torch.cat((edge_weight, _edge_weight), dim=0)

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))

        x = gsp(x, batch, self.k)
        x = x.view(len(x), self.k, -1).permute(0, 2, 1)
        x = F.relu(self.conv1d(x))
        x = x.view(len(x), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    tag_learn = TAGSortPool(5)
    print(tag_learn)
