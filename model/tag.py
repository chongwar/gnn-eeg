import torch
from torch_geometric.nn import TAGConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F


class TAG(torch.nn.Module):
    """
    The topology adaptive graph convolutional networks operator from the
    "Topology Adaptive Graph Convolutional Networks"
    <https://arxiv.org/abs/1710.10370>
    """

    def __init__(self, num_features):
        super(TAG, self).__init__()
        self.conv1 = TAGConv(num_features, 8)
        self.conv2 = TAGConv(8, 16)

        # self.fc = torch.nn.Linear(2 * 16, 1)
        self.fc = torch.nn.Linear(2 * 16, 2)

    def forward(self, data):
        x, edge_index, batch, edge_weight = data.x, data.edge_index, \
                                            data.batch, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))

        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # x = gmp(x, batch)
        # x = gap(x, batch)

        # x = torch.sigmoid(self.fc(x)).squeeze(1)
        x = self.fc(x)

        return x
