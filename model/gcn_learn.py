import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F


class GCNLearn(torch.nn.Module):
    """
    The graph convolutional operator from the "Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.09207>
    """

    def __init__(self, num_features, channels=64):
        super(GCNLearn, self).__init__()
        self.conv1 = GCNConv(num_features, 8, add_self_loops=False)
        self.conv2 = GCNConv(8, 8, add_self_loops=False)

        # self.fc = torch.nn.Linear(2 * 16, 1)
        self.fc = torch.nn.Linear(2 * 8, 2)

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

        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # x = torch.sigmoid(self.fc(x)).squeeze(1)
        x = self.fc(x)

        return x
