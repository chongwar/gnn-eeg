import torch
from torch_geometric.nn import SGConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F


class SGCN(torch.nn.Module):
    """
    The simple graph convolutional operator from the "Simplifying Graph
    Convolutional Networks"
    <https://arxiv.org/abs/1902.07153>
    """

    def __init__(self, num_features):
        super(SGCN, self).__init__()
        self.conv1 = SGConv(num_features, 8, K=2)
        self.conv2 = SGConv(8, 16, K=2)

        # self.fc = torch.nn.Linear(2 * 16, 1)
        self.fc = torch.nn.Linear(2 * 16, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # x = torch.sigmoid(self.fc(x)).squeeze(1)
        x = self.fc(x)

        return x
