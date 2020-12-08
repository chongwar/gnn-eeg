import torch
from torch_geometric.nn import TAGConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F


class TAGWithJK(torch.nn.Module):
    """
    The topology adaptive graph convolutional networks operator from the
    "Topology Adaptive Graph Convolutional Networks"
    <https://arxiv.org/abs/1710.10370>
    """

    def __init__(self, num_features, num_layers):
        super(TAGWithJK, self).__init__()
        self.conv1 = TAGConv(num_features, 8)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(TAGConv(8, 8))
        self.jump = JumpingKnowledge('cat')

        self.fc = torch.nn.Linear(2 * num_layers * 8, 2)

    def forward(self, data):
        x, edge_index, batch, edge_weight = data.x, data.edge_index, \
                                            data.batch, data.edge_attr
        print(x.shape)
        print(edge_index.shape)
        print(edge_weight.shape)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
            xs += [x]

        x = self.jump(xs)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # x = torch.sigmoid(self.fc(x)).squeeze(1)
        x = self.fc(x)

        return x
