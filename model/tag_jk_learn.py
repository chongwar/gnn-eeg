import torch
from torch_geometric.nn import TAGConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F


class TAGWithJKLearn(torch.nn.Module):
    """
    The topology adaptive graph convolutional networks operator from the
    "Topology Adaptive Graph Convolutional Networks"
    <https://arxiv.org/abs/1710.10370>
    """

    def __init__(self, num_features, num_layers, channels=64):
        super(TAGWithJKLearn, self).__init__()
        self.conv1 = TAGConv(num_features, 8)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(TAGConv(8, 8))
        self.jump = JumpingKnowledge('cat')

        self.fc = torch.nn.Linear(2 * num_layers * 8, 2)

        num_edges = channels * channels - channels
        self.edge_weight = torch.nn.Parameter(torch.FloatTensor(num_edges, 1),
                                              requires_grad=True)
        self.edge_weight.data.fill_(1)

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


if __name__ == '__main__':
    tag_jk_learn = TAGWithJKLearn(5, 4)
    print(tag_jk_learn)
