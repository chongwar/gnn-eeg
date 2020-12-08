import torch
from torch_geometric.nn import SAGEConv, Set2Set
import torch.nn.functional as F


class Set2SetNet(torch.nn.Module):
    """
    The global pooling operator based on iterative content-based attention
    from the "order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>
    """

    def __init__(self, num_features):
        super(Set2SetNet, self).__init__()
        self.conv1 = SAGEConv(num_features, 8)
        self.conv2 = SAGEConv(8, 16)

        self.set2set = Set2Set(16, processing_steps=4)

        # self.fc = torch.nn.Linear(2 * 16, 1)
        self.fc = torch.nn.Linear(2 * 16, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = self.set2set(x, batch)
        # x = torch.sigmoid(self.fc(x)).squeeze(1)
        x = self.fc(x)

        return x
