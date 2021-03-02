import torch
from torch_geometric.nn import TAGConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TAGLSTM(torch.nn.Module):
    """
    The topology adaptive graph convolutional networks operator from the
    "Topology Adaptive Graph Convolutional Networks"
    <https://arxiv.org/abs/1710.10370>
    """

    def __init__(self, time_samples=128, channels=64, seq_len=8, input_size=4,
                 hidden_size=4, num_layers=1):
        super(TAGLSTM, self).__init__()
        self.T = time_samples
        self.C = channels
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.num_features = self.T // self.seq_len
        self.gcn_output_size = self.input_size

        self.gcn = TAGConv(self.num_features, self.gcn_output_size)
        self.gcns = torch.nn.ModuleList([self.gcn for i in range(self.seq_len)])

        self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size,
                                  self.num_layers, batch_first=True)

        self.fc = torch.nn.Linear(self.hidden_size, 2)

        num_edges = self.C * self.C - self.C
        self.weights = []
        for i in range(self.seq_len):
            self.edge_weight = torch.nn.Parameter(torch.FloatTensor(num_edges, 1),
                                                  requires_grad=True)
            self.edge_weight.data.fill_(1)
            self.weights.append(self.edge_weight)

    def forward(self, data):
        # B: batch_size;    C: channels;    T: number of features.
        # x: (B * C, T)
        # edge_index: (2, C * (C - 1) * B)
        # batch: (B * C, )
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batch_size = x.shape[0] // self.C
        # x: (B * C, T) => tuple((B * C, self.num_features), ..., (B * C, self.num_features))
        x = list(torch.split(x, self.num_features, dim=-1))

        # edge_weight: (C * (C - 1), 1)  => (C * (C - 1) * B, 1)
        for i in range(len(self.weights)):
            edge_weight = self.weights[i].to(device)
            _edge_weight = edge_weight
            for j in range(batch_size - 1):
                edge_weight = torch.cat((edge_weight, _edge_weight), dim=0)

            x[i] = F.relu(self.gcns[i](x[i], edge_index, edge_weight))
            x[i] = gmp(x[i], batch)

        # x: (B, self.seq_len, self.gcn_output_size)
        x = torch.stack(x, dim=1)

        lstm_out, _ = self.lstm(x, None)
        x = lstm_out[:, -1, :]
        x = self.fc(x)
        return x


if __name__ == '__main__':
    tag_learn = TAGLSTM(5)
    print(tag_learn)
