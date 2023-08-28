import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        p=0.5,
        n_features=120,
    ):
        super().__init__()
        self.p = p

        self.fc_in = nn.Linear(in_features=n_features, out_features=n_features)
        self.hidden1 = nn.Linear(in_features=n_features, out_features=n_features)
        self.hidden2 = nn.Linear(in_features=n_features, out_features=n_features)

        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(1)
        self.bn3 = nn.BatchNorm1d(1)

    # def to(self, device):
    #     # TODO: implement
    #     pass

    def forward(self, x):
        r0 = x

        x = self.bn1(x)
        x = self.fc_in(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training) + r0
        r1 = x

        x = self.bn2(x)
        x = self.hidden1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training) + r1
        r2 = x

        x = self.bn3(x)
        x = self.hidden2(x)

        return x + r2


class RNN(nn.Module):
    def __init__(
        self,
        p=0.5,
        n_features=120,
        n_hidden=120,
        n_layers=1,
    ):
        super().__init__()
        self.p = p

        self.rnn = nn.RNN(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            bidirectional=False,
            batch_first=True,
            dropout=p,
            nonlinearity="relu",
        )

        self.fc = nn.Linear(in_features=n_hidden, out_features=n_features)

        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(1)

    def forward(self, x):
        r0 = x

        self.bn1(x)
        x = self.fc(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training) + r0

        r1 = x

        h0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden).requires_grad_().to(x.device)
        x = self.bn1(x)
        x, hn = self.rnn(x, h0)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.p, training=self.training) + r0
        # r1 = x

        # x = self.bn2(x)
        # x = self.fc_out(x)

        return x + r1
