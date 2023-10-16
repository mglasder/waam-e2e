import torch
import torch.nn.functional as F
from torch import nn


class Layer(nn.Module):
    def __init__(self, n_input_features, n_output_features, p=0.5):
        super().__init__()
        self.p = p

        self.layer = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Linear(in_features=n_input_features, out_features=n_output_features),
            nn.ReLU(),
            nn.Dropout(p=self.p),
        )

    def forward(self, x):
        return self.layer(x)


class ResMLP(nn.Module):
    def __init__(
        self,
        p=0.5,
        n_input_features=120,
        n_hidden=120,
        n_output_features=120,
        n_layers=3,
    ):
        super().__init__()
        self.p = p

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_output_features = n_output_features

        self.in_layer = Layer(n_input_features=n_input_features, n_output_features=n_hidden, p=self.p)

        self.h1 = Layer(n_input_features=n_hidden, n_output_features=n_hidden, p=self.p)
        self.h2 = Layer(n_input_features=n_hidden, n_output_features=n_hidden, p=self.p)
        self.h3 = Layer(n_input_features=n_hidden, n_output_features=n_hidden, p=self.p)

        self.out_layer = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Linear(in_features=n_hidden, out_features=n_output_features),
        )

    def forward(self, x):
        x = self.in_layer(x)
        r1 = x
        x = self.h1(x) + r1
        r2 = x
        x = self.h2(x) + r2
        r3 = x
        x = self.h3(x) + r3

        x = self.out_layer(x)

        return x


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


class SoftResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, p):
        super(SoftResidualBlock, self).__init__()

        self.main_path = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_dim, input_dim),
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))  # Initialize to 0.5 for equal mixing

    def forward(self, x):
        return F.sigmoid(self.alpha) * x + (1 - F.sigmoid(self.alpha)) * self.main_path(x)


class SoftResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, p):
        super(SoftResNet, self).__init__()

        self.blocks = nn.Sequential(*[SoftResidualBlock(input_dim, hidden_dim, p) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x)
