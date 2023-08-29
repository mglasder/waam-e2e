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
