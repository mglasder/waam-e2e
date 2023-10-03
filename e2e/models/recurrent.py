import torch.nn.functional as F
from torch import nn

from e2e.models.layers import MultiGaussianSmoothing, GaussianSmoothing


class LSTM(nn.Module):
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

        self.bn1 = nn.BatchNorm1d(1)
        self.fc = nn.Linear(in_features=n_input_features, out_features=n_output_features)

        self.bn2 = nn.BatchNorm1d(1)

        self.lstm = nn.LSTM(
            input_size=n_output_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            bidirectional=False,
            batch_first=True,
            dropout=p,
        )

        self.fc2 = nn.Linear(in_features=n_hidden, out_features=n_output_features)

        self.out = nn.Linear(in_features=n_output_features, out_features=2)

    def forward(self, x):
        r0 = x

        x = self.bn1(x)
        x = self.fc(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training) + r0
        r1 = x

        x = self.bn2(x)
        x, _ = self.lstm(x)
        x = F.relu(self.fc2(x)) + r1

        x = self.out(x)
        return x
