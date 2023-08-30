from torch import nn
import torch.nn.functional as F
import torch

from e2e.models.simpleconv import SmoothingLayer


class RNN(nn.Module):
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

        self.rnn = nn.RNN(
            input_size=n_output_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            bidirectional=False,
            batch_first=True,
            dropout=p,
            nonlinearity="relu",
        )

        self.fc = nn.Linear(in_features=n_input_features, out_features=n_output_features)

        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(1)

        # self.alpha = nn.Parameter(torch.tensor(0.5))
        # self.beta = nn.Parameter(torch.tensor(0.5))

        # self.alpha = nn.Linear(in_features=n_input_features, out_features=n_output_features, bias=False)
        # self.beta = nn.Linear(in_features=n_input_features, out_features=n_output_features, bias=False)

        self.smooth = SmoothingLayer(max_window_size=30)

        self.out = nn.Linear(in_features=n_hidden, out_features=n_output_features)

    def forward(self, x):
        r0 = x

        self.bn1(x)
        x = self.fc(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)  # + self.alpha(r0)

        # r1 = x

        h0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden).requires_grad_().to(x.device)
        x = self.bn2(x)
        x, _ = self.rnn(x, h0)

        # x = F.relu(x)
        # x = F.dropout(x, p=self.p, training=self.training) + r0
        # r1 = x

        # x = self.bn2(x)
        # x = self.fc_out(x)

        return self.out(x + r0)
