import torch
import torch.nn.functional as F
from torch import nn


class LSTM(nn.Module):
    def __init__(
        self, p=0.5, n_input_features=120, n_hidden=120, n_output_features=120, n_outputs=4, n_layers=3, batch_size=64
    ):
        super().__init__()
        self.p = p

        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.fc = nn.Linear(in_features=n_input_features, out_features=n_output_features, bias=False)
        self.ln1 = nn.LayerNorm((1, n_input_features))

        self.lstm = nn.LSTM(
            input_size=n_output_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            bidirectional=False,
            batch_first=True,
            dropout=p,
            bias=False,
        )

        self.ln2 = nn.LayerNorm((1, n_hidden))

        self.fc2 = nn.Linear(in_features=n_hidden, out_features=n_output_features, bias=False)

        self.ln3 = nn.LayerNorm((1, n_output_features))

        self.combine = nn.Linear(in_features=n_output_features + 1, out_features=n_output_features)

        self.out = nn.Linear(in_features=n_output_features, out_features=n_output_features)

    def forward(self, x):
        m = x[:, :, -1].unsqueeze(1)
        r0 = x[:, :, :-1]

        x = self.fc(r0)
        x = self.ln1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training) + r0
        r1 = x

        x, _ = self.lstm(x)
        x = self.ln2(x)
        x = self.fc2(x) + r1
        x = F.relu(self.ln3(x))

        x_ = torch.concatenate((x, m), dim=2)
        x = F.tanh(self.combine(x_))
        x = self.out(x)
        return x
