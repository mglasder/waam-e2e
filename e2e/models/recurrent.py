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

        # self.lstm = nn.LSTM(
        #     input_size=n_output_features,
        #     hidden_size=n_hidden,
        #     num_layers=n_layers,
        #     bidirectional=False,
        #     batch_first=True,
        #     dropout=p,
        #     bias=False,
        # )

        self.main_fc = nn.Linear(in_features=n_output_features, out_features=n_hidden, bias=False)

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

        # x, _ = self.lstm(x)
        x = self.main_fc(x)
        x = self.ln2(x)
        x = self.fc2(x) + r1
        x = F.relu(self.ln3(x))

        x_ = torch.concatenate((x, m), dim=2)
        x = F.tanh(self.combine(x_))
        x = self.out(x)
        return x


class ShapePointsModel(nn.Module):
    def __init__(
        self,
        p=0.5,
        n_input_features=100,
        n_output_features=4,
        device="cpu",
    ):
        super().__init__()
        self.p = p

        # self.ln1 = nn.LayerNorm(2 * n_input_features)
        self.fc1 = nn.Linear(2 * n_input_features, 4 * n_input_features, bias=True)

        self.fc2 = nn.Linear(4 * n_input_features, 2 * n_input_features, bias=True)
        # self.ln2 = nn.LayerNorm(2 * n_input_features)

        self.fc_out = nn.Linear(2 * n_input_features, n_output_features, bias=True)

        self.t = torch.linspace(0.0, 1.0, n_input_features, device=device)
        self.t2 = self.t * self.t
        self.t3 = self.t2 * self.t
        self.mt = 1 - self.t
        self.mt2 = self.mt * self.mt
        self.mt3 = self.mt2 * self.mt

        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.uniform_(m.bias)
                # m.bias.data.fill_(1.5)

    def forward(self, x):

        x_in = x.clone()

        batch_sz = x.shape[0]

        p0 = x[:, :, 0].clone()
        p3 = x[:, :, -1].clone()

        x = x.reshape(batch_sz, -1)
        r = x.clone()

        # x = self.ln1(x)
        x = self.fc1(x)
        x = F.tanh(F.dropout(x, p=self.p, training=self.training))  # + r

        # x = self.ln2(self.fc2(x))
        x = self.fc2(x)
        x = F.tanh(F.dropout(x, p=self.p, training=self.training))

        p12 = self.fc_out(x + r).reshape(batch_sz, 2, 2)

        pp = torch.stack([p0, p12[:, :, 0], p12[:, :, 1], p3], dim=2)

        px = pp[:, 1, :]
        pz = pp[:, 0, :]

        x_out = self._bezier3_torch(px)
        z_out = self._bezier3_torch(pz)

        stack = torch.stack([z_out, x_out], dim=1)
        return stack

    def _bezier3_torch(self, w: torch.Tensor) -> torch.Tensor:
        r0 = w[:, 0].unsqueeze(1) * self.mt3
        r1 = 3 * w[:, 1].unsqueeze(1) * self.mt2 * self.t
        r2 = 3 * w[:, 2].unsqueeze(1) * self.mt * self.t2
        r3 = w[:, 3].unsqueeze(1) * self.t3
        return r0 + r1 + r2 + r3
