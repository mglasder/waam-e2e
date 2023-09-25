import torch.nn.functional as F
from torch import nn

from e2e.models.layers import MultiGaussianSmoothing, GaussianSmoothing


class Transformer(nn.Module):
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

        self.transformer = nn.Transformer(
            d_model=n_output_features,
            nhead=n_output_features // 3,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=n_hidden,
            batch_first=True,
        )

        self.fc2 = nn.Linear(in_features=n_hidden, out_features=n_output_features)

        self.out = nn.Linear(in_features=n_output_features, out_features=n_output_features)

        self.multi_smooth = MultiGaussianSmoothing(
            output_length=n_output_features,
            window_size=15,
            sigma_inits=[0.3, 1.2, 3.0],
        )

        self.postsmooth = GaussianSmoothing(window_size=15, sigma_init=1.5).requires_grad_(False)

    def forward(self, x):
        r0 = x

        self.bn1(x)
        x = self.fc(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training) + r0
        r1 = x

        x = self.bn2(x)

        x = self.transformer(x, x)

        x = F.relu(self.fc2(x)) + r1

        x = self.out(x)
        x = self.multi_smooth(x)
        x = self.postsmooth(x)
        return x
