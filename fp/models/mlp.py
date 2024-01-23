from torch import nn


class Layer(nn.Module):
    def __init__(self, n_input_features, n_output_features, p=0.5):
        super().__init__()
        self.p = p

        self.layer = nn.Sequential(
            nn.Dropout(p=self.p),
            nn.Linear(in_features=n_input_features, out_features=n_output_features, bias=False),
            nn.LayerNorm(n_output_features, elementwise_affine=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layer(x)


class ResMLPpoints(nn.Module):
    def __init__(
        self,
        p=0.5,
        n_input_features=224,
        n_output_features=224,
    ):
        super().__init__()
        self.p = p
        self.n_input_features = n_input_features

        self.in_layer = Layer(n_input_features=n_input_features * 2, n_output_features=n_input_features * 2, p=self.p)

        self.h1 = Layer(n_input_features=n_input_features * 2, n_output_features=n_input_features, p=self.p)
        self.h2 = Layer(n_input_features=n_input_features, n_output_features=n_input_features, p=self.p)
        self.h3 = Layer(n_input_features=n_input_features, n_output_features=n_input_features, p=self.p)

        self.idx_out = nn.Linear(in_features=n_input_features, out_features=2, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # init bias from out layer
        if type(m) == nn.LayerNorm:
            if m.bias is not None:
                m.bias.data.fill_(0.1)
        if type(m) == nn.Linear:
            if m.bias is not None:
                m.bias.data.fill_(112)

    def forward(self, x):
        # r1 = x.reshape(-1, 2 * self.n_input_features)
        x = self.in_layer(x.reshape(-1, 2 * self.n_input_features))

        x = self.h1(x)
        r2 = x
        x = self.h2(x) + r2
        r3 = x
        x = self.h3(x) + r3

        fp = self.idx_out(x)

        return fp
