from torch import nn


class SimpleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=100,
        kernel_size=224,
        p=0.2,
        stride=1,
        padding=0,
    ):
        super().__init__()
        self.p = p
        self.k = kernel_size

        self.bn = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, kernel_size, kernel_size, stride, padding)
        # self.conv2 = nn.Conv1d(in_channels, kernel_size, kernel_size, stride, padding)
        # self.conv2 = nn.Conv1d(in_channels, kernel_size, kernel_size, stride, padding)
        # self.conv_out = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        # self.dropout = nn.Dropout(self.p)
        # self.dropout = nn.Dropout1d(self.p)
        # self.fc = nn.Linear(in_features=out_channels, out_features=out_channels)
        self.fc = nn.Linear(in_features=kernel_size, out_features=out_channels)
        self.act = nn.ReLU()

    # def to(self, device):
    #     # TODO: implement
    #     pass

    def forward(self, x):
        r = x
        x = self.bn(x)
        x = self.fc(x)
        x = self.act(x)

        x = self.bn(x)
        x = self.conv1(x)

        # x = self.dropout(x)
        # x = x.view(-1, 1, self.k)
        # x = self.bn(x)
        # x = self.conv2(x)
        # x = self.bn(x)
        # x = self.conv2(x)
        # x = x.view(-1, 1, self.k)

        # x = self.bn(x)
        # x = self.conv_out(x)
        x = x.view(-1, 1, self.k)

        return x + r
