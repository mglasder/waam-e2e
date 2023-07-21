from torch import nn


class SimpleConv(nn.Module):
    def __init__(self, in_channels, out_channels=100, kernel_size=224, stride=1, padding=0):
        super().__init__()

        self.bn = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, kernel_size, kernel_size, stride, padding)
        # self.conv2 = nn.Conv1d(in_channels, kernel_size, kernel_size, stride, padding)
        self.conv_out = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.dropout = nn.Dropout(0.5)

    # def to(self, device):
    #     # TODO: implement
    #     pass

    def forward(self, x):
        x = self.bn(x)
        x = self.dropout(x)
        x = self.conv1(x)
        x = x.view(-1, 1, 224)
        # x = self.bn(x)
        # x = self.conv2(x)
        # x = x.view(-1, 1, 224)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        return x
