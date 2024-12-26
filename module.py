class MLFR(nn.Module):

    def __init__(self, channel, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv3_3 = nn.Conv2d(channel // 4, channel // 4, kernel_size=1)
        self.conv4_3 = nn.Conv2d(channel // 2, channel // 2, kernel_size=1)
        self.conv5_3 = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=channel + channel // 2 + channel // 4,
                               out_channels=channel + channel // 2 + channel // 4,
                               kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2)
        self.a1 = nn.Parameter(torch.tensor([0.5]))
        self.a2 = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        x1 = self.a1 * self.upsample(self.conv5_3(x[0]))
        x2 = self.a2 * self.upsample(torch.cat([self.conv4_3(x[1]), x1], self.d))
        x_ = self.conv3(torch.cat([self.conv3_3(x[2]), x2], self.d))
        return x_


class MRFR(nn.Module):
    def __init__(self, channel=128, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv11 = nn.Conv2d(channel, channel // 4, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=1)

    def forward(self, x):
        x1_1 = self.conv11(x)
        x1_2 = self.conv33(x1_1)
        x1_3 = self.conv33(x1_2)
        x1_4 = self.conv33(x1_3)
        x2 = torch.cat([x1_1, x1_2, x1_3, x1_4], self.d)
        x_ = x + x2
        return x_