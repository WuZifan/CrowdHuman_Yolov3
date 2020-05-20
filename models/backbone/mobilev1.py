import torch.nn as nn
import torch


class MobileV1(nn.Module):
    def __init__(self):
        super(MobileV1, self).__init__()
        self.model = nn.Sequential(
            self._conv_bn(3,32,2),
            self._conv_dw(32,64,1),
            self._conv_dw(64,128,2),
            self._conv_dw(128,128,1),
            self._conv_dw(128,256,2),
            self._conv_dw(256,256,1),
            self._conv_dw(256,512,2),
            self._conv_dw(512,512,1),
            self._conv_dw(512, 512, 1),
            self._conv_dw(512, 512, 1),
            self._conv_dw(512, 512, 1),
            self._conv_dw(512, 512, 1),
            self._conv_dw(512,1024,2),
            self._conv_dw(1024,1024,1),
            nn.AvgPool2d(kernel_size=7),
        )

        self.fc = nn.Linear(in_features=1024,
                            out_features=1000)

    def forward(self,x):
        x = self.model(x)

        x = x.view(-1, 1024)

        x = self.fc(x)

        return x

    def _conv_bn(self,in_channel,out_channel,stride):

        net = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False
                      ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True)
        )
        return net

    def _conv_dw(self,in_channel,out_channel,stride):
        net = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=in_channel,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      groups=in_channel,
                      bias=False),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=False)
        )
        return net



if __name__ == '__main__':
    batch_size=8
    test_input = torch.rand([batch_size,3,224,224])
    print(test_input.shape)

    mobilev1 = MobileV1()

    print(mobilev1(test_input).shape)


