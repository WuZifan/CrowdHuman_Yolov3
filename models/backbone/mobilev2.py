import torch.nn as nn
import torch
import torch.nn.functional as F



class Block(nn.Module):
    def __init__(self,in_channel,out_channel,expansion,stride):
        '''

        :param in_channel:
        :param out_channel:
        :param expansion:
        :param stride:  1:need shortcut ; 2: dont need shortcut
        '''
        super(Block, self).__init__()

        self.stride = stride

        expansion_channel = expansion*in_channel

        self.model = nn.Sequential(
            self._conv_bn(in_channel,expansion_channel,stride=1,kernel_size=1),
            self._conv_dw(expansion_channel,expansion_channel,stride),
            self._conv_bn(expansion_channel,out_channel,stride=1,kernel_size=1)
        )

        self.shortcut = None

        if stride==1:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels=in_channel,
                          out_channels=out_channel,
                          kernel_size=1,
                          stride=1,padding=0,bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self,x):
        new_x = self.model(x)

        if self.shortcut is not None:
            new_x+= self.shortcut(x)

        # print('new_x {},y {}'.format(new_x.shape,new_x.shape))

        return new_x

    def _conv_bn(self,in_channel,out_channel,stride,kernel_size = 1):
        ret = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=0 if kernel_size==1 else 1,
                      bias=False
                      ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )
        return ret

    def _conv_dw(self,in_channel,out_channle,stride):
        ret = nn.Sequential(
            nn.Conv2d(in_channels= in_channel,
                      out_channels= in_channel,
                      kernel_size=3,
                      stride = stride,
                      padding=1,
                      bias=False
                      ),
            nn.BatchNorm2d(num_features=in_channel),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channle,
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(out_channle),
            nn.ReLU6(inplace=True)
        )
        return ret


class Mobilev2(nn.Module):
    #(expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 2),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]


    def __init__(self,num_classes=10):
        super(Mobilev2, self).__init__()

        all_inverted_residuals = self._make_layer()

        self.model=nn.Sequential(
            self._conv_bn(3,32,2,3,1),

            all_inverted_residuals,

            self._conv_bn(320,1280,1,1,0),
        )
        # self.model1 = self._conv_bn(3,32,1,3,1)
        # self.model2 = all_inverted_residuals
        # self.model3 = self._conv_bn(320,1280,1,1,0)

        self.header = nn.Linear(1280,num_classes)

    def _make_layer(self):
        net = nn.Sequential()
        in_channel = 32
        for i,(expansion,out_channel,num_block,stride) in enumerate(self.cfg):
            strides = [stride] + [1] * (num_block - 1)
            for j ,temp_stride in enumerate(strides):
                # print(i,j,in_channel,out_channel)
                temp_net = Block(in_channel,out_channel,expansion,temp_stride)
                in_channel=out_channel
                net.add_module(name='Inverted Residuals {}_{}'.format(i,j),module=temp_net)

        return net

    def forward(self,x):
        x = self.model(x)

        print('xxxx',x.shape)

        x = F.avg_pool2d(x, 7)
        print('xxx',x.shape)

        x = x.view(x.size(0), -1)

        x = self.header(x)

        return x

    def _conv_bn(self,in_channel,out_channel,stride,kernel_size,padding):
        net = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      stride=stride,
                      kernel_size=kernel_size,
                      padding=padding,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        return net

if __name__ == '__main__':
    batch_size = 8
    test_input = torch.rand([batch_size,3,224,224])
    mobilev2 = Mobilev2()
    output = mobilev2(test_input)

    print('output shape',output.shape)


