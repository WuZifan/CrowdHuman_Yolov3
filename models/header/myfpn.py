import torch.nn as nn
import torch
import torch.nn.functional as F


from torchvision.models import resnet18

class ResBlock(nn.Module):
    '''

        resBlock由：
            1、shortcut分支
                1.1 非downsample，直接相加
                1.2 downsample，降采样后再加
            2、残差分支组成：
                3个conv2d+bn2d，最后跟一个relu

    '''
    expansion=4
    def __init__(self, in_channel, planes, stride=2, downsample=None):
        super(ResBlock, self).__init__()


        self.downsample = downsample

        # 1*1 kernel_size
        self.conv_bn1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=planes,
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

        # 3*3 kernel_size
        self.conv_bn2 = nn.Sequential(
            nn.Conv2d(in_channels=planes,
                      out_channels=planes,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

        # 1*1
        self.conv_bn3 = nn.Sequential(
            nn.Conv2d(in_channels=planes,
                      out_channels=self.expansion*planes,
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(self.expansion*planes),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        residual = x

        x = self.conv_bn1(x)

        x = self.conv_bn2(x)

        x = self.conv_bn3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        print('residual {} x {}'.format(residual.shape,x.shape))
        x += residual

        x = self.relu(x)

        return x


class FPN_RES18(nn.Module):
    cfg = [(2, 64, 64),
           (2, 256, 128),
           (2, 512, 256),
           (2, 1024, 512)]
    def __init__(self):
        super(FPN_RES18, self).__init__()

        '''
            这一段是Resnet降采样
        '''
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.layer1 = self._make_layer(num_blocks=self.cfg[0][0],
                                       inchannel=self.cfg[0][1],
                                       planes=self.cfg[0][2],
                                       layer_cnt=1)
        self.layer2 = self._make_layer(num_blocks=self.cfg[1][0],
                                       inchannel=self.cfg[1][1],
                                       planes=self.cfg[1][2],
                                       layer_cnt=2)
        self.layer3 = self._make_layer(num_blocks=self.cfg[2][0],
                                       inchannel=self.cfg[2][1],
                                        planes=self.cfg[2][2],
                                       layer_cnt=3)
        self.layer4 = self._make_layer(num_blocks=self.cfg[3][0],
                                       inchannel=self.cfg[3][1],
                                       planes=self.cfg[3][2],
                                       layer_cnt=4)

        '''
            处理最顶层
        '''
        self.toplayer = nn.Conv2d(in_channels=2048,out_channels=256,kernel_size=1,stride=1,padding=0)

        '''
            平滑层
        '''
        self.smooth1 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.smooth2 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.smooth3 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)

        '''
            合并前预处理层
        '''
        self.latlayer1 = nn.Conv2d(in_channels=1024,out_channels=256,kernel_size=1,stride=1,padding=0)
        self.latlayer2 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1,stride=1,padding=0)
        self.latlayer3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1,stride=1,padding=0)


    def forward(self,x):

        # 降采样
        c1 = self.first_layer(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        print('c1 {},c2 {},c3 {},c4 {},c5 {}'.format(c1.shape,c2.shape,c3.shape,c4.shape,c5.shape))

        # 上采样+合并
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5,self.latlayer1(c4))
        print('p5 {},p4 {}'.format(p5.shape,p4.shape))

        p3 = self._upsample_add(p4,self.latlayer2(c3))
        p2 = self._upsample_add(p3,self.latlayer3(c2))

        # smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return p2,p3,p4,p5





    def _upsample_add(self,x,y):
        _,_,H,W =y.size()
        return F.upsample(x,size=(H,W),mode='bilinear')+y


    def _make_layer(self,num_blocks,inchannel,planes,layer_cnt=-1):
        temp_layer = []
        for i in range(num_blocks):
            downsample=None
            temp_inchannel = planes*ResBlock.expansion
            stride= 1
            if i==0:
                if layer_cnt!=1:
                    stride = 2
                downsample=self._get_downsample(inchannel,ResBlock.expansion*planes,stride)
                temp_inchannel = inchannel
            temp_layer.append(ResBlock(temp_inchannel,planes,
                                       stride=stride,
                                       downsample=downsample))
        return nn.Sequential(*temp_layer)

    def _get_downsample(self,in_channel,out_channel,stride=2):
        downsample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        return downsample


class ResNet18(nn.Module):
    cfg = [(2,64,64),
           (2,256,128),
           (2,512,256),
           (2,1024,512)]
    def __init__(self,num_class=1000):
        super(ResNet18, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.residual_layers = self._make_layer()

        self.avg = nn.AvgPool2d(kernel_size=7)

        self.fc = nn.Linear(512*ResBlock.expansion,num_class)

    def forward(self,x):
        x = self.first_layer(x)
        print('after first layer',x.shape)

        x = self.residual_layers(x)

        x = self.avg(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

    def _make_layer(self):
        temp_layer = []
        for j,(num_blocks,inchannel,planes) in enumerate(self.cfg):
            for i in range(num_blocks):
                downsample=None
                temp_inchannel = planes*ResBlock.expansion
                stride= 1
                if i==0:
                    if j!=0:
                        stride = 2
                    downsample = self._get_downsample(inchannel,
                                                      ResBlock.expansion * planes,
                                                      stride=stride)
                    temp_inchannel = inchannel
                temp_layer.append(ResBlock(temp_inchannel,planes,
                                           stride=stride,
                                           downsample=downsample))

        return nn.Sequential(*temp_layer)

    def _get_downsample(self,in_channel,out_channel,stride=2):
        downsample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        return downsample

if __name__ == '__main__':
    test_input = torch.rand([2,3,224,224])

    # model = ResNet18()
    model = FPN_RES18()
    print(model)

    o1,o2,o3,o4 = model(test_input)
    print('o1 {},o2 {},o3 {},o4 {}'.format(o1.shape,o2.shape,o3.shape,o4.shape))




