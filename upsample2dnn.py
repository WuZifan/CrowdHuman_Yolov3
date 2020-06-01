import torch.nn as nn
import torch
from torch.autograd import Variable
import cv2
from models.my_yolo import MyYolov3
import numpy as np

class MyUpSample(nn.Module):
    def __init__(self):
        super(MyUpSample, self).__init__()
        self.model = nn.Sequential(nn.ConvTranspose2d(in_channels=2,
                                        out_channels=2,
                                        kernel_size=2,stride=2)
                                   )

    def forward(self,x):
        return self.model(x)

if __name__ == '__main__':
    test_input = Variable(torch.randn([1,3,416,416]))
    # test_input = Variable(torch.randn([1,2,24,24]))
    # model = MyUpSample()
    # print(model(test_input).shape)

    model = MyYolov3()
    # model = MyUpSample()
    o1,o2,o3 = model(test_input)

    print(o1.shape,o2.shape,o3.shape)

    onnx_path = "./weights/yolov3_transposeconv2d.onnx"

    # torch.onnx.export(model, test_input, onnx_path, export_params=True,verbose=True)
    # loadcv2dnnNetONNX(onnx_path)

    img_path = './data/pics/warehouse1.jpg'
    testcv2nnNet(img_path,onnx_path)