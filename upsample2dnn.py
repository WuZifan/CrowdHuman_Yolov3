import torch.nn as nn
import torch
from torch.autograd import Variable
import cv2
from models.my_yolo import MyYolov3

class MyUpSample(nn.Module):
    def __init__(self):
        super(MyUpSample, self).__init__()
        self.weight = [[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]]

        self.model = nn.ConvTranspose2d(in_channels=2,out_channels=1,kernel_size=2,stride=2,)


        # 返回每一层
        for i in self.modules():
            if isinstance(i,nn.ConvTranspose2d):
                print(type(i),i.weight,i.bias)

        print('======')

        # 仅返回每一组权重（比module更深一层）
        for i in self.parameters():
            print(i)


    def forward(self,x):

        return self.model(x)

def loadcv2dnnNetONNX(onnx_path):
    net = cv2.dnn.readNetFromONNX(onnx_path)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

if __name__ == '__main__':
    test_input = Variable(torch.randn([1,3,416,416]))
    # model = MyUpSample()
    # print(model(test_input).shape)

    model = MyYolov3()

    outputs = model(test_input)

    print(outputs.shape)

    onnx_path = "./weights/yolov3_transposeconv2d.onnx"

    torch.onnx._export(model, test_input, onnx_path, export_params=True,input_names=['input'],output_names = ['output'])


    loadcv2dnnNetONNX(onnx_path)
