import torch.nn as nn
import torch

from utils.utils import build_targets, to_cpu, non_max_suppression
import numpy as np




def sigmoid_np(x):
    s = 1 / (1 + np.exp(-x))
    return s

class YOLO_NP:
    def __init__(self,anchors, num_classes, img_dim=416):
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        # 边框回归
        self.mse_loss = nn.MSELoss()
        # 多标签回归
        self.bce_loss = nn.BCELoss()

        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        # 图像尺寸
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=False):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)

        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
        print(self.anchor_w,self.anchor_h)

        return

    def compute_grid_offset2(self,grid_size,cuda=False):
        self.grid_size = grid_size
        g = self.grid_size

        self.stride = self.img_dim / self.grid_size
        self.grid_x = np.arange(g).reshape(1,g).repeat(g, 0).reshape([1, 1, g, g])
        self.grid_y = np.arange(g).reshape(1,g).repeat(g, 0).transpose(1,0).reshape([1, 1, g, g])
        self.scaled_anchors = np.array([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].reshape((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].reshape((1, self.num_anchors, 1, 1))



    def forward(self,x):
        if isinstance(x,torch.Tensor):
            x = x.detach().numpy()

        num_samples = x.shape[0]
        grid_size = x.shape[2]

        prediction = (
            x.reshape(num_samples, self.num_anchors, self.num_classes + 5,
                      grid_size, grid_size)
                .transpose(0, 1, 3, 4, 2)
        )

        x = sigmoid_np(prediction[..., 0])  # Center x
        y = sigmoid_np(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = sigmoid_np(prediction[..., 4])  # Conf
        pred_cls = sigmoid_np(prediction[..., 5:])  # Cls pred.

        if grid_size != self.grid_size:
            self.compute_grid_offset2(grid_size)

        # Add offset and scale with anchors
        pred_boxes = np.zeros_like(prediction[..., :4])
        print(pred_boxes.shape,self.grid_x.shape)
        pred_boxes[..., 0] = x + self.grid_x
        pred_boxes[..., 1] = y + self.grid_y
        pred_boxes[..., 2] = np.exp(w) * self.anchor_w
        pred_boxes[..., 3] = np.exp(h) * self.anchor_h

        output = np.concatenate(
            (
                pred_boxes.reshape(num_samples, -1, 4) * self.stride,
                pred_conf.reshape(num_samples, -1, 1),
                pred_cls.reshape(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        return output

    def __call__(self, x):
        return self.forward(x)

if __name__ == '__main__':
    # ynp = YOLO_NP([(10,10),(20,20),(30,30)],2,416)
    #
    # X = 'AAA'
    # # print(ynp(X))
    #
    # ynp.compute_grid_offsets(13)
    # print('=====')
    # ynp.compute_grid_offset2(13)
    a = np.arange(5)
    b = torch.arange(5).float()
    print(a,b)
    print(sigmoid_np(a))
    print(torch.sigmoid(b))

