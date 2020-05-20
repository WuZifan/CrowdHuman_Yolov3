from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    # 读取cfg文件第一层[net]中的设置内容
    hyperparams = module_defs.pop(0)
    print('hyperparams {}'.format(hyperparams))
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    cnt = 0
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                "conv_{}".format(module_i),
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module("batch_norm_{}".format(module_i), nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_{}".format(module_i), nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            # print('do maxpool')
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module("_debug_padding_{}".format(module_i), nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module("maxpool_{}".format(module_i), maxpool)

        elif module_def["type"] == "upsample":
            # print('do unsample')

            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module("upsample_{}".format(module_i), upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module("route_{}".format(module_i), EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module("shortcut_{}".format(module_i), EmptyLayer())

        elif module_def["type"] == "yolo":
            # 选择使用哪几个anchors，拿到id
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            # 根据id，拿到anchors的宽高
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            # 分类类别数目
            num_classes = int(module_def["classes"])
            # 输入图片大小（图片被pad到square了）
            img_size = int(hyperparams["height"])
            # Define detection layer
            print('yolo layer anchors {},num_classes {},img_size {}'.format(anchors,num_classes,img_size))
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module("yolo_{}".format(module_i), yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)
        cnt+=1

    # print(output_filters)

    # for module_def,module in zip(module_defs,module_list):
    #     if module_def['type']=='yolo':
    #         print('yolo',module[0])


    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x




class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
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

    def compute_grid_offsets(self, grid_size, cuda=True):
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

    def forward(self, x, targets=None, img_dim=None):

        # print('hahaha',x.shape)

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        # 输入图像大小
        self.img_dim = img_dim
        # N,C,H,W
        # 几个样本
        num_samples = x.size(0)
        # 目前样本的尺寸
        grid_size = x.size(2)

        # print('raw x shape {}'.format(x.shape))
        # print('x view shape {}'.format((num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)))

        '''
            reshape一下，
            [num_samples,num_anchors,grid_size,grid_size,num_class+5]
        '''
        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        '''
            这个...表示取最里面那个num_class+5这个维度的
            x，y是bbox相对于当前cell的偏移量
            w,h是bbox的w,h相对于anchors（在当前feature_map下）的log值
        '''
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # print('heihei',pred_cls.shape)

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # print(self.grid_x)
        # print(self.grid_y)


        '''
            将tx,ty,tw,th恢复成bbox的坐标
        '''

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h


        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            '''
                这个targets，是一个【n,6】的张量
                [第几张图，0，cx,cy,dw,dh]

                obj_mask包含的是和anchors的IOU最大的一批数据
                noobj_mask包含的是除去IOU超过阈值的一批数据
            '''
            import time

            # print(pred_boxes.shape)
            # print(pred_cls.shape)
            # print(targets.shape)
            #
            # print('stop here')
            # time.sleep(1000)
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            obj_mask = obj_mask.bool()  # convert int8 to bool
            noobj_mask = noobj_mask.bool()  # convert int8 to bool



            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            '''
                loss由三部分组成：
                    1、（有物体在的cell && 被选中的anchors）对应的tx,ty,tw,th误差
                    2、（有物体在的cell && 被选中的anchors）对应的前背景分类误差
                    3、（没物体在的cell && 被选中的anchors）对应的前背景分类误差
                    4、（有物体在的cell && 被选中的anchors）对应的类别分类误差
            '''
            # 第一部分
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])

            # 第二部分
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            # 第三部分
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            # 按照不同比例组合
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj

            # 第四部分
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()

        # if isinstance(config_path, str):
        #     self.module_defs = parse_model_config(config_path)
        # elif isinstance(config_path, list):
        #     self.module_defs = config_path
        self.module_defs = parse_model_config(config_path)

        self.hyperparams, self.module_list = create_modules(self.module_defs)

        # for module in self.module_list:
        #     print('module:',module[0])
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]

        # print(self.yolo_layers)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        # print('inpus shape',x.shape)
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        # 执行推断过程
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            # 卷积，上采样，池化，正常推断
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                # FPN的东西，跨层连接
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                # resblock的链接，着了的resblock没有什么花里胡哨的东西，就是直接add（还不是concate，就是add）
                # 说明'+'的两层的size完全一致
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                '''
                    解释一下这里为什么这么写 module[0](x,targets,img_dim)
                    而其他地方是写成module(x)

                    首先self.module_list是一个nn.ModuleList，里面存放的是nn.Sequential
                    然后通过for将其中的每一个nn.Sequential提取出来。
                    nn.Sequential中放的就是自定义的nn.Module等东西
                    nn.Sequential中的forward操作，会顺序的执行里面的nn.Module

                    but,nn.Sequential的forawrd，只能接受一个输入参数，如果自定义模型中的forward，
                    需要接受多个参数，那么直接使用nn.Sequential的forward就会报错。

                    此时就需要将nn.Sequential中的module拿出来，以module的方式zhixingforward即可。

                    这也是这里module[0]操作的意义，即将module从nn.sequential中提取出来了。
                '''
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        # print('yolo outputs',yolo_outputs.shape)
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        if cutoff:
            print('do cutoff')
        else:
            print('not cutoff')

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # print(conv_layer)
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
