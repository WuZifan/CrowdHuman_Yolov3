import torch.nn as nn
import torch
import numpy as np
from utils.utils import build_targets, to_cpu, non_max_suppression
from models.yolov3_output import YOLO_NP

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

class DarkConv(nn.Module):
    def __init__(self,inchannel,outchannel,kernel_size,stride,padding):
        super(DarkConv, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=inchannel,
                      out_channels=outchannel,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride,
                      bias=False),
            nn.BatchNorm2d(outchannel,momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self,x):
        return self.model(x)

class DarkResidual(nn.Module):
    def __init__(self,inchannel):
        super(DarkResidual, self).__init__()
        self.model = nn.Sequential(
            DarkConv(inchannel,inchannel//2,kernel_size=1,stride=1,padding=0),
            DarkConv(inchannel//2,inchannel,kernel_size=3,stride=1,padding=1)
        )

    def forward(self,x):
        return x + self.model(x)


class DarkConvSet(nn.Module):
    def __init__(self,inchannel,outchannel):
        # 384 128
        super(DarkConvSet, self).__init__()
        self.model = nn.Sequential(
            DarkConv(inchannel=inchannel, outchannel=outchannel,
                     kernel_size=1, stride=1, padding=0),
            DarkConv(inchannel=outchannel, outchannel=2*outchannel,
                     kernel_size=3, stride=1, padding=1),
            DarkConv(inchannel=2*outchannel, outchannel=outchannel,
                     kernel_size=1, stride=1, padding=0),
            DarkConv(inchannel=outchannel, outchannel=2*outchannel,
                     kernel_size=3, stride=1, padding=1),
            DarkConv(inchannel=2*outchannel, outchannel=outchannel,
                     kernel_size=1, stride=1, padding=0),
        )

    def forward(self,x):
        return self.model(x)


# class DarkUpSample(nn.Module):
#     def __init__(self):
#         super(DarkUpSample, self).__init__()
#
#     def forward(self,x):
#         return nn.functional.interpolate(x,scale_factor=2,mode='nearest')
#         # return F.upsample(x,scale_factor=2,mode='nearest')

class DarkUpSample2(nn.Module):
    def __init__(self,channel):
        super(DarkUpSample2, self).__init__()
        self.model = nn.ConvTranspose2d(in_channels=channel,
                                        out_channels=channel,
                                        kernel_size=2,stride=2)

    def forward(self,x):
        return self.model(x)




class MyYolov3(nn.Module):

    anchors = [
        [(116, 90), (156, 198), (373, 326)], # 13*13 上预测最大的
        [(30, 61), (62, 45), (59, 119)], # 26*26 上预测次大的
        [(10, 13), (16, 30), (33, 23)], # 13*13 上预测最小的
    ]

    def __init__(self,num_class = 1,img_size=416):
        super(MyYolov3, self).__init__()
        self.start_model=nn.Sequential(
            self._create_conv(inchannel=3,outchannel=32,kernel_size=3,stride=1,padding=1),
            self._create_conv(inchannel=32,outchannel=64,kernel_size=3,stride=2,padding=1),
        )

        # DownSample
        self.block1 = self._create_model(64,1)

        self.downSample1 = self._create_conv(inchannel=64,outchannel=128,kernel_size=3,stride=2,padding=1)

        self.block2 = self._create_model(128,2)

        self.downSample2 = self._create_conv(inchannel=128,outchannel=256,kernel_size=3,stride=2,padding=1)

        self.block3 = self._create_model(256,8)

        self.downSample3 = self._create_conv(inchannel=256,outchannel=512,kernel_size=3,stride=2,padding=1)

        self.block4 = self._create_model(512,8)

        self.downSample4 = self._create_conv(inchannel=512,outchannel=1024,kernel_size=3,stride=2,padding=1)

        self.block5 = self._create_model(1024,4)


        # UpSample
        self.convset1 = DarkConvSet(inchannel=1024,outchannel=512)

        self.header1 = nn.Sequential(
            DarkConv(inchannel=512,outchannel=1024,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=1024,out_channels=3*(1+4+num_class),kernel_size=1,stride=1,padding=0)
        )

        self.upsample1 = self._create_upsample(512)

        self.convset2 = DarkConvSet(inchannel=256+512,outchannel=256)

        self.header2 = nn.Sequential(
            DarkConv(inchannel=256,outchannel=512,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=512,out_channels=3*(1+4+num_class),kernel_size=1,stride=1,padding=0)
        )

        self.upsample2 = self._create_upsample(256)

        self.convset3 = DarkConvSet(inchannel=128+256,outchannel=128)

        self.header3 = nn.Sequential(
            DarkConv(inchannel=128,outchannel=256,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=256,out_channels=3*(1+4+num_class),kernel_size=1,stride=1,padding=0)
        )


        self.yolo_layer1 = YOLOLayer(self.anchors[0],num_class,img_size)
        self.yolo_layer2 = YOLOLayer(self.anchors[1],num_class,img_size)
        self.yolo_layer3 = YOLOLayer(self.anchors[2],num_class,img_size)

        self.yolo_layers = [self.yolo_layer1,self.yolo_layer2,self.yolo_layer3]

        self.seen = 0

    def forward(self,x,targets=None):

        img_dim = x.shape[2]

        x = self.start_model(x)

        x = self.block1(x)
        x = self.downSample1(x)

        x = self.block2(x)
        x = self.downSample2(x)

        layer1 = self.block3(x)
        x = self.downSample3(layer1)

        layer2 = self.block4(x)
        x = self.downSample4(layer2)

        layer3 = self.block5(x)

        x = self.convset1(layer3)
        #print('convset1 x',x.shape)
        output1 = self.header1(x)

        x = self.upsample1(x)

        x = torch.cat((x, layer2), dim=1)
        #print('cat 1 shape',x.shape,layer2.shape)

        x = self.convset2(x)
        #print('convset2 x',x.shape)
        output2 = self.header2(x)


        x = self.upsample2(x)
        x = torch.cat((x,layer1),dim=1)
        #print('cat 2 shape',x.shape,layer1.shape)

        x = self.convset3(x)
        output3 = self.header3(x)


        yolo_output1, loss1= self.yolo_layer1(output1, targets, img_dim)
        yolo_output2, loss2 = self.yolo_layer2(output2, targets, img_dim)
        yolo_output3, loss3 = self.yolo_layer3(output3, targets, img_dim)


        yolo_outputs = to_cpu(torch.cat([yolo_output1,yolo_output2,yolo_output3], 1))

        loss = loss1+loss2+loss3

        return yolo_outputs if targets is None else (loss,yolo_outputs)

    # def _yolo(self,):


    def _create_upsample(self,inchannel):
        model = nn.Sequential(
            self._create_conv(inchannel=inchannel,outchannel=inchannel//2,kernel_size=1,stride=1,padding=0),
            DarkUpSample2(inchannel//2)
        )

        return model

    def _create_model(self,inchannel,repeat):
        layers = []

        for i in range(repeat):
            # layers.append(DarkConv(inchannel=inchannel,outchannel=inchannel//2,
            #                        kernel_size=1,stride=1,padding=0))
            # layers.append(DarkConv(inchannel=inchannel//2,outchannel=inchannel,
            #                        kernel_size=1,stride=1,padding=0))
            layers.append(DarkResidual(inchannel=inchannel))

        model = nn.Sequential(*layers)
        return model

    def _create_conv(self,inchannel,outchannel,kernel_size,stride,padding):
        model = nn.Sequential(
            nn.Conv2d(in_channels=inchannel,
                      out_channels=outchannel,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride,
                      bias=False),
            nn.BatchNorm2d(outchannel,momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.1)
        )
        return model


class MyYolov3_CV(nn.Module):

    anchors = [
        [(116, 90), (156, 198), (373, 326)], # 13*13 上预测最大的
        [(30, 61), (62, 45), (59, 119)], # 26*26 上预测次大的
        [(10, 13), (16, 30), (33, 23)], # 13*13 上预测最小的
    ]

    def __init__(self,num_class = 1,img_size=416):
        super(MyYolov3_CV, self).__init__()
        self.start_model=nn.Sequential(
            self._create_conv(inchannel=3,outchannel=32,kernel_size=3,stride=1,padding=1),
            self._create_conv(inchannel=32,outchannel=64,kernel_size=3,stride=2,padding=1),
        )

        # DownSample
        self.block1 = self._create_model(64,1)

        self.downSample1 = self._create_conv(inchannel=64,outchannel=128,kernel_size=3,stride=2,padding=1)

        self.block2 = self._create_model(128,2)

        self.downSample2 = self._create_conv(inchannel=128,outchannel=256,kernel_size=3,stride=2,padding=1)

        self.block3 = self._create_model(256,8)

        self.downSample3 = self._create_conv(inchannel=256,outchannel=512,kernel_size=3,stride=2,padding=1)

        self.block4 = self._create_model(512,8)

        self.downSample4 = self._create_conv(inchannel=512,outchannel=1024,kernel_size=3,stride=2,padding=1)

        self.block5 = self._create_model(1024,4)


        # UpSample
        self.convset1 = DarkConvSet(inchannel=1024,outchannel=512)

        self.header1 = nn.Sequential(
            DarkConv(inchannel=512,outchannel=1024,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=1024,out_channels=3*(1+4+num_class),kernel_size=1,stride=1,padding=0)
        )

        self.upsample1 = self._create_upsample(512)

        self.convset2 = DarkConvSet(inchannel=256+512,outchannel=256)

        self.header2 = nn.Sequential(
            DarkConv(inchannel=256,outchannel=512,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=512,out_channels=3*(1+4+num_class),kernel_size=1,stride=1,padding=0)
        )

        self.upsample2 = self._create_upsample(256)

        self.convset3 = DarkConvSet(inchannel=128+256,outchannel=128)

        self.header3 = nn.Sequential(
            DarkConv(inchannel=128,outchannel=256,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=256,out_channels=3*(1+4+num_class),kernel_size=1,stride=1,padding=0)
        )



        self.seen = 0

    def forward(self,x,targets=None):

        img_dim = x.shape[2]

        x = self.start_model(x)

        x = self.block1(x)
        x = self.downSample1(x)

        x = self.block2(x)
        x = self.downSample2(x)

        layer1 = self.block3(x)
        x = self.downSample3(layer1)

        layer2 = self.block4(x)
        x = self.downSample4(layer2)

        layer3 = self.block5(x)

        x = self.convset1(layer3)
        #print('convset1 x',x.shape)
        output1 = self.header1(x)

        x = self.upsample1(x)

        x = torch.cat((x, layer2), dim=1)
        #print('cat 1 shape',x.shape,layer2.shape)

        x = self.convset2(x)
        #print('convset2 x',x.shape)
        output2 = self.header2(x)


        x = self.upsample2(x)
        x = torch.cat((x,layer1),dim=1)
        #print('cat 2 shape',x.shape,layer1.shape)

        x = self.convset3(x)
        output3 = self.header3(x)


        return output1,output2,output3

    # def _yolo(self,):


    def _create_upsample(self,inchannel):
        model = nn.Sequential(
            self._create_conv(inchannel=inchannel,outchannel=inchannel//2,kernel_size=1,stride=1,padding=0),
            DarkUpSample2(inchannel//2)
        )

        return model

    def _create_model(self,inchannel,repeat):
        layers = []

        for i in range(repeat):
            # layers.append(DarkConv(inchannel=inchannel,outchannel=inchannel//2,
            #                        kernel_size=1,stride=1,padding=0))
            # layers.append(DarkConv(inchannel=inchannel//2,outchannel=inchannel,
            #                        kernel_size=1,stride=1,padding=0))
            layers.append(DarkResidual(inchannel=inchannel))

        model = nn.Sequential(*layers)
        return model

    def _create_conv(self,inchannel,outchannel,kernel_size,stride,padding):
        model = nn.Sequential(
            nn.Conv2d(in_channels=inchannel,
                      out_channels=outchannel,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride,
                      bias=False),
            nn.BatchNorm2d(outchannel,momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.1)
        )
        return model


if __name__ == '__main__':
    weights_path = '../../weights/yolov3_ckpt_99_0.8027009108794954.pth'
    test_input = torch.rand([2,3,416,416])
    model = MyYolov3_CV(num_class=2)

    # print(model)
    o1,o2,o3 = model(test_input)

    anchors = [
        [(116, 90), (156, 198), (373, 326)],  # 13*13 上预测最大的
        [(30, 61), (62, 45), (59, 119)],  # 26*26 上预测次大的
        [(10, 13), (16, 30), (33, 23)],  # 13*13 上预测最小的
    ]
    yolo1 = YOLO_NP(anchors[0],2,416)
    yolo2 = YOLO_NP(anchors[1],2,416)
    yolo3 = YOLO_NP(anchors[2],2,416)

    o1 = o1.detach().numpy()

    print(o1.shape,type(o1))

    yolo_output1 = yolo1(o1)
    yolo_output2 = yolo1(o2)
    yolo_output3 = yolo1(o3)

    final_output = np.concatenate([yolo_output1,yolo_output2,yolo_output3], 1)

    print(final_output.shape)

    # summary(model, input_size=(3, 416, 416))





