from __future__ import division

from models.my_yolo import *
from test import *

import torch
from torchvision import transforms
from torch.autograd import Variable
import warnings
import cv2
from matplotlib.ticker import NullLocator
from mylogger import MyLogger

local_logger = MyLogger(filename='./logs/train.log',logger_name='train')

warnings.filterwarnings('ignore')

def preprocess(image):
    '''
    预处理
    :param image:
    :return:
    '''
    img = transforms.ToTensor()(image)
    # Pad to square resolution
    img, _ = pad_to_square(img, 0)
    # Resize
    img = resize(img, 416)
    img.unsqueeze_(0)
    return img

def display(detections,image):
    '''
    可视化
    :param detections:
    :param image:
    :return:
    '''
    # 可视化
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        # print("\t+ Label: %s, Conf: %.5f" % (self.classes[int(cls_pred)], cls_conf.item()))

        box_w = x2 - x1
        box_h = y2 - y1

        color = [random.random() for i in range(3)]
        color.append(1)
        color = tuple(color)
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)


    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.show()

    # 考虑是否保存
    # plt.savefig(f"./dog.png", bbox_inches="tight", pad_inches=0.0)
    plt.close()

def get_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyYolov3(num_class=2, img_size=416).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def inference(model):
    img_path = './data/pics/warehouse1.jpg'

    model.eval()

    frame = cv2.imread(img_path)
    image = Image.open(img_path)
    img = preprocess(image)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    input_imgs = Variable(img.type(Tensor))

    with torch.no_grad():
        # 原始的输出坐标是(center x, center y, width, height)
        detections = model(input_imgs)
        # nms中会转换成(x1, y1, x2, y2)
        detections = non_max_suppression(detections, 0.5, 0.4)[0]

    print('detect res ', len(detections))

    if detections is not None:
        org_h, org_w = frame.shape[:2]
        detections = rescale_boxes(detections, 416, (org_h, org_w))
        display(detections, image)



def get_part_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyYolov3_CV(num_class=2,img_size=416).to(device)
    model.load_state_dict(torch.load(model_path,map_location=device))
    return model


def part_inference(model):
    img_path = './data/pics/warehouse1.jpg'

    model.eval()

    frame = cv2.imread(img_path)
    image = Image.open(img_path)
    img = preprocess(image)

    anchors = [
        [(116, 90), (156, 198), (373, 326)],  # 13*13 上预测最大的
        [(30, 61), (62, 45), (59, 119)],  # 26*26 上预测次大的
        [(10, 13), (16, 30), (33, 23)],  # 13*13 上预测最小的
    ]
    yolo1 = YOLO_NP(anchors[0], 2, 416)
    yolo2 = YOLO_NP(anchors[1], 2, 416)
    yolo3 = YOLO_NP(anchors[2], 2, 416)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    input_imgs = Variable(img.type(Tensor))
    with torch.no_grad():
        o1,o2,o3 = model(input_imgs)
        yolo_output1 = yolo1(o1)
        yolo_output2 = yolo2(o2)
        yolo_output3 = yolo3(o3)
        detections = np.concatenate([yolo_output1,yolo_output2,yolo_output3], 1)

        detections = non_max_suppression_np(detections, 0.5, 0.4)[0]

    print('detect res ', len(detections))
    if detections is not None:
        org_h, org_w = frame.shape[:2]
        detections = rescale_boxes(detections, 416, (org_h, org_w))
        display(detections, image)




if __name__ == '__main__':
    # inference()

    model_path= './weights/yolov3-myyolov3_99_0.96_warehouse.pth'
    # model_path= './weights/yolov3-myyolov3_99_0.355_crowdhuman.t7'

    model_part = get_part_model(model_path)
    dets1 = part_inference(model_part)

    # model = get_model(model_path)
    # dets2 = inference(model)



    # test_input = torch.randn([1,3,416,416])

    # torch.onnx.export(model, test_input, "./weights/yolov3-myyolov3_99_0.355_crowdhuman.onnx", export_params=True)

    # inference(model)
    # print(model.state_dict())

    # torch.save(model.state_dict(),'weights/yolov3-myyolov3_99_0.355_crowdhuman.t7')


