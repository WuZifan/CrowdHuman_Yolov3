import torch.nn as nn
from models.my_yolo import *
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import torch
from collections import OrderedDict
from models.models import *
from utils.datasets import *
from utils.utils import rescale_boxes
from PIL import Image
from matplotlib.ticker import NullLocator
import numpy as np

def loadonnx(path):
    cv2.dnn.readNetFromONNX(path)
    print('load onnx successful')

def pad2square_cv2(image):
    h,w,c = image.shape
    dim_diff = np.abs(h-w)
    pad1,pad2= dim_diff//2 ,dim_diff-dim_diff//2

    if h<=w:
        image = cv2.copyMakeBorder(image,pad1,pad2,0,0,cv2.BORDER_CONSTANT,value=0)
    else:
        image = cv2.copyMakeBorder(image,0,0,pad1,pad2,cv2.BORDER_CONSTANT,value=0)

    return image

def convertcv2nnResult(outs,h,w):
    # 绘制检测矩形
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            print(detection)
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            # numbers are [center_x, center_y, width, height]
            if confidence > 0.5:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    return boxes,confidences

def preprocess4cv2dnn(image_path):
    image = cv2.imread(img_path)

    image = pad2square_cv2(image)

    h, w = image.shape[:2]

    # 干嘛要水平翻转？
    # image = cv2.flip(image, 1)

    blobImage = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (416, 416), None, True, False)
    return blobImage,image

def loadcv2dnnNet(cfg_path,weight_path):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)
    #
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return net

def loadcv2dnnNetT7():
    net = cv2.dnn.readNetFromTorch('./weights/yolov3-myyolov3_99_0.355_crowdhuman.t7')

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return net

def loadcv2dnnNetONNX():
    net = cv2.dnn.readNetFromONNX('./weights/yolov3-myyolov3_99_0.355_crowdhuman.onnx')

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return net

def testcv2nnNet(cfg_path,weight_path,img_path):
    blobImage,image = preprocess4cv2dnn(img_path)
    h,w = image.shape[:2]
    # net = loadcv2dnnNet(cfg_path,weight_path)
    # net = loadcv2dnnNetT7()
    net = loadcv2dnnNetONNX()

    outNames = net.getUnconnectedOutLayersNames()
    net.setInput(blobImage)
    outs = net.forward(outNames)

    boxes,confidences = convertcv2nnResult(outs,h,w)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255), 2, 8, 0)

    cv2.imshow('YOLOv3-tiny-Detection-Demo', image)
    cv2.waitKey(10000)

if __name__ == '__main__':

    cfg_path = './config/yolov3-tiny-person.cfg'
    weight_path='./weights/yolov3_tiny_person_new.weights'
    img_path = './data/pics/persons.jpg'

    testcv2nnNet(cfg_path,weight_path,img_path)

