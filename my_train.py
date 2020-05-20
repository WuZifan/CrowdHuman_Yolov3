from __future__ import division

from models.my_yolo import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import *


# from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import warnings
from tqdm import tqdm

from mylogger import MyLogger

local_logger = MyLogger(filename='./logs/train.log',logger_name='train')

warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '2'



def img_label_map(img_list):
    root_path = str(os.getcwd())
    local_path = 'data/labels'
    labels_dir = os.path.join(root_path,local_path)
    label_file_list = []
    for img_path in img_list:
        img_path = img_path.strip()
        temp_path = img_path.split('/')[-1].replace('.jpg','.txt')
        label_file_list.append(os.path.join(labels_dir,temp_path))
    return label_file_list




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    # parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    # parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")

    parser.add_argument("--model_def", type=str, default="config/yolov3_person.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/person.data", help="path to data config file")
    parser.add_argument("--pretrained_weights",default=None,type=str, help="if specified starts from checkpoint model")

    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()

    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    print(class_names)

    # Initiate model
    model = MyYolov3(num_class=2,img_size=416).to(device)
    # 自定义的模块，初始化要模块内部做掉了
    # model.apply(weights_init_normal)

    print(model.yolo_layers)
    print('sceeussfuly load my model')
    time.sleep(1000)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training,mydataset=img_label_map)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]


    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in tqdm(enumerate(dataloader)):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            # print(imgs.shape)

            # 跑了forward方法
            loss, outputs = model(imgs, targets)

            # 反向传播
            loss.backward()

            # 每opt.gradient_accumulations 更新一次参数
            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress  下面就是在打印log之类的东西。
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *["YOLO Layer {}".format(i) for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [("{}_{}".format(name,j+1), metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            # log_str += AsciiTable(metric_table).table
            # print(metric_table)
            log_str +='\n'.join(['\t'.join(line) for line in metric_table])
            log_str += "\nTotal loss {}".format(loss.item())

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += "\n---- ETA {}".format(time_left)

            local_logger.info(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            local_logger.info("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
                mydataset=img_label_map
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            # print(AsciiTable(ap_table).table)
            local_logger.info(str(ap_table))
            local_logger.info("---- mAP {}".format(AP.mean()))

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), "checkpoints/yolov3_ckpt_{}_{}_myyolov3.pth".format(epoch,AP.mean()))
