# -*- coding: utf-8 -*-
"""
@Time          : 2020/05/06 15:07
@Author        : Tianxiaomo
@File          : train.py
@Noice         :
@Modificattion : 添加注释
    @Author    : FelixFu
    @Time      : 2020年11月1日
    @Detail    :

"""
import time
import logging
import os, sys, math
import argparse
from collections import deque
import datetime

import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict

from dataset import Yolo_dataset
from cfg import Cfg
from models import Yolov4
from tool.darknet2pytorch import Darknet

from tool.tv_reference.utils import collate_fn as val_collate
from tool.tv_reference.coco_utils import convert_to_coco_api
from tool.tv_reference.coco_eval import CocoEvaluator

import warnings
warnings.filterwarnings("ignore")


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])  # size (14,9,2)
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])  # size (14,9,2)
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])  # size (14,9,2)
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])  # size (14,9,2)
        # centerpoint distance squared，中心距离的平方
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
                (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]    # a box的w,h; size 14
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]    # b box的w,h; size 9
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)  # size (14,9),值全部是1
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())   size (14,9)
    area_u = area_a[:, None] + area_b - area_i  # size (14,9)
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


class Yolo_loss(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        """初始化Yolo loss
        :param n_classes:   类别数，80
        :param n_anchors:   anchors数，3
        :param device:      设备，cuda
        :param batch:       batch，8
        """
        super(Yolo_loss, self).__init__()
        self.device = device            # 设备
        self.strides = [8, 16, 32]      # 步长
        image_size = 608                # 图片大小 eg.608
        self.n_classes = n_classes      # number of classes eg. 80
        self.n_anchors = n_anchors      # number of anchors eg.3

        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5          # 阈值，iou
        # mask_anchors是三组9个anchors（单位grid）
        # ref_anchors是（9，4）的参考anchors
        # grid_x,grid_y,anchor_w,anchor_h是（8，3，76，76）的索引值和anchor的w/h
        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            # 所有的anchor（单位grid）     eg.[(1.5, 2.0), (2.375, 4.5), (5.0, 3.5), ..., (57.375, 50.125)]
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            # 一组yolo layer层的anchors（单位grid）， 通过anch_masks筛选所有anchors。 eg. [[1.5 2.], [2.375 4.5 ], [5. 3.5 ]]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            # 参考的anchors，（9，4）。并赋值每层yolo layer的anchor值
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)    # size (9, 4)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)

            # 此代码段预定义了grid_x/y, anchor_w/h的占位，供以后使用
            fsize = image_size // self.strides[i]   # 特征图大小  eg. 76
            # grid占位  eg. (8, 3, 76, 76) 8是batch size，3是通道，76是w/h
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)
            # (8,3,76,76) 8是batch size， 3是anchor1，2，3  76是w/h
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(device)
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(device)

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)    # 参考的anchors
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        """通过bbox偏移（pred）、labels等,构建mask等
        :param pred:预测bbox框（bx,by,bw,bh) eg.(8, 3, 76, 76, 4)
        :param labels:真实标签，（8，60，5）
        :param batchsize:batchsize eg. 8
        :param fsize:fsize eg.76
        :param n_ch:eg. 85
        :param output_id: yolo layers id, (0,1,2) . eg. 0
        :return:
        """
        # 此代码段是 target assignment
        # 用于筛选目标，将output值进行重置，将有object数值不变，无object的数值为0。   target mask (8,3,76,76,84)
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device)
        # 判断是否有object。  object mask (8, 3, 76, 76)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device)
        # 对损失函数中x,y 和 w,h 进行缩放，使之数据相近。      target scale(8,3,76,76,2)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        # 网络学习的目标，即将labels变成target，用来匹配 output。     target (8,3,76,76,85)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)

        # 每个图片上object个数。 eg. tensor([ 9, 10, 18,  2, 28, 10, 15, 24], device='cuda:0')
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)

        # 此段代码是，一个batch size中所有图片中object的（cx,cy,w,h),（单位是grid）
        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)   # center x, 相对于grid （8，60）
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)   # center y, 相对于grid （8，60）
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]         # anchor w, 相对于grid （8，60）
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]         # anchor h, 相对于grid （8，60）
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()                             # grid左上角x，   grid，（8，60）
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()                             # grid左上角y，   grid，（8，60）

        # 一个batch size中的每个图片的情况
        for b in range(batchsize):
            n = int(nlabel[b])  # 此张图片中object个数
            if n == 0:  # 这张图片无object，循环下一张图片
                continue

            # 此段代码是，从truth_w/h_all中获取此图片的object的truth_box（n,4)(单位grid）
            truth_box = torch.zeros(n, 4).to(self.device)   # size （14，4）
            truth_box[:n, 2] = truth_w_all[b, :n]   # 将第b张图片中所有object的w赋值给truth_box
            truth_box[:n, 3] = truth_h_all[b, :n]   # 将第b张图片中所有object的h赋值给truth_box
            truth_box[:n, 0] = truth_x_all[b, :n]   # 将第b张图片中所有object的x赋值给truth_box
            truth_box[:n, 1] = truth_y_all[b, :n]   # 将第b张图片中所有object的y赋值给truth_box
            truth_i = truth_i_all[b, :n]            # 将第b张图片中所有object的i赋值给truth_i
            truth_j = truth_j_all[b, :n]            # 将第b张图片中所有object的j赋值给truth_j


            # 计算真实标签（n, 4)和参考anchors(9, 4)的IOU。一张图片上所有object对应的9个anchor的IOU值。
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id], CIoU=True)   # size (14,9)
            """
            eg. (10,9)， 这张图片一共有14个object，然后用14个object的bbox分别与9个anchors对比，得到（14，9）的矩阵。
            tensor([
                [-0.1996, -0.0912, -0.1339,  0.2007, -0.0199,  0.3255,  0.0429,  0.0893, -0.1339],在这个方向找最大值的下标
                [ 0.1819,  0.6982,  0.2370,  0.2094,  0.0398, -0.0615, -0.1277, -0.1708, -0.2239],
                [-0.2092, -0.1494, -0.1591,  0.0056,  0.0293,  0.4500,  0.2312,  0.3901, -0.0069],
                [ 0.2895,  0.1899, -0.0348, -0.0565, -0.1670, -0.1592, -0.2219, -0.2152, -0.2568],
                [-0.2041, -0.1482, -0.1334, -0.0072,  0.0409,  0.3716,  0.4265,  0.5468,  0.0404],
                [-0.1681, -0.0588, -0.0200,  0.2427,  0.3198,  0.8763,  0.3942,  0.1269, -0.1035],
                [-0.0317,  0.3243,  0.2540,  0.6721,  0.2483,  0.0966, -0.0122, -0.1132, -0.1979],
                [-0.1459, -0.0229,  0.0968,  0.2865,  0.6624,  0.3237,  0.3662,  0.0256, -0.1312],
                [-0.0436,  0.2667,  0.3486,  0.8022,  0.3719,  0.1292,  0.0281, -0.1024, -0.1886],
                .......
                [-0.2374, -0.2255, -0.2185, -0.1976, -0.1845, -0.1346, -0.1066,  0.0428,  0.5345]
            ])
            """

            # 此段代码是，best_n_mask的作用是在best_n_all中筛选本yolo layer层中的内容,然后使用best_n来确定是本层yolo layers的第几个anchor
            # 一张图片中每个object对应的anchor值最大情况的下标。 truth对参考9个anchor中IOU最好的情况的n个下标
            best_n_all = anchor_ious_all.argmax(dim=1)  # eg. tensor([6, 6, 2, 7, 2, 6, 3, 4, 5, 3, 5, 1, 5, 3, 7, 6])
            # 此操作是将anchor对应到不同层，其中0代表yolo layer0的anchors，1代表yolo layer 1的anchors，类推3
            best_n = best_n_all % 3  # tensor([0, 2, 2, 1, 0, 0, 2, 0, 0, 2, 0, 0, 1, 1])，
            # 筛选best_n_all中，属于本yolo layer层的anchor
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |  # self.anch_masks[output_id][0] = 0/3/6
                           (best_n_all == self.anch_masks[output_id][1]) |  # self.anch_masks[output_id][1] = 1/4/7
                           (best_n_all == self.anch_masks[output_id][2]))   # self.anch_masks[output_id][2] = 2/5/8

            # 如果本yolo layer层无最大匹配（truth和ref_anchors的IOU），则循环下一张图片
            if sum(best_n_mask) == 0:
                continue

            # 此段代码是，计算预测pred和真实truth的IOU，然后将IOU小于阈值的位置设置成0。
            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)  # pred（17328，4），truth_box (14,4)
            pred_best_iou, _ = pred_ious.max(dim=1)  # size 17328，eg. [True,...]
            pred_best_iou = (pred_best_iou > self.ignore_thre)  # size 17328
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])   # size (3,76,76)
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~ pred_best_iou   # 设置成0意思这些，pred[b]和truth_box的IOU小于阈值，无object

            # 一张图片中，每个object的情况
            for ti in range(best_n.shape[0]):   # 与真实标签最靠近的14个anchors，也是本张图片上的14个object
                if best_n_mask[ti] == 1:    # best_n_mask的作用是在best_n_all中筛选本yolo layer层中的内容，也是这个object是否在本yolo layer层
                    i, j = truth_i[ti], truth_j[ti]  # 第ti（target i）object的i和j
                    a = best_n[ti]                   # 本层yolo layer的第a个anchor
                    obj_mask[b, a, j, i] = 1        # 这个图片（batchsize），这个anchor，这个ij位置，有object， （8，3，76，76）
                    tgt_mask[b, a, j, i, :] = 1     # 这个图片，这个anchor，这个ij位置，有object (8,3,76,76,84)，无置信度

                    # target (8,3,76,76,85)， tensor(0.0588, device='cuda:0', dtype=torch.float64)，这个是网络要学习的目标，偏移值
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    # 将truth内容转换为和pred相同的输出格式
                    target[b, a, j, i, 2] = torch.log(truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1   # 置信度
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1  # 类别为1

                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        # xin是YoloV4网络的输出，形状为[(8,255,76,76),(8,255,38,38),(8,255,19,19)]，值为实数，表示偏移位置； labels
        for output_id, output in enumerate(xin):    # 循环求三个yolo层的loss
            batchsize = output.shape[0]  # output (batchsize, channels, width, height)
            fsize = output.shape[2]     # feature map大小
            n_ch = 5 + self.n_classes   # dx,dy,dw,dh,p,classes

            # 偏移位置，output (batchsize, channels, gridsize, gridsize, [bx,by,bw,bh,pro,cls])
            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2).contiguous()  # .contiguous()

            # logistic activation for xy, obj, cls。 偏移位置sigmoid，变成（0，1）之间
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

            # 偏移位置 与grid融合，相对于在grid上的偏移
            pred = output[..., :4].clone()  # (8,3,76,76,4)
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)

            # loss calculation
            output[..., 4] *= obj_mask  # 设置置信度，无object为0，有object为1
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask  # 通过tgt_mask重置output，使有object的值不变，没有object的值为0。
            output[..., 2:4] *= tgt_scale   # 缩放w，h的比例，使w和h的大小  与  x和y的大小相近

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2],
                                              weight=tgt_scale * tgt_scale, reduction='sum')
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], reduction='sum') / 2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], reduction='sum')
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], reduction='sum')
            loss_l2 += F.mse_loss(input=output, target=target, reduction='sum')

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes


def train(model, device, config, epochs=5, batch_size=1, save_cp=True, log_step=20, img_scale=0.5):
    """训练模型
    :param model: 模型
    :param device: 设备
    :param config: 配置
    :param epochs:  epochs
    :param batch_size:  batchsize
    :param save_cp: 是否存储模型
    :param log_step: log输出间隔
    :param img_scale:   img_scale
    :return:
    """
    # 读取train和val数据集
    train_dataset = Yolo_dataset(config.train_label, config, train=True)
    val_dataset = Yolo_dataset(config.val_label, config, train=False)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch // config.subdivisions, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=config.batch // config.subdivisions, shuffle=True, num_workers=8,
                            pin_memory=True, drop_last=True, collate_fn=val_collate)

    writer = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR,
                           filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}',
                           comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}')

    max_itr = config.TRAIN_EPOCHS * n_train

    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {config.batch}
        Subdivisions:    {config.subdivisions}
        Learning rate:   {config.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:     {config.width}
        Optimizer:       {config.TRAIN_OPTIMIZER}
        Dataset classes: {config.classes}
        Train label path:{config.train_label}
        Pretrained:{config.pretrained}
    ''')

    # learning rate setup
    def burnin_schedule(i):
        if i < config.burn_in:
            factor = pow(i / config.burn_in, 4)
        elif i < config.steps[0]:
            factor = 1.0
        elif i < config.steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    # 优化器
    if config.TRAIN_OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate / config.batch,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config.learning_rate / config.batch,
            momentum=config.momentum,
            weight_decay=config.decay,
        )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=6, min_lr=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 0.001, 1e-6, 20)

    # 损失函数
    criterion = Yolo_loss(device=device, batch=config.batch // config.subdivisions, n_classes=config.classes)

    save_prefix = 'Yolov4_epoch'
    saved_models = deque()
    model.train()
    for epoch in range(epochs):
        # model.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=50) as pbar:
            for i, batch in enumerate(train_loader):
                global_step += 1
                images = batch[0]
                bboxes = batch[1]

                images = images.to(device=device, dtype=torch.float32)
                bboxes = bboxes.to(device=device)

                bboxes_pred = model(images)
                loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)
                loss.backward()

                epoch_loss += loss.item()

                if global_step % config.subdivisions == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if global_step % (log_step * config.subdivisions) == 0:
                    writer.add_scalar('train/Loss', loss.item(), global_step)
                    writer.add_scalar('train/loss_xy', loss_xy.item(), global_step)
                    writer.add_scalar('train/loss_wh', loss_wh.item(), global_step)
                    writer.add_scalar('train/loss_obj', loss_obj.item(), global_step)
                    writer.add_scalar('train/loss_cls', loss_cls.item(), global_step)
                    writer.add_scalar('train/loss_l2', loss_l2.item(), global_step)
                    writer.add_scalar('lr', scheduler.get_lr()[0] * config.batch, global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item(), 'loss_xy': loss_xy.item(),
                                        'loss_wh': loss_wh.item(),
                                        'loss_obj': loss_obj.item(),
                                        'loss_cls': loss_cls.item(),
                                        'loss_l2': loss_l2.item(),
                                        'lr': scheduler.get_lr()[0] * config.batch
                                        })
                    logging.debug('Train step_{}: loss : {},loss xy : {},loss wh : {},'
                                  'loss obj : {}，loss cls : {},loss l2 : {},lr : {}'
                                  .format(global_step, loss.item(), loss_xy.item(),
                                          loss_wh.item(), loss_obj.item(),
                                          loss_cls.item(), loss_l2.item(),
                                          scheduler.get_lr()[0] * config.batch))

                pbar.update(images.shape[0])

            if cfg.use_darknet_cfg:
                eval_model = Darknet(cfg.cfgfile, inference=True)
            else:
                eval_model = Yolov4(cfg.pretrained, n_classes=cfg.classes, inference=True)
            if torch.cuda.device_count() > 1:
                eval_model.load_state_dict(model.module.state_dict())
            else:
                eval_model.load_state_dict(model.state_dict())
            eval_model.to(device)
            evaluator = evaluate(eval_model, val_loader, config, device)
            del eval_model

            stats = evaluator.coco_eval['bbox'].stats
            writer.add_scalar('train/AP', stats[0], global_step)
            writer.add_scalar('train/AP50', stats[1], global_step)
            writer.add_scalar('train/AP75', stats[2], global_step)
            writer.add_scalar('train/AP_small', stats[3], global_step)
            writer.add_scalar('train/AP_medium', stats[4], global_step)
            writer.add_scalar('train/AP_large', stats[5], global_step)
            writer.add_scalar('train/AR1', stats[6], global_step)
            writer.add_scalar('train/AR10', stats[7], global_step)
            writer.add_scalar('train/AR100', stats[8], global_step)
            writer.add_scalar('train/AR_small', stats[9], global_step)
            writer.add_scalar('train/AR_medium', stats[10], global_step)
            writer.add_scalar('train/AR_large', stats[11], global_step)

            if save_cp:
                try:
                    # os.mkdir(config.checkpoints)
                    os.makedirs(config.checkpoints, exist_ok=True)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                save_path = os.path.join(config.checkpoints, f'{save_prefix}{epoch + 1}.pth')
                torch.save(model.state_dict(), save_path)
                logging.info(f'Checkpoint {epoch + 1} saved !')
                saved_models.append(save_path)
                if len(saved_models) > config.keep_checkpoint_max > 0:
                    model_to_remove = saved_models.popleft()
                    try:
                        os.remove(model_to_remove)
                    except:
                        logging.info(f'failed to remove {model_to_remove}')

    writer.close()


@torch.no_grad()
def evaluate(model, data_loader, cfg, device, logger=None, **kwargs):
    """ finished, tested
    """
    # cpu_device = torch.device("cpu")
    model.eval()
    # header = 'Test:'

    coco = convert_to_coco_api(data_loader.dataset, bbox_fmt='coco')
    coco_evaluator = CocoEvaluator(coco, iou_types = ["bbox"], bbox_fmt='coco')

    for images, targets in data_loader:
        model_input = [[cv2.resize(img, (cfg.w, cfg.h))] for img in images]
        model_input = np.concatenate(model_input, axis=0)
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = torch.from_numpy(model_input).div(255.0)
        model_input = model_input.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(model_input)

        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # outputs = outputs.cpu().detach().numpy()
        res = {}
        # for img, target, output in zip(images, targets, outputs):
        for img, target, boxes, confs in zip(images, targets, outputs[0], outputs[1]):
            img_height, img_width = img.shape[:2]
            # boxes = output[...,:4].copy()  # output boxes in yolo format
            boxes = boxes.squeeze(2).cpu().detach().numpy()
            boxes[...,2:] = boxes[...,2:] - boxes[...,:2] # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
            boxes[...,0] = boxes[...,0]*img_width
            boxes[...,1] = boxes[...,1]*img_height
            boxes[...,2] = boxes[...,2]*img_width
            boxes[...,3] = boxes[...,3]*img_height
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # confs = output[...,4:].copy()
            confs = confs.cpu().detach().numpy()
            labels = np.argmax(confs, axis=1).flatten()
            labels = torch.as_tensor(labels, dtype=torch.int64)
            scores = np.max(confs, axis=1).flatten()
            scores = torch.as_tensor(scores, dtype=torch.float32)
            res[target["image_id"].item()] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator


def get_args(**kwargs):
    """用--args方式，更新cfg.py中设置的参数。
    :param kwargs: 参数字典
    :return:    更新后的参数字典
    """
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?',
                        default=128, help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?',
                        default=0.001, help='Learning rate', dest='learning_rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default="weights/pytorch/yolov4.pth",
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='0', help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str, default="datasets/mscoco2017",
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-pretrained', type=str, default="weights/pytorch/yolov4.conv.137.pth",
                        help='pretrained yolov4.conv.137')
    parser.add_argument('-classes', type=int, default=80, help='dataset classes')
    parser.add_argument('-train_label_path', dest='train_label', type=str,
                        default='data/train.txt', help="train label path")
    parser.add_argument('-optimizer', type=str, default='adam', help='training optimizer',  dest='TRAIN_OPTIMIZER')
    parser.add_argument('-iou-type', type=str, default='iou', help='iou type (iou, giou, diou, ciou)', dest='iou_type')
    parser.add_argument('-keep-checkpoint-max', type=int, default=10,
                        help='maximum number of checkpoints to keep. If set 0, all checkpoints will be kept',
                        dest='keep_checkpoint_max')
    args = vars(parser.parse_args())
    cfg.update(args)
    """
    print(edict(cfg))
    {
        'use_darknet_cfg': True, 'cfgfile': '/root/yolov4/cfg/yolov4.cfg', 'batch': 128, 
        'subdivisions': 16, 'width': 608, 'height': 608, 'channels': 3, 'momentum': 0.949, 
        'decay': 0.0005, 'angle': 0, 'saturation': 1.5, 'exposure': 1.5, 'hue': 0.1, 
        'learning_rate': 0.001, 'burn_in': 1000, 'max_batches': 500500, 'steps': [400000, 450000], 
        'policy': [400000, 450000], 'scales': [0.1, 0.1], 'letter_box': 0, 'jitter': 0.2, 
        'classes': 80, 'track': 0, 'w': 608, 'h': 608, 'flip': 1, 'blur': 0, 'gaussian': 0, 
        'boxes': 60, 'TRAIN_EPOCHS': 300, 'train_label': 'data/train.txt', 'val_label': '/root/yolov4/data/val.txt', 
        'TRAIN_OPTIMIZER': 'adam', 'cutmix': 0, 'mosaic': 1, 'mixup': 3, 'checkpoints': '/root/yolov4/checkpoints', 
        'TRAIN_TENSORBOARD_DIR': '/root/yolov4/log', 'iou_type': 'iou', 'keep_checkpoint_max': 10, 
        'batchsize': 128, 'load': 'weights/pytorch/yolov4.pth', 'gpu': '0', 'dataset_dir': 'datasets/mscoco2017', 
        'pretrained': 'weights/pytorch/yolov4.conv.137.pth'}

    """

    return edict(cfg)


def init_logger(log_file=None, log_dir=None, log_level=logging.DEBUG, mode='w', stdout=True):
    """初始化logging变量
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = 'log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=log_level,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


def _get_date_str():
    """返回当前时间
    :return:
    """
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M')


if __name__ == "__main__":
    logging = init_logger(log_dir='log')    # 配置日志logging输出
    cfg = get_args(**Cfg)                   # 将cfg.py和执行参数整合
    """
    {
        'use_darknet_cfg': True, 
        'cfgfile': '/root/yolov4/cfg/yolov4.cfg', 
        'batch': 128, 
        'subdivisions': 16, 
        'width': 608, 
        'height': 608, 
        'channels': 3, 
        'momentum': 0.949, 
        'decay': 0.0005, 
        'angle': 0, 
        'saturation': 1.5, 
        'exposure': 1.5, 
        'hue': 0.1, 
        'learning_rate': 0.00261, 
        'burn_in': 1000, 
        'max_batches': 500500, 
        'steps': [400000, 450000], 
        'policy': [400000, 450000], 
        'scales': [0.1, 0.1], 
        'letter_box': 0, 
        'jitter': 0.2, 
        'classes': 80, 
        'track': 0, 
        'w': 608, 
        'h': 608, 
        'flip': 1, 
        'blur': 0, 
        'gaussian': 0, 
        'boxes': 60, 
        'TRAIN_EPOCHS': 300, 
        'train_label': '/root/yolov4/data/train.txt', 
        'val_label': '/root/yolov4/data/val.txt', 
        'TRAIN_OPTIMIZER': 'adam', 
        'cutmix': 0, 
        'mosaic': 1, 
        'mixup': 3, 
        'checkpoints': '/root/yolov4/checkpoints', 
        'TRAIN_TENSORBOARD_DIR': '/root/yolov4/log', 
        'iou_type': 'iou', 
        'keep_checkpoint_max': 10
    }
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu    # 设定GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 是否使用GPU
    logging.info(f'Using device {device}')      # 显示使用GPU

    # 定义模型
    if cfg.use_darknet_cfg:
        model = Darknet(cfg.cfgfile)
    else:
        model = Yolov4(cfg.pretrained, n_classes=cfg.classes)

    # 多GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)

    try:    # 训练
        train(model=model, config=cfg, epochs=cfg.TRAIN_EPOCHS, device=device)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
