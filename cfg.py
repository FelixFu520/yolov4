# -*- coding: utf-8 -*-
"""
@Time          : 2020/05/06 21:05
@Author        : Tianxiaomo
@File          : Cfg.py
@Noice         :
@Modificattion : 注释
    @Author    : FelixFU
    @Time      : 2020年10月20日
    @Detail    : 添加注释

"""
import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

Cfg.use_darknet_cfg = True
Cfg.cfgfile = os.path.join(_BASE_DIR, 'cfg', 'yolov4.cfg')

Cfg.batch = 128         # batchsize
Cfg.subdivisions = 16   # batchsize = batch // subdivisions
Cfg.width = 608         # image width
Cfg.height = 608        # image height
Cfg.channels = 3        # image channels
Cfg.momentum = 0.949    # 优化器momentum
Cfg.decay = 0.0005      #
Cfg.angle = 0           #
Cfg.saturation = 1.5    # 饱和度
Cfg.exposure = 1.5      # 明度
Cfg.hue = .1            # 色调

Cfg.learning_rate = 0.00261     # 学习率
Cfg.burn_in = 1000              # 调整学习率使用
Cfg.max_batches = 500500        #
Cfg.steps = [400000, 450000]    # 调整学习率使用
Cfg.policy = Cfg.steps          #
Cfg.scales = .1, .1             #


Cfg.letter_box = 0      # 标签名字
Cfg.jitter = 0.2        # 抖动
Cfg.classes = 80        # number of classes
Cfg.track = 0           #
Cfg.w = Cfg.width       # image width
Cfg.h = Cfg.height      # image height
Cfg.flip = 1            # 反转
Cfg.blur = 0            # 模糊
Cfg.gaussian = 0        # 高斯
Cfg.boxes = 60          # box num

Cfg.TRAIN_EPOCHS = 300                                              # epochs
Cfg.train_label = os.path.join(_BASE_DIR, 'data', 'train.txt')      #   训练数据标签
Cfg.val_label = os.path.join(_BASE_DIR, 'data', 'val.txt')          #   验证数据标签
Cfg.TRAIN_OPTIMIZER = 'adam'                                        #   优化器
'''
image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
...
'''


Cfg.cutmix = 0          # cutmix, two pics
Cfg.mosaic = 1          # mosaic, four pics
if Cfg.mosaic and Cfg.cutmix:
    Cfg.mixup = 4   # cutmix and mosaic
elif Cfg.cutmix:
    Cfg.mixup = 2   # cutmix
elif Cfg.mosaic:
    Cfg.mixup = 3   # mosaic

Cfg.checkpoints = os.path.join(_BASE_DIR, 'checkpoints')            # checkpoints DIR
Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'log')          # TensorBoard DIR

Cfg.iou_type = 'iou'  # 'giou', 'diou', 'ciou'                      # IOUs

Cfg.keep_checkpoint_max = 10                                        # 保存模型的最多个数
