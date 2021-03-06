# -*- coding: utf-8 -*-
"""
@Time          : 2020/05/06 21:09
@Author        : Tianxiaomo，modify by felixfu
@File          : dataset.py
@Noice         :
@Modificattion : 注释
    @Author    : FelixFu
    @Time      : 2020/10/20
    @Detail    : 添加注释

"""

import os
import random

import cv2
import numpy as np

import torch
from torch.utils.data.dataset import Dataset


def rand_uniform_strong(min_value, max_value):
    """
    随机在（min，max）中产生个值
    :param min_value:
    :param max_value:
    :return:
    """
    if min_value > max_value:
        swap = min_value
        min_value = max_value
        max_value = swap
    return random.random() * (max_value - min_value) + min_value


def rand_scale(s):
    """
    随机在 (1, s)和 1/（1，s）中选择个数
    :param s:
    :return:
    """
    scale = rand_uniform_strong(1, s)
    if random.randint(0, 1) % 2:
        return scale
    return 1. / scale


def rand_precalc_random(min, max, random_part):
    """
    功能：return (random_part * (max - min)) + min
    :param min:
    :param max:
    :param random_part:
    :return:
    """
    if max < min:
        swap = min
        min = max
        max = swap
    return (random_part * (max - min)) + min


def fill_truth_detection(bboxes, num_boxes, classes, flip, dx, dy, sx, sy, net_w, net_h):
    """
    功能：根据裁剪图片，修改truth
    :param bboxes: 这张图片的所有bbox。bboxes
    :param num_boxes: self.cfg.boxes
    :param classes: 80个类别，self.cfg.classes
    :param flip: 翻转，flip
    :param dx: pleft
    :param dy: ptop
    :param sx: swidth
    :param sy: sheight
    :param net_w: self.cfg.w
    :param net_h: self.cfg.h
    :return: 更新的bboxes, min_w_h
    """
    if bboxes.shape[0] == 0:    # eg. bboxes.shape (11,5)
        return bboxes, 10000
    np.random.shuffle(bboxes)

    # bboxes: x1,y1,x2,y2,id    x1,y1,x2,y2,id     x1,y1,x2,y2,id ...。缩小bbox使之适应裁剪内容
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    # 删除无框bboxes，按where条件删除
    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    if bboxes.shape[0] == 0:
        return bboxes, 10000

    bboxes = bboxes[np.where((bboxes[:, 4] < classes) & (bboxes[:, 4] >= 0))[0]]

    if bboxes.shape[0] > num_boxes:
        bboxes = bboxes[:num_boxes]

    # 最小的bbox的w或h
    min_w_h = np.array([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]]).min()

    # 使bboxes相对于net_w(608)的位置，修改为相对于sx（裁剪大小）的位置
    bboxes[:, 0] *= (net_w / sx)
    bboxes[:, 2] *= (net_w / sx)
    bboxes[:, 1] *= (net_h / sy)
    bboxes[:, 3] *= (net_h / sy)

    if flip:
        temp = net_w - bboxes[:, 0]
        bboxes[:, 0] = net_w - bboxes[:, 2]
        bboxes[:, 2] = temp

    return bboxes, min_w_h


def rect_intersection(a, b):
    """
    功能：矩形框a和b的 交集框的下标
    :param a:
    :param b:
    :return:[minx, miny, maxx, maxy]，矩形框a和b的 交集框的下标
    """
    minx = max(a[0], b[0])
    miny = max(a[1], b[1])

    maxx = min(a[2], b[2])
    maxy = min(a[3], b[3])
    return [minx, miny, maxx, maxy]


def image_data_augmentation(mat, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp, gaussian_noise, blur,
                            truth):
    """
    功能：对mat进行数据增强操作
    ai = image_data_augmentation(img, self.cfg.w, self.cfg.h, pleft, ptop, swidth, sheight, flip,
                                         dhue, dsat, dexp, gaussian_noise, blur, truth)
    :param mat: img
    :param w: self.cfg.w,
    :param h: self.cfg.h
    :param pleft: pleft
    :param ptop: ptop
    :param swidth: swidth
    :param sheight: sheight
    :param flip: flip
    :param dhue:dhue
    :param dsat:dsat
    :param dexp:dexp
    :param gaussian_noise:gaussian_noise
    :param blur:blur
    :param truth:truth，裁剪后的labels，这张图片的
    :return:sized， bboxes隐式返回
    """
    try:
        img = mat
        oh, ow, _ = img.shape
        pleft, ptop, swidth, sheight = int(pleft), int(ptop), int(swidth), int(sheight)
        # crop
        src_rect = [pleft, ptop, swidth + pleft, sheight + ptop]  # x1,y1,x2,y2, 截取rect。坐标相对自己
        img_rect = [0, 0, ow, oh]  # img图片的rect
        new_src_rect = rect_intersection(src_rect, img_rect)  # 截取rect和img图片rect交集, [106, 0, 640, 408]。坐标相对自己

        # 裁剪结果图坐标，在w和h的相对位置上，坐标相对于img
        dst_rect = [max(0, -pleft), max(0, -ptop), max(0, -pleft) + new_src_rect[2] - new_src_rect[0],
                    max(0, -ptop) + new_src_rect[3] - new_src_rect[1]] # eg. [0, 32, 534, 440]

        # cv2.Mat sized
        if src_rect[0] == 0 and src_rect[1] == 0 and src_rect[2] == img.shape[0] and src_rect[3] == img.shape[1]:
            sized = cv2.resize(img, (w, h), cv2.INTER_LINEAR)   # 截取rect同img大小
        else:   # 截取rect同img不一样大
            cropped = np.zeros([sheight, swidth, 3])    # size （440， 540， 3），原图大小（640， 247， 3）
            cropped[:, :, ] = np.mean(img, axis=(0, 1)) # shape 3, eg. [106, 89, 84]

            # cropped[32:440, 0:534] = img[0:408, 106:640], img[src_rect  U img_rect] --> cropped
            cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = \
                img[new_src_rect[1]:new_src_rect[3], new_src_rect[0]:new_src_rect[2]]

            # resize
            sized = cv2.resize(cropped, (w, h), cv2.INTER_LINEAR)

        # flip
        if flip:
            # cv2.Mat cropped
            sized = cv2.flip(sized, 1)  # 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)

        # HSV augmentation
        # cv2.COLOR_BGR2HSV, cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2BGR, cv2.COLOR_HSV2RGB
        if dsat != 1 or dexp != 1 or dhue != 0:
            if img.shape[2] >= 3:
                hsv_src = cv2.cvtColor(sized.astype(np.float32), cv2.COLOR_RGB2HSV)  # RGB to HSV
                hsv = cv2.split(hsv_src)
                hsv[1] *= dsat
                hsv[2] *= dexp
                hsv[0] += 179 * dhue
                hsv_src = cv2.merge(hsv)
                sized = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)  # HSV to RGB (the same as previous)
            else:
                sized *= dexp

        if blur:
            if blur == 1:
                dst = cv2.GaussianBlur(sized, (17, 17), 0)
                # cv2.bilateralFilter(sized, dst, 17, 75, 75)
            else:
                ksize = (blur / 2) * 2 + 1
                dst = cv2.GaussianBlur(sized, (ksize, ksize), 0)

            if blur == 1:
                img_rect = [0, 0, sized.cols, sized.rows]
                for b in truth:
                    left = (b.x - b.w / 2.) * sized.shape[1]
                    width = b.w * sized.shape[1]
                    top = (b.y - b.h / 2.) * sized.shape[0]
                    height = b.h * sized.shape[0]
                    roi(left, top, width, height)
                    roi = roi & img_rect
                    dst[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[3]] = sized[roi[0]:roi[0] + roi[2],
                                                                          roi[1]:roi[1] + roi[3]]

            sized = dst

        if gaussian_noise:
            noise = np.array(sized.shape)
            gaussian_noise = min(gaussian_noise, 127)
            gaussian_noise = max(gaussian_noise, 0)
            cv2.randn(noise, 0, gaussian_noise)  # mean and variance
            sized = sized + noise
    except:
        print("OpenCV can't augment image: " + str(w) + " x " + str(h))
        sized = mat

    return sized


def filter_truth(bboxes, dx, dy, sx, sy, xd, yd):
    """
    功能：未注释
    :param bboxes:
    :param dx:
    :param dy:
    :param sx:
    :param sy:
    :param xd:
    :param yd:
    :return:
    """
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    bboxes[:, 0] += xd
    bboxes[:, 2] += xd
    bboxes[:, 1] += yd
    bboxes[:, 3] += yd

    return bboxes


def blend_truth_mosaic(out_img, img, bboxes, w, h, cut_x, cut_y, i_mixup,
                       left_shift, right_shift, top_shift, bot_shift):
    """
    功能：未注释
    :param out_img:
    :param img:
    :param bboxes:
    :param w:
    :param h:
    :param cut_x:
    :param cut_y:
    :param i_mixup:
    :param left_shift:
    :param right_shift:
    :param top_shift:
    :param bot_shift:
    :return:
    """
    left_shift = min(left_shift, w - cut_x)
    top_shift = min(top_shift, h - cut_y)
    right_shift = min(right_shift, cut_x)
    bot_shift = min(bot_shift, cut_y)

    if i_mixup == 0:
        bboxes = filter_truth(bboxes, left_shift, top_shift, cut_x, cut_y, 0, 0)
        out_img[:cut_y, :cut_x] = img[top_shift:top_shift + cut_y, left_shift:left_shift + cut_x]
    if i_mixup == 1:
        bboxes = filter_truth(bboxes, cut_x - right_shift, top_shift, w - cut_x, cut_y, cut_x, 0)
        out_img[:cut_y, cut_x:] = img[top_shift:top_shift + cut_y, cut_x - right_shift:w - right_shift]
    if i_mixup == 2:
        bboxes = filter_truth(bboxes, left_shift, cut_y - bot_shift, cut_x, h - cut_y, 0, cut_y)
        out_img[cut_y:, :cut_x] = img[cut_y - bot_shift:h - bot_shift, left_shift:left_shift + cut_x]
    if i_mixup == 3:
        bboxes = filter_truth(bboxes, cut_x - right_shift, cut_y - bot_shift, w - cut_x, h - cut_y, cut_x, cut_y)
        out_img[cut_y:, cut_x:] = img[cut_y - bot_shift:h - bot_shift, cut_x - right_shift:w - right_shift]

    return out_img, bboxes


def draw_box(img, bboxes):
    """
    功能：将bboxes中的矩形框画在img上
    :param img:
    :param bboxes:
    :return:
    """
    for b in bboxes:
        img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    return img


class Yolo_dataset(Dataset):
    def __init__(self, lable_path, cfg, train=True):
        super(Yolo_dataset, self).__init__()
        if cfg.mixup == 2:
            print("cutmix=1 - isn't supported for Detector")
            raise
        elif cfg.mixup == 2 and cfg.letter_box:
            print("Combination: letter_box=1 & mosaic=1 - isn't supported, use only 1 of these parameters")
            raise

        self.cfg = cfg
        self.train = train

        # 读取数据标签到truth中
        truth = {}
        f = open(lable_path, 'r', encoding='utf-8')
        '''
        数据格式
        image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
        image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
        ...
        '''
        for line in f.readlines():
            data = line.split(" ")
            truth[data[0]] = []
            for i in data[1:]:
                truth[data[0]].append([int(float(j)) for j in i.split(',')])

        self.truth = truth
        self.imgs = list(self.truth.keys())

    def __len__(self):
        return len(self.truth.keys())

    def __getitem__(self, index):
        """
        功能：读取train和val的数据
        :param index:
        :return:
        """
        if not self.train:
            return self._get_val_item(index)

        img_path = self.imgs[index]                                     # 图片路径
        bboxes = np.array(self.truth.get(img_path), dtype=np.float)     # bboxes
        use_mixup = self.cfg.mixup                                      # 使用cutmix、mosica

        # 随机选择，是否mosica，或者cutmix
        if random.randint(0, 1):
            use_mixup = 0

        if use_mixup == 3:  # 进行mosaic，选取中心点， 在图片中随机选择一点，作为cutmix中心点
            min_offset = 0.2
            cut_x = random.randint(int(self.cfg.w * min_offset), int(self.cfg.w * (1 - min_offset)))    # eg. 464
            cut_y = random.randint(int(self.cfg.h * min_offset), int(self.cfg.h * (1 - min_offset)))    # eg. 357

        dhue, dsat, dexp, flip, blur, gaussian_noise = 0, 0, 0, 0, 0, 0    # delta的HSE，是否翻转、模糊、高斯

        out_img = np.zeros([self.cfg.h, self.cfg.w, 3])                    # 输出图片
        out_bboxes = []                                                    # 输出bboxes

        # for循环，cutmix多张图片（cutmix1张，cutmix4张）
        for i in range(use_mixup + 1):
            if i != 0:  # 从datasets中随机选择一张图片，用作mosica
                img_path = random.choice(list(self.truth.keys()))
                bboxes = np.array(self.truth.get(img_path), dtype=np.float)

            # 读取原始图片
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                continue


            oh, ow, oc = img.shape  # original height/width/channels， eg. (427,640,3)
            dh, dw, dc = np.array(np.array([oh, ow, oc]) * self.cfg.jitter, dtype=np.int)  # delta height/width/channels， eg. (85,128,0)

            # delta的色彩三要素
            dhue = rand_uniform_strong(-self.cfg.hue, self.cfg.hue) # delta H=色相决定是什么颜色
            dsat = rand_scale(self.cfg.saturation)                  # delta S=纯度决定颜色浓淡
            dexp = rand_scale(self.cfg.exposure)                    # delta B=明度决定照射在颜色上的白光有多亮

            # 首先生成一些随机偏移的坐标，分别代表左右上下
            pleft = random.randint(-dw, dw)
            pright = random.randint(-dw, dw)
            ptop = random.randint(-dh, dh)
            pbot = random.randint(-dh, dh)

            # 是否反转
            flip = random.randint(0, 1) if self.cfg.flip else 0

            # 是否图像模糊
            if self.cfg.blur:
                tmp_blur = random.randint(0, 2)  # 0 - disable, 1 - blur background, 2 - blur the whole image
                if tmp_blur == 0:
                    blur = 0
                elif tmp_blur == 1:
                    blur = 1
                else:
                    blur = self.cfg.blur

            # 是否高斯噪音
            if self.cfg.gaussian and random.randint(0, 1):
                gaussian_noise = self.cfg.gaussian
            else:
                gaussian_noise = 0

            # 是否矫正box
            if self.cfg.letter_box:
                img_ar = ow / oh
                net_ar = self.cfg.w / self.cfg.h
                result_ar = img_ar / net_ar     # 看img和net那个更扁（长方形）
                print(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f, result_ar = %f \n" % (ow, oh,
                      self.cfg.w, self.cfg.h, img_ar, net_ar, result_ar));
                if result_ar > 1:  # sheight - should be increased， width大
                    oh_tmp = ow / net_ar
                    delta_h = (oh_tmp - oh) / 2
                    ptop = ptop - delta_h
                    pbot = pbot - delta_h
                    print(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot = %f \n" % (
                          result_ar, oh_tmp, delta_h, ptop, pbot));
                else:  # swidth - should be increased，height大
                    ow_tmp = oh * net_ar
                    delta_w = (ow_tmp - ow) / 2
                    pleft = pleft - delta_w
                    pright = pright - delta_w
                    print(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f, pright = %f \n" % (
                          result_ar, ow_tmp, delta_w, pleft, pright));

            # 裁剪部分的长和宽
            swidth = ow - pleft - pright
            sheight = oh - ptop - pbot

            # 根据裁剪图片，修改truth，并获得bboxes最小的w或h
            truth, min_w_h = fill_truth_detection(bboxes, self.cfg.boxes, self.cfg.classes, flip, pleft, ptop, swidth,
                                                  sheight, self.cfg.w, self.cfg.h)
            if (min_w_h / 8) < blur and blur > 1:  # disable blur if one of the objects is too small
                blur = min_w_h / 8

            # 数据增强，只有个flip会改变labels，其他操作均不会，返回resized img
            ai = image_data_augmentation(img, self.cfg.w, self.cfg.h, pleft, ptop, swidth, sheight, flip,
                                         dhue, dsat, dexp, gaussian_noise, blur, truth)

            if use_mixup == 0:  # 不 cutmix
                out_img = ai
                out_bboxes = truth
            if use_mixup == 1:  # 两张图片 cutmix
                if i == 0:
                    old_img = ai.copy()
                    old_truth = truth.copy()
                elif i == 1:
                    out_img = cv2.addWeighted(ai, 0.5, old_img, 0.5)
                    out_bboxes = np.concatenate([old_truth, truth], axis=0)
            elif use_mixup == 3:    # 4张图片 cutmix
                if flip:
                    tmp = pleft
                    pleft = pright
                    pright = tmp

                left_shift = int(min(cut_x, max(0, (-int(pleft) * self.cfg.w / swidth))))
                top_shift = int(min(cut_y, max(0, (-int(ptop) * self.cfg.h / sheight))))

                right_shift = int(min((self.cfg.w - cut_x), max(0, (-int(pright) * self.cfg.w / swidth))))
                bot_shift = int(min(self.cfg.h - cut_y, max(0, (-int(pbot) * self.cfg.h / sheight))))

                out_img, out_bbox = blend_truth_mosaic(out_img, ai, truth.copy(), self.cfg.w, self.cfg.h, cut_x,
                                                       cut_y, i, left_shift, right_shift, top_shift, bot_shift)
                out_bboxes.append(out_bbox)
                # print(img_path)
        if use_mixup == 3:
            out_bboxes = np.concatenate(out_bboxes, axis=0)
        out_bboxes1 = np.zeros([self.cfg.boxes, 5])
        out_bboxes1[:min(out_bboxes.shape[0], self.cfg.boxes)] = out_bboxes[:min(out_bboxes.shape[0], self.cfg.boxes)]
        return out_img, out_bboxes1

    def _get_val_item(self, index):
        """

        :param index:
        :return: img, target
        """
        img_path = self.imgs[index]
        bboxes_with_cls_id = np.array(self.truth.get(img_path), dtype=np.float)
        img = cv2.imread(img_path)
        # img_height, img_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (self.cfg.w, self.cfg.h))
        # img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        num_objs = len(bboxes_with_cls_id)
        target = {}
        # boxes to coco format
        boxes = bboxes_with_cls_id[...,:4]
        boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # box width, box height
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(bboxes_with_cls_id[...,-1].flatten(), dtype=torch.int64)
        target['image_id'] = torch.tensor([get_image_id(img_path)])
        target['area'] = (target['boxes'][:,3])*(target['boxes'][:,2])
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        return img, target


def get_image_id(filename:str) -> int:
    """
    Convert a string to a integer.
    Make sure that the images and the `image_id`s are in one-one correspondence.
    There are already `image_id`s in annotations of the COCO dataset,
    in which case this function is unnecessary.
    For creating one's own `get_image_id` function, one can refer to
    https://github.com/google/automl/blob/master/efficientdet/dataset/create_pascal_tfrecord.py#L86
    or refer to the following code (where the filenames are like 'level1_123.jpg')
    # >>> lv, no = os.path.splitext(os.path.basename(filename))[0].split("_")
    # >>> lv = lv.replace("level", "")
    # >>> no = f"{int(no):04d}"
    # >>> return int(lv+no)
    """
    no = os.path.splitext(os.path.basename(filename))[0]
    return int(no)
    # raise NotImplementedError("Create your own 'get_image_id' function")
    # lv, no = os.path.splitext(os.path.basename(filename))[0].split("_")
    # lv = lv.replace("level", "")
    # no = f"{int(no):04d}"
    # return int(lv+no)


if __name__ == "__main__":
    from cfg import Cfg
    import matplotlib.pyplot as plt

    random.seed(2020)
    np.random.seed(2020)
    dataset = Yolo_dataset(Cfg.train_label, Cfg)
    for i in range(10):
        out_img, out_bboxes = dataset.__getitem__(i)
        a = draw_box(out_img.copy(), out_bboxes.astype(np.int32))
        plt.imshow(a.astype(np.int32))
        plt.show()
