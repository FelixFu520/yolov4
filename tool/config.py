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
import torch
from tool.torch_utils import convert2cpu


def parse_cfg(cfgfile):
    """转cfgfile(yolov4.cfg) 为 list(dict)格式，方便以后读取
    :param cfgfile: yolov4的配置文件
    :return: list（dict）
    """
    blocks = []
    fp = open(cfgfile, 'r')
    block = None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key, value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks


def print_cfg(blocks):
    """通过blocks 打印 网络结构
    :param blocks:
    :return:
    """
    print('layer     filters    size              input                output');
    prev_width = 416
    prev_height = 416
    prev_filters = 3
    out_filters = []
    out_widths = []
    out_heights = []
    ind = -2
    for block in blocks:
        ind = ind + 1
        if block['type'] == 'net':
            prev_width = int(block['width'])
            prev_height = int(block['height'])
            continue
        elif block['type'] == 'convolutional':
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = (kernel_size - 1) // 2 if is_pad else 0
            width = (prev_width + 2 * pad - kernel_size) // stride + 1
            height = (prev_height + 2 * pad - kernel_size) // stride + 1
            print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                ind, 'conv', filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width,
                height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            width = prev_width // stride
            height = prev_height // stride
            print('%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                ind, 'max', pool_size, pool_size, stride, prev_width, prev_height, prev_filters, width, height,
                filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'avgpool':
            width = 1
            height = 1
            print('%5d %-6s                   %3d x %3d x%4d   ->  %3d' % (
                ind, 'avg', prev_width, prev_height, prev_filters, prev_filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'softmax':
            print('%5d %-6s                                    ->  %3d' % (ind, 'softmax', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'cost':
            print('%5d %-6s                                     ->  %3d' % (ind, 'cost', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'reorg':
            stride = int(block['stride'])
            filters = stride * stride * prev_filters
            width = prev_width // stride
            height = prev_height // stride
            print('%5d %-6s             / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                ind, 'reorg', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            filters = prev_filters
            width = prev_width * stride
            height = prev_height * stride
            print('%5d %-6s           * %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                ind, 'upsample', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
            if len(layers) == 1:
                print('%5d %-6s %d' % (ind, 'route', layers[0]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                print('%5d %-6s %d %d' % (ind, 'route', layers[0], layers[1]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert (prev_width == out_widths[layers[1]])
                assert (prev_height == out_heights[layers[1]])
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            elif len(layers) == 4:
                print('%5d %-6s %d %d %d %d' % (ind, 'route', layers[0], layers[1], layers[2], layers[3]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert (prev_width == out_widths[layers[1]] == out_widths[layers[2]] == out_widths[layers[3]])
                assert (prev_height == out_heights[layers[1]] == out_heights[layers[2]] == out_heights[layers[3]])
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]] + out_filters[layers[2]] + out_filters[
                    layers[3]]
            else:
                print("route error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                        sys._getframe().f_code.co_name, sys._getframe().f_lineno))

            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] in ['region', 'yolo']:
            print('%5d %-6s' % (ind, 'detection'))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'shortcut':
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id + ind
            print('%5d %-6s %d' % (ind, 'shortcut', from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'connected':
            filters = int(block['output'])
            print('%5d %-6s                            %d  ->  %3d' % (ind, 'connected', prev_filters, filters))
            prev_filters = filters
            out_widths.append(1)
            out_heights.append(1)
            out_filters.append(prev_filters)
        else:
            print('unknown type %s' % (block['type']))


"""
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   608 x 608 x   3   ->   608 x 608 x  32
    1 conv     64  3 x 3 / 2   608 x 608 x  32   ->   304 x 304 x  64
    2 conv     64  1 x 1 / 1   304 x 304 x  64   ->   304 x 304 x  64
    3 route  1
    4 conv     64  1 x 1 / 1   304 x 304 x  64   ->   304 x 304 x  64
    5 conv     32  1 x 1 / 1   304 x 304 x  64   ->   304 x 304 x  32
    6 conv     64  3 x 3 / 1   304 x 304 x  32   ->   304 x 304 x  64
    7 shortcut 4
    8 conv     64  1 x 1 / 1   304 x 304 x  64   ->   304 x 304 x  64
    9 route  8 2
   10 conv     64  1 x 1 / 1   304 x 304 x 128   ->   304 x 304 x  64
   11 conv    128  3 x 3 / 2   304 x 304 x  64   ->   152 x 152 x 128
   12 conv     64  1 x 1 / 1   152 x 152 x 128   ->   152 x 152 x  64
   13 route  11
   14 conv     64  1 x 1 / 1   152 x 152 x 128   ->   152 x 152 x  64
   15 conv     64  1 x 1 / 1   152 x 152 x  64   ->   152 x 152 x  64
   16 conv     64  3 x 3 / 1   152 x 152 x  64   ->   152 x 152 x  64
   17 shortcut 14
   18 conv     64  1 x 1 / 1   152 x 152 x  64   ->   152 x 152 x  64
   19 conv     64  3 x 3 / 1   152 x 152 x  64   ->   152 x 152 x  64
   20 shortcut 17
   21 conv     64  1 x 1 / 1   152 x 152 x  64   ->   152 x 152 x  64
   22 route  21 12
   23 conv    128  1 x 1 / 1   152 x 152 x 128   ->   152 x 152 x 128
   24 conv    256  3 x 3 / 2   152 x 152 x 128   ->    76 x  76 x 256
   25 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128
   26 route  24
   27 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128
   28 conv    128  1 x 1 / 1    76 x  76 x 128   ->    76 x  76 x 128
   29 conv    128  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 128
   30 shortcut 27
   31 conv    128  1 x 1 / 1    76 x  76 x 128   ->    76 x  76 x 128
   32 conv    128  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 128
   33 shortcut 30
   34 conv    128  1 x 1 / 1    76 x  76 x 128   ->    76 x  76 x 128
   35 conv    128  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 128
   36 shortcut 33
   37 conv    128  1 x 1 / 1    76 x  76 x 128   ->    76 x  76 x 128
   38 conv    128  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 128
   39 shortcut 36
   40 conv    128  1 x 1 / 1    76 x  76 x 128   ->    76 x  76 x 128
   41 conv    128  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 128
   42 shortcut 39
   43 conv    128  1 x 1 / 1    76 x  76 x 128   ->    76 x  76 x 128
   44 conv    128  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 128
   45 shortcut 42
   46 conv    128  1 x 1 / 1    76 x  76 x 128   ->    76 x  76 x 128
   47 conv    128  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 128
   48 shortcut 45
   49 conv    128  1 x 1 / 1    76 x  76 x 128   ->    76 x  76 x 128
   50 conv    128  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 128
   51 shortcut 48
   52 conv    128  1 x 1 / 1    76 x  76 x 128   ->    76 x  76 x 128
   53 route  52 25
   54 conv    256  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 256
   55 conv    512  3 x 3 / 2    76 x  76 x 256   ->    38 x  38 x 512
   56 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
   57 route  55
   58 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
   59 conv    256  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 256
   60 conv    256  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 256
   61 shortcut 58
   62 conv    256  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 256
   63 conv    256  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 256
   64 shortcut 61
   65 conv    256  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 256
   66 conv    256  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 256
   67 shortcut 64
   68 conv    256  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 256
   69 conv    256  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 256
   70 shortcut 67
   71 conv    256  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 256
   72 conv    256  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 256
   73 shortcut 70
   74 conv    256  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 256
   75 conv    256  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 256
   76 shortcut 73
   77 conv    256  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 256
   78 conv    256  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 256
   79 shortcut 76
   80 conv    256  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 256
   81 conv    256  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 256
   82 shortcut 79
   83 conv    256  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 256
   84 route  83 56
   85 conv    512  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 512
   86 conv   1024  3 x 3 / 2    38 x  38 x 512   ->    19 x  19 x1024
   87 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512
   88 route  86
   89 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512
   90 conv    512  1 x 1 / 1    19 x  19 x 512   ->    19 x  19 x 512
   91 conv    512  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x 512
   92 shortcut 89
   93 conv    512  1 x 1 / 1    19 x  19 x 512   ->    19 x  19 x 512
   94 conv    512  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x 512
   95 shortcut 92
   96 conv    512  1 x 1 / 1    19 x  19 x 512   ->    19 x  19 x 512
   97 conv    512  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x 512
   98 shortcut 95
   99 conv    512  1 x 1 / 1    19 x  19 x 512   ->    19 x  19 x 512
  100 conv    512  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x 512
  101 shortcut 98
  102 conv    512  1 x 1 / 1    19 x  19 x 512   ->    19 x  19 x 512
  103 route  102 87
  104 conv   1024  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x1024
  105 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512
  106 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024
  107 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512
  108 max          5 x 5 / 1    19 x  19 x 512   ->    19 x  19 x 512
  109 route  107
  110 max          9 x 9 / 1    19 x  19 x 512   ->    19 x  19 x 512
  111 route  107
  112 max          13 x 13 / 1    19 x  19 x 512   ->    19 x  19 x 512
  113 route  112 110 108 107
  114 conv    512  1 x 1 / 1    19 x  19 x2048   ->    19 x  19 x 512
  115 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024
  116 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512
  117 conv    256  1 x 1 / 1    19 x  19 x 512   ->    19 x  19 x 256
  118 upsample           * 2    19 x  19 x 256   ->    38 x  38 x 256
  119 route  85
  120 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
  121 route  120 118
  122 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
  123 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512
  124 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
  125 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512
  126 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
  127 conv    128  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 128
  128 upsample           * 2    38 x  38 x 128   ->    76 x  76 x 128
  129 route  54
  130 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128
  131 route  130 128
  132 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128
  133 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256
  134 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128
  135 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256
  136 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128
  137 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256
  138 conv    255  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 255
  139 detection
  140 route  136
  141 conv    256  3 x 3 / 2    76 x  76 x 128   ->    38 x  38 x 256
  142 route  141 126
  143 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
  144 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512
  145 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
  146 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512
  147 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
  148 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512
  149 conv    255  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 255
  150 detection
  151 route  147
  152 conv    512  3 x 3 / 2    38 x  38 x 256   ->    19 x  19 x 512
  153 route  152 116
  154 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512
  155 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024
  156 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512
  157 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024
  158 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512
  159 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024
  160 conv    255  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 255
  161 detection
"""


def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).reshape(conv_model.weight.data.shape));
    start = start + num_w
    return start


def save_conv(fp, conv_model):
    if conv_model.bias.is_cuda:
        convert2cpu(conv_model.bias.data).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        conv_model.bias.data.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).reshape(conv_model.weight.data.shape));
    start = start + num_w
    return start


def save_conv_bn(fp, conv_model, bn_model):
    if bn_model.bias.is_cuda:
        convert2cpu(bn_model.bias.data).numpy().tofile(fp)
        convert2cpu(bn_model.weight.data).numpy().tofile(fp)
        convert2cpu(bn_model.running_mean).numpy().tofile(fp)
        convert2cpu(bn_model.running_var).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        bn_model.bias.data.numpy().tofile(fp)
        bn_model.weight.data.numpy().tofile(fp)
        bn_model.running_mean.numpy().tofile(fp)
        bn_model.running_var.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)


def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]));
    start = start + num_w
    return start


def save_fc(fp, fc_model):
    fc_model.bias.data.numpy().tofile(fp)
    fc_model.weight.data.numpy().tofile(fp)


if __name__ == '__main__':
    import sys

    blocks = parse_cfg('../cfg/yolov4.cfg')
    if len(sys.argv) == 2:
        blocks = parse_cfg(sys.argv[1])
    print_cfg(blocks)

