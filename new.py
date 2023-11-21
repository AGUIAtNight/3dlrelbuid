from collections import OrderedDict
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
# from util.visualizer import save_images
from PIL import Image
import time
import math
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from skimage.color import rgb2lab
import numpy as np
import torch
import os
import cv2

import torchvision.transforms.functional as TF


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--loadSize", type=int)
parser.add_argument("--randomSize", action="store_true")
parser.add_argument("--name", type=str)
parser.add_argument("--dataroot", type=str)
parser.add_argument("--checkpoints_dir", type=str)
parser.add_argument("--fineSize", type=int)
parser.add_argument("--model", type=str)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--keep_ratio", action="store_true")
parser.add_argument("--phase", type=str)
parser.add_argument("--gpu_ids", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--lambda_L1", type=int)
parser.add_argument("--num_threads", type=int)
parser.add_argument("--dataset_mode", type=str)
parser.add_argument("--mask_train", type=str)
parser.add_argument("--optimizer", type=str)
parser.add_argument("--n", type=int)
parser.add_argument("--ks", type=int)
parser.add_argument("--lr_policy", type=str)
parser.add_argument("--lr_decay_iters", type=int)
parser.add_argument("--shadow_loss", type=float)
parser.add_argument("--tv_loss", type=float)
parser.add_argument("--grad_loss", type=float)
parser.add_argument("--pgrad_loss", type=float)
parser.add_argument("--save_epoch_freq", type=int)
parser.add_argument("--niter", type=int)
parser.add_argument("--niter_decay", type=int)

args = parser.parse_args()

opt = TestOptions().parse()
# 设置opt的相应属性为args的对应值
opt.loadSize = args.loadSize
opt.randomSize = args.randomSize
opt.name = args.name
opt.dataroot = args.dataroot
opt.checkpoints_dir = args.checkpoints_dir
opt.fineSize = args.fineSize
opt.model = args.model
opt.batch_size = args.batch_size
opt.keep_ratio = args.keep_ratio
opt.phase = args.phase
opt.gpu_ids = args.gpu_ids
opt.lr = args.lr
opt.lambda_L1 = args.lambda_L1
opt.num_threads = args.num_threads
opt.dataset_mode = args.dataset_mode
opt.mask_train = args.mask_train
opt.optimizer = args.optimizer
opt.n = args.n
opt.ks = args.ks
opt.lr_policy = args.lr_policy
opt.lr_decay_iters = args.lr_decay_iters
opt.shadow_loss = args.shadow_loss
opt.tv_loss = args.tv_loss
opt.grad_loss = args.grad_loss
opt.pgrad_loss = args.pgrad_loss
opt.save_epoch_freq = args.save_epoch_freq
opt.niter = args.niter
opt.niter_decay = args.niter_decay


def tensor2im(tensor):
    # 将张量转换为图像
    image = TF.to_pil_image(tensor)

    # 进行其他图像处理操作，如调整大小、格式转换等

    # 返回图像
    return image


def save_images(visuals, img_path, aspect_ratio=None, width=None):
    # 检查visuals是否包含需要保存的图像数据
    if 'image' not in visuals:
        raise ValueError("No 'image' key found in the visuals dictionary.")

    # 获取图像数据
    image = visuals['image']

    # 可选参数处理
    if aspect_ratio is None:
        aspect_ratio = 1.0
    if width is None:
        width = image.shape[1]

    # 计算图像的高度
    height = int(width / aspect_ratio)

    # 调整图像大小
    image = cv2.resize(image, (width, height))

    # 保存图像
    cv2.imwrite(img_path, image)


opt = TestOptions().parse()  # 解析测试参数
# 加载数据集

test_dataset = create_dataset(opt)
# 创建模型

model = create_model(opt)
# 设置模型为测试模式

model.eval()
# 遍历测试数据集进行预测

for i, data in enumerate(test_dataset):
    model.set_input(data)  # 设置模型输入
    model.forward()  # 进行前向传递

# 将输出图像保存到本地
visuals = model.get_current_visuals()
img_path = model.get_image_paths()
save_images(visuals, img_path, aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize)

# 可以将模型的输出结果保存到本地
output_image = tensor2im(model.final)  # 将模型输出转换为图像
pil_image = Image.fromarray(output_image)  # 创建PIL图像对象
pil_image.save('result_%d.png' % i)  # 保存结果图像
