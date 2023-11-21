import util.util as util
import models.networks as networks
import torch.utils.data as data
import importlib
from collections import OrderedDict
from options.test_options import TestOptions
# from data import create_dataset
# from models import create_model
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


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass

    def __len__(self):
        return 0


def find_dataset_using_name(dataset_name):
    # Given the option --dataset_mode [datasetname],
    # the file "data/datasetname_dataset.py"
    # will be imported.
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        print("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (
            dataset_filename, target_dataset_name))
        exit(0)

    return dataset


def create_dataset(opt):
    dataset = find_dataset_using_name(opt.dataset_mode)
    instance = dataset()
    instance.initialize(opt)
    print("dataset [%s] was created" % (instance.name()))
    return instance


class BaseModel():

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def train(self):
        print('switching to training mode')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()
    # make models eval mode during test time

    def eval(self):
        print('switching to testing mode')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.epoch = 0
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(
            self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        # if opt.resize_or_crop != 'scale_width':
        #     torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []

    # def set_input(self, input):
    #    self.input = input

    def set_input(self, input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.shadow_mask = (self.shadow_mask > 0.9).type(torch.float)*2-1
        # self.shadow_mask = (self.shadow_mask==1).type(torch.float)*2-1
        self.nim = self.input_img.shape[1]
        self.shadowfree_img = input['C'].to(self.device)
        self.shadow_mask_3d = (self.shadow_mask > 0).type(
            torch.float).expand(self.input_img.shape)
        # self.shadow_mask_3d_over = (self.shadow_mask_over>0).type(torch.float).expand(self.input_img.shape)

    def get_prediction(self, input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.shadow_mask = (self.shadow_mask > 0.9).type(torch.float)*2-1
        self.shadow_mask_3d = (self.shadow_mask > 0).type(
            torch.float).expand(self.input_img.shape)
        inputG = torch.cat([self.input_img, self.shadow_mask], 1)
        out = self.netG(inputG)
        return util.tensor2im(out)

    def forward(self):
        pass

    # load and print networks; create schedulers
    def setup(self, opt, parser=None):
        print(self.name)
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(
                optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain or opt.continue_train or opt.finetuning:
            print("LOADING %s" % (self.name))
            self.load_networks(opt.epoch)
        self.print_networks(opt.verbose)

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop

    def test(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self, loss=None):
        for scheduler in self.schedulers:
            if not loss:
                scheduler.step()
            else:
                scheduler.step(loss)

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        t = time.time()
        nim = self.shadow.shape[0]
        visual_ret = OrderedDict()
        all = []
        for i in range(0, min(nim-1, 5)):
            row = []
            for name in self.visual_names:
                if isinstance(name, str):
                    if hasattr(self, name):
                        im = util.tensor2im(
                            getattr(self, name).data[i:i+1, :, :, :])
                        row.append(im)
            row = tuple(row)
            all.append(np.hstack(row))
        all = tuple(all)

        allim = np.vstack(all)
        return OrderedDict([(self.opt.name, allim)])

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                if hasattr(self, 'loss_' + name):
                    errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # save models to the disk
    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(
                state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def load_networks(self, epoch, save_dir=None):
        print(epoch)
        if save_dir is None:
            save_dir = self.save_dir

        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(save_dir, load_filename)
                if self.opt.finetuning:

                    load_filename = '%s_net_%s.pth' % (
                        self.opt.finetuning_epoch, name)
                    load_path = os.path.join(
                        self.opt.finetuning_dir, load_filename)

                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(
                    load_path, map_location=str(self.device))
                #
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # need to copy keys here because we mutate in loop
                for key in list(state_dict.keys()):
                    self.__patch_instance_norm_state_dict(
                        state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' %
                      (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


def find_model_using_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (
            model_filename, target_model_name))
        exit(0)

    return model


def create_model(opt):
    model = find_model_using_name(opt.model)
    instance = model()
    instance.initialize(opt)
    print("model [%s] was created" % (instance.name()))
    return instance


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
