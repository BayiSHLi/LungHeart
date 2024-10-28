import argparse
import os
import warnings
import numpy as np
import torch
import yaml
import albumentations as A
import random
from model.clap import PretrainedCLAP

from utils import *
from run import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='ICBHI-1009')
    parser.add_argument('--model_name', default='ICBHI')
    parser.add_argument('--model_cfg', default='./config/ICBHIMode.yaml')
    parser.add_argument('--data_root', default= '/dataset/ICBHI/')
    parser.add_argument('--sample_rate', default= 16000)
    parser.add_argument('--desired_length', default= 8)
    parser.add_argument('--pad_types', default= 'zero')
    parser.add_argument('--fade_samples_ratio', default= 16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--batch_size', type = int, default=80)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--lr_minimum', type=float, default=0.000001)
    parser.add_argument('--lr_head', type=float, default=0.001)
    parser.add_argument('--lr_scheduler', type=bool, default=True)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--save_path', default= './experiments/')
    parser.add_argument('--distributed', action='store_true',
                        help='No Distributed training for debug')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--ckpt_path', default='./experiments/ICBHI-1009/066/ckpt/ICBHI_val_2_0.7748.pth')
    args = parser.parse_args()

    print(args)
    return args


def init_cfg(args):
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    cfg = {
        "exp_name": args.exp_name,
        "model_name": args.model_name,
        "device": device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "start_epoch": args.start_epoch,
        "early_stop": args.early_stop,
        "lr": args.lr,
        "lr_head": args.lr_head,
        "lr_minimum": args.lr_minimum,
        "lr_scheduler": args.lr_scheduler,
        "exp_path": os.path.join(args.save_path, args.exp_name),
        "vis": args.vis,
    }

    if not os.path.exists(cfg['exp_path']):
        os.makedirs(cfg['exp_path'])
    existing_folders = [f for f in os.listdir(cfg['exp_path'])
                        if os.path.isdir(os.path.join(cfg['exp_path'], f))]
    numbered_folders = sorted([int(f) for f in existing_folders if f.isdigit()])
    if len(numbered_folders) == 0:
        cfg['num_exp'] = "001"
    else:
        new_folder_number = numbered_folders[-1] + 1
        cfg['num_exp']  = f"{new_folder_number:03d}"
    cfg['exp_path'] = os.path.join(cfg['exp_path'], cfg['num_exp'])
    os.makedirs(cfg['exp_path'] , exist_ok=True)

    cfg['ckpt_path'] = os.path.join(cfg['exp_path'], 'ckpt')
    cfg['vis_path'] = os.path.join(cfg['exp_path'], 'vis')
    cfg['save_path']= os.path.join(cfg['exp_path'], 'save')

    os.makedirs(cfg['ckpt_path'] , exist_ok=True)
    os.makedirs(os.path.join(cfg['vis_path'], 'val'), exist_ok=True)
    os.makedirs(os.path.join(cfg['vis_path'], 'test'), exist_ok=True)
    os.makedirs(cfg['save_path'] , exist_ok=True)

    cfg['metrics'] = [BinaryAccuracy(), BinaryAccuracy(), MultiClassAccuracy(num_classes=8), NormalAbnormalAccuracy()]

    return cfg

def init_model(args):
    with open(args.model_cfg, 'r', encoding='utf-8') as f:
        model_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    model = PretrainedCLAP(**model_cfg)
    if args.pretrained:
        model.load_state_dict(torch.load(args.ckpt_path),
                              strict=True)
        print('load ckpt')
    return model

def init_dataset(args):
    root = args.data_root

    transform = A.Compose([
        # A.HorizontalFlip(p=0.5),  # 50% 概率水平翻转
        # A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.3),  # 随机平移、缩放和旋转
        A.RandomGridShuffle(grid=(3,3), p=0.3),
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # 颜色抖动
        A.RandomBrightnessContrast(p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # 随机高斯模糊
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # 添加随机噪声
        # ToTensorV2()  # 转换为PyTorch张量
    ])

    data_cfg = {"data_root": args.data_root,
                "sample_rate": args.sample_rate,
                "desired_length": args.desired_length,
                "pad_types": args.pad_types,
                "fade_samples_ratio": args.fade_samples_ratio,
                "transform":transform}

    train_set = ICBHIDataset(**data_cfg, mode='train')

    data_cfg["transform"] = None
    test_set = ICBHIDataset(**data_cfg, mode='test')

    return train_set, test_set

def init_exp(args):
    cfg = init_cfg(args)
    model = init_model(args)
    train_set, test_set = init_dataset(args)

    return cfg, model, train_set, test_set

def seed_torch(seed=1029):
    # torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    args = parse_args()

    seed_torch(seed=args.seed)

    cfg, model, train_set, test_set = init_exp(args)

    trainer = Trainer(model=model, train_set=train_set, test_set=test_set, **cfg)
    if args.test:
        trainer.test(args.ckpt_path)
        # trainer.f_measure(args.ckpt_path)
        # trainer.model.backbone.patch_embed1.vis_attn_map()
    else:
        best_ckpt_path = trainer.train()
        result = trainer.test(best_ckpt_path)



if __name__ == '__main__':
        main()

