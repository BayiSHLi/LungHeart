import argparse
import os
import warnings
import numpy as np
import torch
import yaml
import albumentations as A
# from albumentations.pytorch import ToTensorV2
import random

import utils
import model
from run import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='ICBHI-1009')
    parser.add_argument('--model_name', default='ICBHI')
    parser.add_argument('--model_cfg', default='./config/ICBHIMode.yaml')
    parser.add_argument('--data_root', default= '/dataset/ICBHI/')
    parser.add_argument('--sample_rate', default= 16000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--batch_size', type = int, default=40)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--lr_head', type=float, default=0.001)
    parser.add_argument('--lr_scheduler', type=bool, default=True)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--save_path', default= './experiments/')
    parser.add_argument('--distributed', action='store_true',
                        help='No Distributed training for debug')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--ckpt_path', default='./best_ckpt.pth')
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

    cfg['metrics'] = MeanIoU(num_classes=19)

    return cfg

def init_model(args):
    with open(args.model_cfg, 'r', encoding='utf-8') as f:
        model_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    model = AudioMambaModel(**model_cfg)
    if args.pretrained:
        model.load_state_dict(torch.load(args.ckpt_path),
                              strict=True)
        print('load ckpt')
    return model

def init_dataset(args):
    root = args.data_root
    transform = AudioAugmentor(args.sample_rate)

    data_cfg = {"data_root": args.data_root,
               "mode": args.mode,
                "sample_rate": args.sample_rate,
                "desired_length": args.desired_length,
                "pad_types": args.pad_types,
                "fade_samples_ratio": args.fade_samples_ratio,
                "transform":transform}

    train_set = ICBHIDataset(**data_cfg)

    data_cfg["transform"] = None
    test_set = ICBHIDataset(**data_cfg)

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
        trainer.f_measure(args.ckpt_path)
        trainer.model.backbone.patch_embed1.vis_attn_map()
    else:
        best_ckpt_path = trainer.train()
        result = trainer.test(best_ckpt_path)



if __name__ == '__main__':
        main()
