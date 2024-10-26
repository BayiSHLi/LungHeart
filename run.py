import os
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from PIL import Image
from sklearn.metrics import *
from tqdm import tqdm
from utils.metrics import *
from utils.losses import LovaszLoss
from utils.vis import vis_result
from torchmetrics.functional import f1_score
from fvcore.nn import FlopCountAnalysis

import copy
import wandb


def calculate_iou(pred, masks, num_classes=19):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        masks_cls = (masks == cls)

        intersection = (pred_cls & masks_cls).sum().float().item()
        union = (pred_cls | masks_cls).sum().float().item()

        if union == 0:
            iou = float('nan')  # Avoid division by zero
        else:
            iou = intersection / union
        ious.append(iou)

    return ious


def calculate_mean_iou(pred, masks, num_classes=19):
    ious = calculate_iou(pred, masks, num_classes)
    valid_ious = [iou for iou in ious if not torch.isnan(torch.tensor(iou))]
    mean_iou = sum(valid_ious) / len(valid_ious)
    return mean_iou, ious


class Trainer():
    def __init__(self, model, train_set, test_set,
                 model_name, device, batch_size, num_workers, epochs,
                 exp_name, exp_path, ckpt_path, vis_path, save_path, num_exp,
                 start_epoch=0, early_stop=5, lr=1e-5, lr_head=1e-4,
                 lr_scheduler=True, save_threshold=0.7, metrics=None, vis=False):
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = epochs
        self.start_epoch = start_epoch
        self.early_stop = early_stop
        self.vis = vis

        self.exp_name = exp_name
        self.num_exp = num_exp
        self.exp_path = exp_path
        self.ckpt_path = ckpt_path
        self.vis_path = vis_path
        self.save_path = save_path

        self.save_threshold = save_threshold

        self.lr = lr
        self.lr_head = lr_head

        self.metrics = metrics

        self.CEloss = nn.CrossEntropyLoss()
        self.Lovaszloss = LovaszLoss()

        self.best_ckpt_path = None

        self.optimizer = torch.optim.Adam([
            {"params": self.model.backbone.parameters()},
            {"params": self.model.head.parameters(), "lr": self.lr_head},],
            lr=self.lr, weight_decay=5e-5)
        if lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=1e-6,
                verbose=True)

        self.train_data = DataLoader(
            self.train_set, self.batch_size, num_workers=self.num_workers,
            pin_memory=True, shuffle=True, drop_last=True,
        )

        self.val_data = DataLoader(
            self.val_set, self.batch_size, num_workers=self.num_workers,
            pin_memory=True, shuffle=True, drop_last=True,
        )

        self.test_data = DataLoader(
            self.test_set, batch_size=1, shuffle=False,
        )

    def train(self):
        wandb.init(
            project = 'celebA',
            name=self.exp_name + '_' + self.num_exp,
            config={
                "learning_rate": self.lr,
                "model name": self.model_name,
                "epochs": self.num_epochs,
            }
        )

        since = time.time()
        self.model.to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total trainable parameters: {total_params}')
        assert total_params < 2_000_000, f"Model has {total_params} parameters, which exceeds the limit."

        best_model_wts_test = copy.deepcopy(self.model.state_dict())
        best_iou_test = 0.0
        best_epoch_test = 0
        is_earlystop = False

        print('-' * 10)
        print('TRAIN')
        print('-' * 10)

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            if is_earlystop:
                break
            # Train phase
            self.model.train()

            running_loss = 0.0

            pbar = tqdm(self.train_data)

            for i, batch in enumerate(pbar):
                self.optimizer.zero_grad()
                images, masks = batch

                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.CEloss(outputs, masks) + self.Lovaszloss(outputs, masks)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)

                train_loss = running_loss / ((i + 1)*self.batch_size)

                wandb.log({"train_loss": train_loss})
                wandb.log({"loss": loss})

                pbar.set_description(
                    f'Epoch {epoch}/{self.start_epoch + self.num_epochs - 1}, training loss {train_loss:.4f}')

            epoch_loss = running_loss / len(self.train_set)
            print('Train Loss: {:.4f} '.format(epoch_loss))

            if self.scheduler:
                self.scheduler.step()
                wandb.log({"learning rate": self.optimizer.param_groups[0]['lr']})
                wandb.log({"learning rate head": self.optimizer.param_groups[1]['lr']})

            # Validation phase
            if (epoch+1) % 1 == 0:
                self.model.eval()
                # for metric in self.metrics:
                #     metric.reset_epoch_stats()
                self.metrics.reset(self.device)
                print('-' * 10)
                print('VAL')
                print('-' * 10)

                val_loss = 0.0
                val_pred = []
                val_label = []

                pbar = tqdm(self.val_data)
                vis_save = []
                for i, batch in enumerate(pbar):
                    images, masks = batch
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    with torch.no_grad():
                        outputs = self.model(images)
                        loss = self.CEloss(outputs, masks) + self.Lovaszloss(outputs, masks)
                        self.metrics.update(outputs, masks)
                    preds = torch.argmax(outputs, dim=1)
                    if self.vis:
                        vis_outs = vis_result(images, preds, masks)
                        vis_save = vis_save + vis_outs
                    val_loss += loss.item() * images.size(0)
                    val_pred.extend(preds.view(-1).tolist())
                    val_label.extend(masks.view(-1).tolist())


                epoch_loss_val = val_loss / len(self.val_set)
                print('Val Loss: {:.4f} '.format(epoch_loss_val))
                # for metric in self.metrics:
                #     metric.update(val_label, val_pred)
                miou = self.metrics.compute()
                print('Val mIoU: {:.4f} '.format(miou))
                wandb.log({"val_miou": miou})


                if miou > best_iou_test:
                    best_iou_test = miou
                    best_epoch_test = epoch
                    best_model_wts_test = copy.deepcopy(self.model.state_dict())
                    torch.save(best_model_wts_test,
                               f"{self.ckpt_path}/{self.model_name}_val_{best_epoch_test}_{best_iou_test:.4f}.pth")
                    print(f"Saved model at epoch {best_epoch_test} with IOU: {best_iou_test:.4f}")
                    if self.vis:
                        for i in range(len(vis_save)):
                            cv2.imwrite(os.path.join(self.vis_path,'val', f'vis_mask_{i}.png'), vis_save[i])
                else:
                    if epoch - best_epoch_test >= self.early_stop - 1:
                        is_earlystop = True
                        print("Early stopping at epoch " + str(epoch))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print("Best model on val: epoch " + str(best_epoch_test) + " with IOU: " + str(best_iou_test))

        self.best_ckpt_path = f"{self.ckpt_path}/{self.model_name}_val_{best_epoch_test}_{best_iou_test:.4f}.pth"

        return self.best_ckpt_path


    def test(self, best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location='cpu')
        self.model.load_state_dict(best_ckpt)
        self.model.to(self.device)

        self.model.eval()

        with torch.no_grad():
            for i, (images, file_names) in enumerate(self.test_data):
                images = images.to(self.device)
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                vis_out = vis_result(images, preds)

                for j in range(preds.size(0)):
                    save_vis_path = os.path.join(self.vis_path, 'test', os.path.splitext(file_names[j])[0] + '.png')
                    cv2.imwrite(save_vis_path, vis_out[j])

                    save_file_path = os.path.join(self.save_path, os.path.splitext(file_names[j])[0] + '.png')
                    mask_np = preds[j].cpu().numpy().astype(np.uint8)
                    mask_image = Image.fromarray(mask_np)
                    mask_image.save(save_file_path)
                    print(f'Saved mask to {save_file_path}')

    def f_measure(self, best_ckpt_path):
        # f1_score = F1(num_classes=18, average=None).to(self.device)  # No averaging for individual F1
        # confusion_matrix = ConfusionMatrix(num_classes=18).to(self.device)

        best_ckpt = torch.load(best_ckpt_path, map_location='cpu')
        self.model.load_state_dict(best_ckpt)
        self.model.to(self.device)

        rape_params = sum(p.numel() for p in self.model.backbone.patch_embed1.parameters())
        fgde_params = sum(p.numel() for p in self.model.head.fine_grained.parameters())
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f'RAPE Params {rape_params}')
        print(f'FGDE Params {fgde_params}')
        print(f'Total Params {total_params}')

        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i, (images, masks) in enumerate(self.val_data):
                images = images.to(self.device)
                masks = masks.to(self.device)
                if i == 0:
                    flops_rape = FlopCountAnalysis(self.model.backbone.patch_embed1, images).total()
                    flops_total = FlopCountAnalysis(self.model, images).total()
                    print(f'RAPE Flops {flops_rape}')
                    print(f'Total Flops {flops_total}')
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds)
                all_labels.append(masks)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        f1_per_class = f1_score(all_preds, all_labels, task="multiclass", num_classes=19, average=None)
        f1_average = f1_score(all_preds, all_labels, task="multiclass", num_classes=19)

        print("F1 Score for each class:")
        print(f1_per_class)
        print("Average F1 Score")
        print(f1_average)

        mean_iou, ious = calculate_mean_iou(all_preds, all_labels)
        print(f"Mean IoU: {mean_iou}")
        print(f"IoU per class: {ious}")

    def save_checkpoint(self, epoch, best_iou, best_epoch):
        state = {
            'epoch': epoch,
            'best_iou': best_iou,
            'best_epoch': best_epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None
        }
        ckpt_path = f"{self.ckpt_path}/{self.model_name}_epoch_{epoch}_iou_{best_iou:.4f}.pth"
        torch.save(state, ckpt_path)
        self.best_ckpt_path = ckpt_path
        print(f"Checkpoint saved to {ckpt_path}")

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.scheduler and checkpoint['scheduler_state']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        best_iou = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']
        print(f"Checkpoint loaded from {ckpt_path}, starting from epoch {self.start_epoch}")
        return best_iou, best_epoch

    def resume(self, exp_name, num_exp):
        ckpt_path = f"{self.ckpt_path}/{self.model_name}_exp_{exp_name}_{num_exp}.pth"
        best_iou, best_epoch = self.load_checkpoint(ckpt_path)
        self.train()


#tensor([0.9350, 0.9490, 0.9270, 0.8744, 0.4691, 0.4750, 0.4725, 0.4243, 0.5018,
#         0.4047, 0.8861, 0.8560, 0.8744, 0.9300, 0.7155, 0.5753, 0.0815, 0.8644,
#         0.7433], device='cuda:1')
