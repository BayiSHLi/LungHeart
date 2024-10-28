import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import tqdm
from tqdm import tqdm

# from torchmetrics.functional import f1_score
from sklearn.model_selection import KFold
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
                 start_epoch=0, early_stop=5, lr=1e-5, lr_head=1e-4, lr_minimum=1e-6,
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

        self.num_folds = 5
        self.kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        self.lr = lr
        self.lr_head = lr_head
        self.lr_minimum = lr_minimum

        self.metrics = metrics

        self.CEloss = nn.CrossEntropyLoss()
        self.BCEloss = nn.BCELoss()

        self.best_ckpt_path = None

        self.optimizer = torch.optim.Adam([
            {"params": self.model.audio_features.parameters()},
            {"params": self.model.classifier.parameters(), "lr": self.lr_head},],
            lr=self.lr, weight_decay=5e-5)
        if lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=self.lr_minimum,
                verbose=True)

    def train(self):
        wandb.init(
            project = 'ICBHI',
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

        best_model_wts_test = copy.deepcopy(self.model.state_dict())
        best_iou_test = 0.0
        best_epoch_test = 0

        print('-' * 10)
        print('TRAIN')
        print('-' * 10)

        for fold, (train_idx, val_idx) in enumerate(self.kf.split(self.train_set)):
            print(f'Fold {fold + 1}/{self.num_folds}')
            train_subset = Subset(self.train_set, train_idx)
            val_subset = Subset(self.train_set, val_idx)

            self.train_data = DataLoader(
                train_subset, self.batch_size, num_workers=self.num_workers,
                pin_memory=True, shuffle=True, drop_last=True,
            )
            self.val_data = DataLoader(
                val_subset, self.batch_size, num_workers=self.num_workers,
                pin_memory=True, shuffle=False,
            )
            is_earlystop = False

            for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
                if is_earlystop:
                    break
                # Train phase
                self.model.train()

                running_loss = 0.0

                pbar = tqdm(self.train_data)

                for i, batch in enumerate(pbar):
                    self.optimizer.zero_grad()
                    audio, label, text_inputs, attn_masks = batch

                    audio = audio.to(self.device)
                    gt_crackles, gt_wheezes, gt_diagnosis = label
                    gt_crackles, gt_wheezes, gt_diagnosis = gt_crackles.to(self.device), \
                                                            gt_wheezes.to(self.device), gt_diagnosis.to(self.device)
                    gt_crackles, gt_wheezes, gt_diagnosis = \
                        gt_crackles[:, None].type(torch.float32), gt_wheezes[:, None].type(torch.float32), \
                        gt_diagnosis[:, None].long().squeeze()
                    text_inputs = text_inputs.to(self.device)
                    attn_masks = attn_masks.to(self.device)

                    pred_crackles, pred_wheezes, pred_diagnosis = self.model(audio, text_inputs, attn_masks)

                    loss = self.BCEloss(pred_crackles, gt_crackles) + self.BCEloss(pred_wheezes, gt_wheezes) + \
                           0.2 * self.CEloss(pred_diagnosis, gt_diagnosis)

                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item() * self.batch_size

                    train_loss = running_loss / ((i + 1)*self.batch_size)

                    wandb.log({"train_loss": train_loss})
                    wandb.log({"loss": loss})

                    pbar.set_description(
                        f'Epoch {epoch}/{self.start_epoch + self.num_epochs - 1}, training loss {train_loss:.4f}')

                epoch_loss = running_loss / len(self.train_set)
                print('Epoch {}, Train Loss: {:.4f} '.format(epoch+1, epoch_loss))

                if self.scheduler:
                    self.scheduler.step()
                    wandb.log({"learning rate": self.optimizer.param_groups[0]['lr']})
                    wandb.log({"learning rate head": self.optimizer.param_groups[1]['lr']})

                # Validation phase
                if (epoch+1) % 1 == 0:
                    self.model.eval()
                    for metric in self.metrics:
                        metric.reset()
                    print('-' * 10)
                    print('VAL')
                    print('-' * 10)

                    val_loss = 0.0

                    pbar = tqdm(self.val_data)
                    vis_save = []
                    for i, batch in enumerate(pbar):
                        audio, label, text_inputs, attn_masks = batch

                        audio = audio.to(self.device)
                        gt_crackles, gt_wheezes, gt_diagnosis = label
                        gt_crackles, gt_wheezes, gt_diagnosis = gt_crackles.to(self.device), \
                                                                gt_wheezes.to(self.device), gt_diagnosis.to(self.device)
                        gt_crackles, gt_wheezes, gt_diagnosis = \
                            gt_crackles[:, None].type(torch.float32), gt_wheezes[:, None].type(torch.float32), \
                            gt_diagnosis[:, None].long().squeeze()
                        label = (gt_crackles, gt_wheezes, gt_diagnosis)
                        text_inputs = text_inputs.to(self.device)
                        attn_masks = attn_masks.to(self.device)

                        with torch.no_grad():
                            pred_crackles, pred_wheezes, pred_diagnosis = self.model(audio, text_inputs, attn_masks)
                            loss = self.BCEloss(pred_crackles, gt_crackles) + self.BCEloss(pred_wheezes, gt_wheezes) + \
                                   0.2 * self.CEloss(pred_diagnosis, gt_diagnosis)
                            self.metrics[0].update(pred_crackles, gt_crackles)
                            self.metrics[1].update(pred_wheezes, gt_wheezes)
                            self.metrics[2].update(pred_diagnosis, gt_diagnosis)
                            self.metrics[3].update(pred_crackles, pred_wheezes, gt_crackles, gt_wheezes)

                        # if self.vis:
                        #     vis_outs = vis_result(images, preds, masks)
                        #     vis_save = vis_save + vis_outs
                        val_loss += loss.item() * self.batch_size

                    epoch_loss_val = val_loss / len(val_subset)
                    print('Val Loss: {:.4f} '.format(epoch_loss_val))

                    acc = []
                    for metric in self.metrics:
                        acc.append(metric.compute())
                    print('Crackles Val Acc: {:.4f}. Wheezes Val Acc: {:.4f}. Diagnosis Val Acc: {:.4f}.'
                          .format(acc[0], acc[1], acc[2]))
                    print('	ICBHI Score: {:.4f}. Sensitivity: {:.4f}. Specificity: {:.4f}.'
                          .format(acc[3][0], acc[3][1], acc[3][2]))

                    wandb.log({"Crackles_Acc": acc[0]})
                    wandb.log({"Wheezes_Acc": acc[1]})
                    wandb.log({"Diagnosis_Acc": acc[2]})
                    wandb.log({"ICBHI Score": acc[3][0]})
                    wandb.log({"Sensitivity": acc[3][1]})
                    wandb.log({"Specificity": acc[3][2]})


                    if acc[3][0] > best_iou_test:
                        best_iou_test = acc[3][0]
                        best_epoch_test = epoch
                        best_model_wts_test = copy.deepcopy(self.model.state_dict())
                        torch.save(best_model_wts_test,
                                   f"{self.ckpt_path}/{self.model_name}_val_{best_epoch_test}_{best_iou_test:.4f}.pth")
                        print(f"Saved model at epoch {best_epoch_test} with IOU: {best_iou_test:.4f}")

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

        classifier_params = sum(p.numel() for p in self.model.classifier.parameters())
        fusion_params = sum(p.numel() for p in self.model.classifier.shared_layer.parameters())
        backbone_params = sum(p.numel() for p in self.model.audio_features.parameters())
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f'Head Params {classifier_params}')
        print(f'Fusion Params {fusion_params}')
        print(f'Encoder Params {backbone_params}')
        print(f'Total Params {total_params}')

        self.model.eval()
        for metric in self.metrics:
            metric.reset()

        self.test_data = DataLoader(
            self.test_set, batch_size=1, shuffle=False,
        )

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_data)):
                audio, label, text_inputs, attn_masks = batch

                audio = audio.to(self.device)
                gt_crackles, gt_wheezes, gt_diagnosis = label
                gt_crackles, gt_wheezes, gt_diagnosis = gt_crackles.to(self.device), \
                                                        gt_wheezes.to(self.device), gt_diagnosis.to(self.device)
                gt_crackles, gt_wheezes, gt_diagnosis = \
                    gt_crackles[:, None].type(torch.float32), gt_wheezes[:, None].type(torch.float32), \
                    gt_diagnosis[:, None].long()
                text_inputs = text_inputs.to(self.device)
                attn_masks = attn_masks.to(self.device)

                if i == 0:
                    flops_total = FlopCountAnalysis(self.model, (audio, text_inputs, attn_masks)).total()
                    print(f'Total Flops {flops_total}')

                pred_crackles, pred_wheezes, pred_diagnosis = self.model(audio, text_inputs, attn_masks)
                self.metrics[0].update(pred_crackles, gt_crackles)
                self.metrics[1].update(pred_wheezes, gt_wheezes)
                self.metrics[2].update(pred_diagnosis, gt_diagnosis)
                self.metrics[3].update(pred_crackles, pred_wheezes, gt_crackles, gt_wheezes)

            acc = []
            for metric in self.metrics:
                acc.append(metric.compute())
            print('Crackles Val Acc: {:.4f}. Wheezes Val Acc: {:.4f}. Diagnosis Val Acc: {:.4f}.'
                  .format(acc[0], acc[1], acc[2]))
            print('	ICBHI Score: {:.4f}. Sensitivity: {:.4f}. Specificity: {:.4f}.'
                  .format(acc[3][0], acc[3][1], acc[3][2]))




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


