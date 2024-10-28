from transformers import ClapModel, ClapAudioModelWithProjection
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


class MultiTaskModel(nn.Module):
    def __init__(self, dim, out_dim):
        super(MultiTaskModel, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        # 每个任务对应一个输出头
        self.crackles_head = nn.Linear(out_dim, 1)
        self.wheezes_head = nn.Linear(out_dim, 1)
        self.diagnosis_head = nn.Linear(out_dim, 8)  # 8个类别

    def forward(self, x):
        # flops_fusion = FlopCountAnalysis(self.shared_layer, x).total()
        # print(f'Total Flops {flops_fusion}')
        x = self.shared_layer(x)
        # flops_c = FlopCountAnalysis(self.crackles_head, x).total()
        # flops_w = FlopCountAnalysis(self.wheezes_head, x).total()
        # flops_d = FlopCountAnalysis(self.diagnosis_head, x).total()
        # print(f'Total Flops {flops_c + flops_w + flops_d}')
        crackles_out = torch.sigmoid(self.crackles_head(x))
        wheezes_out = torch.sigmoid(self.wheezes_head(x))
        diagnosis_out = self.diagnosis_head(x)  # 不需要激活函数, 使用CrossEntropyLoss
        return crackles_out, wheezes_out, diagnosis_out


# from transformers import set_seed
# set_seed(1)
class PretrainedCLAPWithProjection(nn.Module):
    def __init__(self, pretrained_name, final_feat_dim):
        super().__init__()

        self.pretrained = pretrained_name
        self.audio_features = ClapAudioModelWithProjection.from_pretrained(pretrained_name)
        self.final_feat_dim = final_feat_dim

    def forward(self, x, args=None, alpha=None, training=False):
        x = self.audio_features(x)
        return x.audio_embeds


class PretrainedCLAP(nn.Module):
    def __init__(self, pretrained_name, final_feat_dim):
        super().__init__()

        self.pretrained = pretrained_name
        self.audio_features = ClapModel.from_pretrained(pretrained_name)
        self.final_feat_dim = final_feat_dim

        self.classifier = MultiTaskModel(dim=self.audio_features.projection_dim*2, out_dim=final_feat_dim)

    def forward(self, audio, text_inputs, attn_masks):
        # flops_encoder = FlopCountAnalysis(self.audio_features, (text_inputs, attn_masks, audio)).total()
        # print(f'Encoder Flops {flops_encoder}')

        x = self.audio_features(input_ids=text_inputs, attention_mask=attn_masks, input_features=audio)
        x = torch.concat([x.text_embeds, x.audio_embeds], dim=1)

        pred_crackles, pred_wheezes, pred_diagnosis = self.classifier(x)
        return pred_crackles, pred_wheezes, pred_diagnosis
