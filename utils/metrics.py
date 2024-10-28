import torch

class BinaryAccuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, outputs, targets):
        """
        outputs: 模型的二分类输出 [batch_size]，用 0.5 作为阈值判断预测
        targets: 实际标签 [batch_size]
        """
        preds = (outputs > 0.5).int()  # 将概率转为二进制分类结果
        self.correct += (preds == targets).sum().item()
        self.total += targets.size(0)

    def compute(self):
        # 返回二分类任务的准确率
        return self.correct / self.total if self.total > 0 else 0.0


# 多分类任务的准确率计算
class MultiClassAccuracy:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, outputs, targets):
        """
        outputs: 多分类模型的输出，shape为 [batch_size, num_classes]
        targets: 多分类标签，shape为 [batch_size]
        """
        preds = torch.argmax(outputs, dim=1)  # 获取每个样本的预测类别
        self.correct += (preds == targets).sum().item()
        self.total += targets.size(0)

    def compute(self):
        # 返回多分类任务的准确率
        return self.correct / self.total if self.total > 0 else 0.0


class NormalAbnormalAccuracy:
    def __init__(self):
        self.normal_correct = 0  # 正确检测为正常的数量
        self.abnormal_correct = 0  # 正确检测为异常的数量
        self.normal_total = 0  # 实际正常的总数
        self.abnormal_total = 0  # 实际异常的总数

    def reset(self):
        self.normal_correct = 0
        self.abnormal_correct = 0
        self.normal_total = 0
        self.abnormal_total = 0

    def update(self, crackles_pred, wheezes_pred, crackles_labels, wheezes_labels):
        """
        crackles_pred: Crackles 的预测, tensor 形状为 [batch_size]
        wheezes_pred: Wheezes 的预测, tensor 形状为 [batch_size]
        crackles_labels: Crackles 的真实标签, tensor 形状为 [batch_size]
        wheezes_labels: Wheezes 的真实标签, tensor 形状为 [batch_size]
        """
        # 将预测和标签转换为“正常”与“异常”分类
        pred_abnormal = ((crackles_pred > 0.5) | (wheezes_pred > 0.5)).int()  # 预测结果是否异常
        label_abnormal = ((crackles_labels == 1) | (wheezes_labels == 1)).int()  # 实际标签是否异常

        # 计算总数
        self.normal_total += (label_abnormal == 0).sum().item()
        self.abnormal_total += (label_abnormal == 1).sum().item()

        # 计算正确预测数
        self.normal_correct += ((pred_abnormal == 0) & (label_abnormal == 0)).sum().item()
        self.abnormal_correct += ((pred_abnormal == 1) & (label_abnormal == 1)).sum().item()

    def compute(self):
        # 计算“正常”的准确率和“异常”的准确率
        normal_accuracy = self.normal_correct / self.normal_total if self.normal_total > 0 else 0.0
        abnormal_accuracy = self.abnormal_correct / self.abnormal_total if self.abnormal_total > 0 else 0.0
        ICBHI_score = (abnormal_accuracy + normal_accuracy) / 2
        return ICBHI_score, normal_accuracy, abnormal_accuracy
