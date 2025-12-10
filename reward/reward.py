import evaluate
import torch
from torch import nn
from typing import List
import json
from sklearn.metrics import f1_score, jaccard_score

# preds = ["这里写你的生成文本..."]
# refs = ["我色出任何戳穿哼哧哼哧不知..."]
# bertscore = evaluate.load("bertscore")
# results = bertscore.compute(predictions=preds, references=refs, model_type="bert-base-multilingual-cased")

# print(results)


class RewardClass(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BERTScoreReward(RewardClass):
    def __init__(self, model_type="facebook/bart-large", lang="en", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bertscore = evaluate.load("bertscore")
        self.model_type = model_type

    def forward(self, predictions, references):
        results = self.bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            model_type=self.model_type,
        )
        return results


class BoundaryReward(RewardClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def boundary_to_segment(boundary_seq):
        # 1. 累积求和生成ID (0, 1, 1, 2, 2...)
        # 遇到边界1时ID会增加，正好符合分段逻辑
        seg_ids = torch.cumsum(torch.tensor(boundary_seq), dim=0)
        # 2. 生成掩码：保留中间段，掐头去尾
        # valid: ID必须大于0 (去掉第一个1之前) 且 小于总边界数 (去掉最后一个1之后)
        total_boundaries = seg_ids[-1]
        mask = (seg_ids > 0) & (seg_ids < total_boundaries)
        # 3. 应用掩码 (False区域置0)
        return seg_ids * mask

    def dice_segment(self, predictions, references):
        preds = predictions.numpy()
        refs = references.numpy()
        dice = f1_score(refs, preds, average="macro")
        return dice

    def iou_segment(self, predictions, references):
        preds = predictions.numpy()
        refs = references.numpy()
        iou = jaccard_score(refs, preds, average="macro")
        return iou

    def forward(self, predictions, references):
        predictions_seq = self.boundary_to_segment(predictions)
        references_seq = self.boundary_to_segment(references)
        dice = self.dice_segment(predictions_seq, references_seq)
        iou = self.iou_segment(predictions_seq, references_seq)
        return dice


class SegmentExpReward(RewardClass):
    def __init__(self, alpha=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def forward(self, predictions, references):
        predictions = torch.tensor(predictions, dtype=torch.float32)
        references = torch.tensor(references, dtype=torch.float32)
        rewards = torch.exp(-self.alpha * torch.abs(predictions - references))
        return rewards


class SegmentQuadraticReward(RewardClass):
    def __init__(self, alpha=0.01, base=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.base = base

    def forward(self, predictions, references):
        predictions = torch.tensor(predictions, dtype=torch.float32)
        references = torch.tensor(references, dtype=torch.float32)
        rewards = self.base - self.alpha * (references - predictions) ** 2
        return rewards


class JSONFormatReward(RewardClass):
    def __init__(self, pos_reward=0.5, neg_reward=-1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_reward = pos_reward
        self.neg_reward = neg_reward

    def forward(self, predictions: List[str]):
        rewards = []

        for pred in predictions:
            try:
                jsonstr = pred.lstrip('"""json').rstrip('"""')
                json.loads(jsonstr)
                reward = self.pos_reward
            except Exception:
                reward = self.neg_reward

            rewards.append(reward)

        return rewards


if __name__ == "__main__":
    # input_list = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    # 期望看到:   0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0

    # 转换
    # output_seq = boundary_to_segment(input_list)

    # print("原始输入:", input_list)
    # print("转换结果:", output_seq.tolist())

    # 假设这是模型预测结果 (你提供的序列)
    predss = [9, 9, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9]
    # 假设这是真实标签 (Ground Truth) - 我故意改了几个数来模拟误差
    target = [9, 9, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9]

    # --- 1. 计算 Dice (即 F1 Score) ---
    # average='macro': 计算每个类别的指标然后求平均 (最常用)
    # average='weighted': 按类别出现频率加权
    dice = f1_score(target, predss, average="macro")

    # --- 2. 计算 IoU (即 Jaccard Score) ---
    iou = jaccard_score(target, predss, average="macro")

    # --- 3. 查看每个类别的详细 IoU (不求平均) ---
    iou_per_class = jaccard_score(target, predss, average=None)

    print(f"Mean Dice (F1): {dice:.4f}")
    print(f"Mean IoU (Jaccard): {iou:.4f}")
    print(f"各类别 IoU: {iou_per_class}")
