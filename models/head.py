import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class KHead(nn.Module):
    """
    段数预测头：对 0..K_max 做分类。
    - 从 video_mask 指定的视频 token 上做 masked mean pooling 得到全局表示
    - 小 MLP 输出 logits
    """

    def __init__(self, embed_dim: int, k_max: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.k_max = k_max
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, k_max + 1),  # ignore 0
        )

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, D]
        mask: [B, S] bool
        return: [B, D]
        """
        mask_f = mask.to(x.dtype).unsqueeze(-1)  # [B, S, 1]
        summed = (x * mask_f).sum(dim=1)  # [B, D]
        denom = mask_f.sum(dim=1).clamp_min(1.0)  # [B, 1] 防 0
        return summed / denom

    def forward(self, hidden_states: torch.Tensor, video_mask: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, S, D]
        video_mask:    [B, S] bool，True 表示该 token 是视频帧特征
        returns:
            k_logits: [B, K_max+1]
        """
        pooled = self.masked_mean(hidden_states, video_mask)  # [B, D]
        k_logits = self.net(pooled)  # [B, K_max+1]
        return k_logits


class BoundaryHead(nn.Module):
    """
    帧对齐版边界头：
    - 输入 hidden_states:[B,S,D], video_mask:[B,S] (bool), num_frames:[B] (int)
    - 思路：对视频 token 的时序序列做自适应池化 -> 帧级特征 [T,D]
      * two_frame_group=True: 先池到 G=ceil(T/2)，再线性插值到 T
      * two_frame_group=False: 直接池到 T
    - 输出:
        start_logits:[B,T_max], end_logits:[B,T_max], valid_mask:[B,T_max]
    """

    def __init__(self, embed_dim: int, proj_dim: int = 512, dropout: float = 0.1, two_frame_group: bool = True):
        super().__init__()
        self.two_frame_group = two_frame_group

        # 先把 token 级特征做一个轻 MLP，增强局部表达
        self.token_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # 帧级分类头（共享干路，最后一层分出 start/end）
        self.frame_head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, 2),  # 0: start, 1: end
        )

    def forward(self, hidden_states: torch.Tensor, video_mask: torch.Tensor, num_frames: torch.Tensor):
        """
        hidden_states: [B,S,D]
        video_mask:    [B,S] (bool), True 表示视频 token
        num_frames:    [B]  (int), 每条样本的帧数 T_b
        returns:
            start_logits: [B,T_max]
            end_logits:   [B,T_max]
            valid_mask:   [B,T_max] (用于损失mask)
        """
        B, S, D = hidden_states.shape
        device = hidden_states.device

        # 统一到模块参数 dtype，避免进入 LayerNorm 时 dtype 冲突
        try:
            ln_tok = self.token_proj[0]
            param_dtype = (
                ln_tok.weight.dtype if hasattr(ln_tok, "weight") and ln_tok.weight is not None else hidden_states.dtype
            )
        except Exception:
            param_dtype = next(self.parameters()).dtype
        hidden_states = hidden_states.to(dtype=param_dtype)

        per_sample_logits = []
        lengths = []
        for b in range(B):
            T = int(num_frames[b])
            lengths.append(T)
            vid_feats = hidden_states[b][video_mask[b]]  # [N_v, D]
            tok = self.token_proj(vid_feats)  # [N_v, C]
            tok = tok.transpose(0, 1).unsqueeze(0)  # [1, C, N_v]

            fr = F.adaptive_avg_pool1d(tok, output_size=T)  # [1, C, T]

            fr = fr.squeeze(0).transpose(0, 1)  # [T, C]
            logits = self.frame_head(fr)  # [T, 2]
            per_sample_logits.append(logits)

        # pad 到批内最大帧数 T_max
        padded = pad_sequence(per_sample_logits, batch_first=True, padding_value=0.0)  # [B, T_max, 2]
        start_logits = padded[..., 0]  # [B, T_max]
        end_logits = padded[..., 1]  # [B, T_max]

        # 有效帧 mask
        if len(lengths) == 0:
            valid_mask = torch.zeros((B, 0), dtype=torch.bool, device=device)
        else:
            T_max = padded.size(1)
            lens = torch.tensor(lengths, device=device).unsqueeze(1)  # [B,1]
            ar = torch.arange(T_max, device=device).unsqueeze(0)  # [1,T_max]
            valid_mask = ar < lens  # [B,T_max] bool

        return start_logits, end_logits, valid_mask


# ---------------------------
# 使用示例（训练时的损失计算）
# ---------------------------
if __name__ == "__main__":
    B, S, D = 3, 128, 1024
    K_MAX = 10

    hidden_states = torch.randn(B, S, D)
    # 假设每条样本前 40/60/25 个 token 是视频，其余是文本（仅示例；实际用你自己的 video_mask）
    video_lengths = torch.tensor([40, 60, 25])
    video_mask = torch.zeros(B, S, dtype=torch.bool)
    for b, t in enumerate(video_lengths):
        video_mask[b, :t] = True

    # 构建头
    k_head = KHead(embed_dim=D, k_max=K_MAX)
    bd_head = BoundaryHead(embed_dim=D)

    # 前向
    k_logits = k_head(hidden_states, video_mask)  # [B, K_MAX+1]
    start_logits, end_logits, valid_mask = bd_head(hidden_states, video_mask)  # [B,T_max], [B,T_max], [B,T_max]

    # 伪造一些 GT（演示）
    K_gt = torch.tensor([3, 2, 1])  # 每条样本的段数
    # 构造 start/end 的 0/1 序列（padding 部分全 0）
    T_max = start_logits.size(1)
    start_gt = torch.zeros(B, T_max)
    end_gt = torch.zeros(B, T_max)
    # 举例：第一条样本（长度 40）在帧 3、15、27 开始，在 10、20、35 结束
    start_gt[0, [3, 15, 27]] = 1.0
    end_gt[0, [10, 20, 35]] = 1.0
    # 其它样本随便填一点
    start_gt[1, [5, 30]] = 1.0
    end_gt[1, [20, 55]] = 1.0
    start_gt[2, [7]] = 1.0
    end_gt[2, [20]] = 1.0

    # —— 段数头损失（交叉熵）
    loss_k = F.cross_entropy(k_logits, K_gt.clamp_max(K_MAX))

    # —— 边界头损失（BCEWithLogits，带 valid_mask）
    def masked_bce_with_logits(logits, targets, mask):
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = loss * mask.float()
        denom = mask.float().sum().clamp_min(1.0)
        return loss.sum() / denom

    loss_start = masked_bce_with_logits(start_logits, start_gt, valid_mask)
    loss_end = masked_bce_with_logits(end_logits, end_gt, valid_mask)

    # 总损失（仅示例权重）
    loss = loss_k + 0.5 * (loss_start + loss_end)
    print("loss:", float(loss))
