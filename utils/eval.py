# eval.py

import torch
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.text import BERTScore
from tqdm import tqdm


def get_topk_for_bdhead(input_pt, k):
    return input_pt.topk(k).indices.sort().values.cpu()


@torch.inference_mode()
def eval_lm(model, processor, dataloader, accelerator):
    model.eval()

    # —— 每个 rank 的“本地”缓存 —— #
    local_frames_pred = []
    local_frames_label = []
    local_ks = []
    local_klabels = []
    local_pred_texts = []  # 先在各自 rank 解码成字符串，再聚合
    local_ref_texts = []

    # 只有主进程展示进度条，防止多卡重叠打印
    iterator = tqdm(dataloader) if accelerator.is_main_process else dataloader

    for batch in iterator:
        inputs_lm, assist = batch

        # 生成 & 截断（仍在各自 rank 上做）
        bound_ids = model.base_model.generate(**inputs_lm, max_new_tokens=1024)

        # 再跑一次前向拿隐藏态
        last_hidden_states = model.base_model(**inputs_lm, output_hidden_states=True).hidden_states[-1]

        k_logits = model.khead(last_hidden_states, assist["video_mask"]).argmax(dim=-1).cpu()
        start_logits, end_logits, valid_mask = model.bdhead(
            last_hidden_states, assist["video_mask"], assist["num_frames"]
        )
        start_frame_preds = get_topk_for_bdhead(start_logits, k_logits)
        start_frame_labels = get_topk_for_bdhead(assist["probs_start"], k_logits)
        end_frame_preds = get_topk_for_bdhead(end_logits, k_logits)
        end_frame_labels = get_topk_for_bdhead(assist["probs_end"], k_logits)
        accelerator.print(torch.stack([start_frame_preds, end_frame_preds], dim=1).shape)
        local_frames_pred.append(torch.stack([start_frame_preds, end_frame_preds], dim=1))  # [B, 2, K]
        local_frames_label.append(torch.stack([start_frame_labels, end_frame_labels], dim=1))  # [B, 2, K]
        local_ks.append(k_logits)
        local_klabels.append(assist["k_label"].cpu())

        # —— 文本：各 rank 先截断再解码成字符串，再汇总为 Python 对象 —— #
        # 注意：这里按你原始逻辑，用 -100 的位置做截断
        for pred_ids, ref_texts, label_ids in zip(bound_ids, assist["answer_texts"], assist["text_label"]):
            ignore_pos = torch.where(label_ids == -100)[0].max() + 2  # assist + "\n" 的两个 token
            pred_trim = pred_ids[ignore_pos:]
            pred_str = processor.decode(pred_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            ref_str = processor.decode(ref_texts, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            local_pred_texts.append(pred_str)
            local_ref_texts.append(ref_str)

    # —— 把各 rank 的张量结果聚合到 rank0 —— #
    # 先把每个 rank 的列表拼起来（本地）
    if len(local_frames_pred):
        frames_pred = torch.cat(local_frames_pred, dim=0).cpu()
        frames_label = torch.cat(local_frames_label, dim=0).cpu()
        ks = torch.cat(local_ks, dim=0).cpu()
        klabels = torch.cat(local_klabels, dim=0).cpu()
    else:
        frames_pred = torch.empty(0, 2, 1, dtype=torch.long).cpu()
        frames_label = torch.empty(0, 2, 1, dtype=torch.long).cpu()
        ks = torch.empty(0, dtype=torch.long).cpu()
        klabels = torch.empty(0, dtype=torch.long).cpu()

    accelerator.print("- Inference completed, gathering data ...")
    accelerator.print(f"- data shape: {frames_pred.size()}, {frames_label.size()}, {ks.size()}, {klabels.size()}")

    # 使用 Accelerate 的收集接口（会自动处理不同世界大小）
    # 文本/任意对象：用 gather_object 聚合到 rank0
    frames_pred_all = accelerator.gather_for_metrics(frames_pred, use_gather_object=True)
    frames_label_all = accelerator.gather_for_metrics(frames_label, use_gather_object=True)
    ks_all = accelerator.gather_for_metrics(ks, use_gather_object=True)
    klabels_all = accelerator.gather_for_metrics(klabels, use_gather_object=True)
    pred_texts_all_lists = accelerator.gather_for_metrics(local_pred_texts, use_gather_object=True)
    ref_texts_all_lists = accelerator.gather_for_metrics(local_ref_texts, use_gather_object=True)

    accelerator.print("- Gathering done, start computing metrics ...")
    accelerator.print(
        f"- total data shape: {frames_pred_all.size()}, {frames_label_all.size()}, {ks_all.size()}, {klabels_all.size()}"
    )

    if accelerator.is_main_process:
        # 将多个列表拍平
        pred_texts_all = []
        ref_texts_all = []
        for lst in pred_texts_all_lists:
            pred_texts_all.extend(lst)
        for lst in ref_texts_all_lists:
            ref_texts_all.extend(lst)

        # 计算指标（仅在 rank0）
        bertscore = BERTScore(model_name_or_path="roberta-base", device="auto", truncation=True)
        metric_frame = MulticlassF1Score(num_classes=30)
        metric_k = MulticlassF1Score(num_classes=16)

        # frames：你原来是把 start/end 拼起来后再算一个 F1；保持一致
        # 维度 [N, 2, K] -> 拼接第二维
        frames_pred_cat = frames_pred_all.view(frames_pred_all.size(0), -1)
        frames_label_cat = frames_label_all.view(frames_label_all.size(0), -1)
        accelerator.print(len(pred_texts_all))
        bert_score = bertscore(pred_texts_all, ref_texts_all)
        bert_score = {k: v.mean().item() for k, v in bert_score.items()}
        bound_score = metric_frame(frames_pred_cat, frames_label_cat)
        k_score = metric_k(ks_all, klabels_all)

        return {
            "bert_score": bert_score,
            "bound_score": bound_score,
            "k_score": k_score,
        }
    else:
        # 非主进程返回空字典即可
        return {}
