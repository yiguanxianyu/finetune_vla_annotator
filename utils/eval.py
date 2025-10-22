import torch
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.text import BERTScore
from tqdm import tqdm


def get_topk_for_bdhead(input_pt, k):
    return input_pt.topk(k).indices.sort().values.cpu()


@torch.inference_mode(True)
def eval_lm(model, processor, dataloader):
    model.eval()
    frames_pred = []
    frames_label = []
    ks = []
    klabels = []
    outputs_truncated = []
    labels_truncated = []
    bertscore = BERTScore(model_name_or_path="roberta-base", device="cuda:1")
    metric_frame = MulticlassF1Score(num_classes=30)
    metric_k = MulticlassF1Score(num_classes=16)

    for batch in tqdm(dataloader):
        inputs_lm, assist = batch
        bound_score = model.base_model.generate(**inputs_lm, max_new_tokens=1024)  # 先生成完整输出
        # 再前向一次获取隐藏层
        last_hidden_states = model.base_model(**inputs_lm, output_hidden_states=True).hidden_states[-1]

        k_logits = model.khead(last_hidden_states, assist["video_mask"]).argmax(dim=-1).cpu()
        start_logits, end_logits, valid_mask = model.bdhead(
            last_hidden_states, assist["video_mask"], assist["num_frames"]
        )
        start_frame_preds = get_topk_for_bdhead(start_logits, k_logits)
        start_frame_labels = get_topk_for_bdhead(assist["probs_start"], k_logits)
        end_frame_preds = get_topk_for_bdhead(end_logits, k_logits)
        end_frame_labels = get_topk_for_bdhead(assist["probs_end"], k_logits)
        frames_pred.append(start_frame_preds)
        frames_pred.append(end_frame_preds)
        frames_label.append(start_frame_labels)
        frames_label.append(end_frame_labels)
        ks.append(k_logits)
        klabels.append(assist["k_label"].cpu())

        for i, j, k in zip(bound_score, assist["answer_texts"], assist["text_label"]):
            ignore_token_pos = torch.where(k == -100)[0].max() + 2  # 截断到 -100 之后的位置,以及assist +"\n"的两个token
            i_trunc = i[ignore_token_pos:]
            outputs_truncated.append(i_trunc)
            labels_truncated.append(j)

    sentense = processor.batch_decode(outputs_truncated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    reference = processor.batch_decode(labels_truncated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    for i in range(len(reference)):
        sentense[i] = sentense[i].strip()
        reference[i] = "```json" + reference[i].strip() + "```"

    frames_pred = torch.cat(frames_pred, dim=1)
    frames_label = torch.cat(frames_label, dim=1)
    ks = torch.cat(ks, dim=0)
    klabels = torch.cat(klabels, dim=0)

    bert_score = bertscore(sentense, reference)
    bert_score = {k: v.mean().item() for k, v in bert_score.items()}
    bound_score = metric_frame(frames_pred, frames_label)
    k_score = metric_k(ks, klabels)

    return {
        "bert_score": bert_score,
        "bound_score": bound_score,
        "k_score": k_score,
    }
