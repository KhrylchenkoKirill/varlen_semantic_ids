import numpy as np


def calculate_metrics(candidates, targets, k=100):
    total_ndcg = 0.
    total_recall = 0.
    step = 0
    for user_id, gt in targets.iter_rows():
        if len(gt) == 0:
            continue
        gt = set(gt)
        if user_id not in candidates:
            total_ndcg -= total_ndcg / (step + 1)
            total_recall -= total_recall / (step + 1)
            continue
        
        pred = candidates[user_id][:k]
        target_mask = np.array([el in gt for el in pred])
        assert target_mask.shape[0] <= k

        dcg = (target_mask.astype(np.float32) / np.log2(np.arange(2, len(pred) + 2))).sum()
        ideal_dcg = (1. / np.log2(np.arange(2, min(k, len(gt)) + 2))).sum()
        ndcg = dcg / ideal_dcg
        total_ndcg += (ndcg - total_ndcg) / (step + 1)

        recall = target_mask.sum() / min(k, len(gt))
        total_recall += (recall - total_recall) / (step + 1)
        
        step += 1

    return {'recall': float(total_recall), 'ndcg': float(total_ndcg)}
