import json
import numpy as np
# /home/gaocheng/LeCaRD/data/prediction/bm25/bm25_top10_in_candidates.json
# golden

with open('/home/gaocheng/LeCaRD/data/label/golden_labels.json', 'r') as f:
    golden_labels = json.load(f)

with open('/home/gaocheng/LeCaRD/data/prediction/bm25/bm25_rank_in_candidates_shortquery_words.json', 'r') as f:
    data = json.load(f)

with open('/home/gaocheng/LeCaRD/data/label/label_top30_dict.json') as f:
    label_top30_dict = json.load(f)

# calculate the p@10

def precision_at(data, golden_labels, precise_at):
    p = 0
    for key, value in data.items():
        for i in range(precise_at):
            if value[i] in golden_labels[key]:
                p += 1
    return p/(len(data) * precise_at)

def mean_average_precision(data, golden_labels):
    all_ap = 0
    for key, value in data.items():
        ap = 0
        xg = 0
        for i in range(len(value)):
            if value[i] in golden_labels[key]:
                xg += 1
                ap += xg/(i+1)
        if xg == 0:
            continue
        ap /= xg
        all_ap += ap
    return all_ap / len(data)

def NDCG(data, label_top30_dict, n):
    ndcg_sum = 0
    for key, value in data.items():
        dcg = 0
        for i in range(n):
            if str(value[i]) not in label_top30_dict[str(key)]:
                g = 0
            else:
                g = label_top30_dict[str(key)][str(value[i])]
            dcg += g / np.log2(i + 2)
        idcg = 0
        # print(scores)
        scores = list(label_top30_dict[str(key)].values())
        scores.sort(reverse=True)
        # print(scores)
        # print("-------------------------")
        for i in range(n):
            g = scores[i]
            idcg += g / np.log2(i + 2)
        ndcg = dcg / idcg
        ndcg_sum += ndcg
    # print(f"ndcg_sum is {ndcg_sum}")
    return ndcg_sum / len(data)
        
print(f"p@5 is {precision_at(data, golden_labels, 5)}")
print(f"p@10 is {precision_at(data, golden_labels, 10)}")
print(f"map is {mean_average_precision(data, golden_labels)}")
print(f"NDCG@10 is {NDCG(data, label_top30_dict, 10)}")
print(f"NDCG@20 is {NDCG(data, label_top30_dict, 20)}")
print(f"NDCG@30 is {NDCG(data, label_top30_dict, 30)}")