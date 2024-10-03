import os
import re
import numpy as np
import json
from tqdm import tqdm
import argparse
from gensim.summarization import bm25
import jieba

with open("/home/gaocheng/LeCaRD/data/others/stopword.txt", 'r') as g:
    words = g.readlines()
stopwords = [i.strip() for i in words]
stopwords.extend(['.','（','）','-'])

with open("/home/gaocheng/CAIL2022/stage2/query_stage2_valid_onlystage2_40_simplified.json", 'r') as f:
    dicts = json.load(f)

def makemodel(query_dict: dict):
    target_path = os.path.join("/home/gaocheng/CAIL2022/stage2/candidates_stage2_valid", str(query_dict['ridx']))
    if not os.path.exists(target_path):
        print(f"{query_dict['ridx']} not found")
        return
    # print(target_path)
    corpus = []
    id = []
    files = os.listdir(target_path)
    for file in files:
        if file.endswith(".json"):
            id.append(int(file[:-5]))
            json_file_path = os.path.join(target_path, file)
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                corpus.append(data['jieba_cut'])
    bm25Model = bm25.BM25(corpus)
    return bm25Model, id

with open("/home/gaocheng/CAIL2022/stage2/bm25_results_CAIL2022simplified.json", 'r') as f:
    rankdic = json.load(f)

# For cut off:

for query_dict in tqdm(dicts):
    if query_dict['ridx'] in rankdic.keys():
        continue
    bm25Model, id = makemodel(query_dict)
    a = jieba.cut(query_dict['q_short'], cut_all=False)
    tem = " ".join(a).split()
    q = [i for i in tem if i not in stopwords]
    index_list = np.array(bm25Model.get_scores(q)).argsort().tolist()
    rankdic[query_dict['ridx']] = [id[i] for i in reversed(index_list)]

with open("/home/gaocheng/CAIL2022/stage2/bm25_results_CAIL2022simplified.json", 'w') as f:
    json.dump(rankdic, f, ensure_ascii=False)
    