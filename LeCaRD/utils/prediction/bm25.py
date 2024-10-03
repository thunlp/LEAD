import os
import re
import numpy as np
import json
from tqdm import tqdm
import argparse
from gensim.summarization import bm25
import jieba


parser = argparse.ArgumentParser(description="Help info.")
parser.add_argument('--s', type=str, default='data/others/stopword.txt', help='Stopword path.')
parser.add_argument('--q', type=str, default='/home/gaocheng/LeCaRD/data/query/query_simplify_add_words.json', help='Query path.')
parser.add_argument('--split-dir', type=str, default='/home/gaocheng/LeCaRD/data/candidates/similar_case', help='Split corpus path.')
parser.add_argument('--w', type=str, default='data/prediction/bm25/bm25_rank_in_candidates_shortquery_words.json', help='Write path.')

args = parser.parse_args()

# with open(args.split, 'r') as f:
#     corpus = json.load(f)


# print(corpus)
# exit(0)
# corpus = [jieba.lcut(data) for data in self.data_list]


with open(args.s, 'r') as g:
    words = g.readlines()
stopwords = [i.strip() for i in words]
stopwords.extend(['.','（','）','-'])

with open(args.q, 'r') as f:
    dicts = json.load(f)

def makemodel(query_dict: dict):
    candidate1_path = os.path.join(args.split_dir, 'candidates1')
    candidate2_path = os.path.join(args.split_dir, 'candidates2')
    target_path = os.path.join(candidate1_path, str(query_dict['ridx']))
    if not os.path.exists(target_path):
        target_path = os.path.join(candidate2_path, str(query_dict['ridx']))
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
                corpus.append(data['words'])
    # id.sort()
    # print(id)
    # exit(0)
    bm25Model = bm25.BM25(corpus)
    # if len(corpus) != 100:
    #     print(f"{query_dict['ridx']} has {len(corpus)} candidates")
    return bm25Model, id
rankdic = {}

# For cut off:
label_dic = json.load(open('/home/gaocheng/LeCaRD/data/label/label_top30_dict.json','r'))

for query_dict in tqdm(dicts):
    bm25Model, id = makemodel(query_dict)

    # a = jieba.cut(query_dict['q_short'], cut_all=False)
    # tem = " ".join(a).split()
    tem = query_dict['q_short_words']
    q = [i for i in tem]
    index_list = np.array(bm25Model.get_scores(q)).argsort().tolist()
    rankdic[query_dict['ridx']] = [id[i] for i in reversed(index_list)]
    # print(rankdic[query_dict['ridx']])
    # print(eval(line)['ridx'], corpus[np.array(bm25Model.get_scores(q)).argsort()[-2]])
    # For cut off:
    # all_results = bm25Model.get_scores(q)
    # results = {i:all_results[int(i)] for i in list(label_dic[str(eval(line)['ridx'])].keys())[:30]}
    # sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    # rankdic[eval(line)['ridx']] = {i[0]:i[1] for i in sorted_results}

with open(args.w, 'w') as f:
    json.dump(rankdic, f, ensure_ascii=False)
    