from deepthulac import LacModel
import json

with open('/home/gaocheng/LeCaRD/data/query/query_simplify.json') as f:
    query_list = json.load(f)


query_short_list = []
for query in query_list:
    query_short_list.append(query["q_short"])

corpus = lac_seg.seg(query_short_list, show_progress_bar=True)['seg']['res']
corpus_pos = lac_pos.seg(query_short_list, show_progress_bar=True)['pos']['res']

for i, query in enumerate(query_list):
    for c in corpus[i]:
        assert c in query["q_short"]
    query["q_short_words"] = corpus[i]
    query["q_short_pos"] = corpus_pos[i]

with open('/home/gaocheng/LeCaRD/data/query/query_simplify_add_words.json', 'w') as f:
    json.dump(query_list, f, ensure_ascii=False, indent=4)