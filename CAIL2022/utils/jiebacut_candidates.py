import json
import jieba
import os
from tqdm import tqdm
with open("/home/gaocheng/LeCaRD/data/others/stopword.txt", 'r') as g:
    lines = g.readlines()
stopwords = [i.strip() for i in lines]
stopwords.extend(['.','（','）','-'])

# a = jieba.cut(s, cut_all=False)
# tem = " ".join(a).split()
# tem = [i for i in tem if not i in stopwords]
# print(tem)

paths = []
base_path = "/home/gaocheng/CAIL2022/stage2/candidates_stage2_valid"
for dir_name in os.listdir(base_path):
    for file_name in os.listdir(os.path.join(base_path, dir_name)):
        if file_name.endswith(".json"):
            paths.append(os.path.join(base_path, dir_name, file_name))

# print(len(paths), paths)
for file in tqdm(paths):
    with open(file, 'r') as f:
        data = json.load(f)
    if "jieba_cut" in data.keys():
        continue
    a = jieba.cut(data["ajjbqk"], cut_all=False)
    tem = " ".join(a).split()
    tem = [i for i in tem if not i in stopwords]
    data['jieba_cut'] = tem
    with open(file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)