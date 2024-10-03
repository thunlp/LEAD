import jieba
import json
from tqdm import tqdm
from deepthulac import LacModel
lac = LacModel.load(path="/liuzyai04/thunlp/gaocheng/deepthulac-pos-model", device='cuda:0') # 加载模型，path为模型文件夹路径，POS_MODEL表示自动从huggingface下载，device设置为cuda或cpu

file_path = "/liuzyai04/thunlp/gaocheng/raw_data/xs_split_1M.jsonl"

def cut_words(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    # count = 0
    sents = []
    for i in range(len(data)):
        # count += 1
        # if count == 1000:
        #     break
        sents.append(data[i]["ajjbqk"])
        sents[-1].strip()
        if not sents[-1]:
            sents[-1] = "无"
        # a = jieba.cut(data[i]["ajjbqk"], cut_all=False)
        # tem = " ".join(a).split()
        # data[i]["words"] = tem
    print(len(sents))
    results = lac.seg(sents, show_progress_bar=True)['pos']['res']
    for i in range(len(data)):
        data[i]["words"] = results[i]
    with open("/liuzyai04/thunlp/gaocheng/raw_data/xs_split_1M_addpos.jsonl", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


cut_words(file_path)
