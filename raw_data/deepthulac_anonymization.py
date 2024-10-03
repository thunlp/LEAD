from deepthulac import LacModel
import json
from tqdm import tqdm

with open("/home/gaocheng/Law_data/data_xs/data_xs.json", "r") as f:
    data_xs = json.load(f)
query_list = [d["query"] for d in data_xs]

lac_seg = LacModel.load(path="/home/gaocheng/deepthulac/deepthulac-seg", device='cuda:0') # 加载模型，path为模型文件夹路径，SEG_MODEL表示自动从huggingface下载，device设置为cuda/cpu/mps
lac_pos = LacModel.load(path="/home/gaocheng/deepthulac/deepthulac-pos", device='cuda:0') # 加载模型，path为模型文件夹路径，POS_MODEL表示自动从huggingface下载，device设置为cuda或cpu

corpus_pos = lac_pos.seg(query_list, show_progress_bar=True)['pos']['res']
with open ("/home/gaocheng/Law_data/data_xs/data_xs_pos.json", "w") as f:
    json.dump(corpus_pos, f, ensure_ascii=False, indent=4)
print(len(corpus_pos))

# n/名词 np/人名 ns/地名 ni/机构名 nz/其它专名
# m/数词 q/量词 mq/数量词 t/时间词 f/方位词 s/处所词
# v/动词 a/形容词 d/副词 h/前接成分 k/后接成分 i/习语
# j/简称 r/代词 c/连词 p/介词 u/助词 y/语气助词
# e/叹词 o/拟声词 g/语素 w/标点 x/其它
n_list = ["np", "ns", "ni", "nz"]
Alpha_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
              "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
              "U", "V", "W", "X", "Y", "Z"]
count = 0
for i in tqdm(range(len(corpus_pos))):
    new_query = ""
    judge = True
    name_dict = {}
    hulue_list = []
    for j in range(len(corpus_pos[i])):
        if j in hulue_list:
            continue
        w_p = corpus_pos[i][j].split("_")
        if w_p[1] not in n_list:
            if w_p[1] == "t" and "某" not in w_p[0] and (j == 0 or corpus_pos[i][j-1].split("_")[0] != "某"):
                if "年" in w_p[0]:
                    new_query += "某年"
                if "月" in w_p[0]:
                    new_query += "某月"
                if "日" in w_p[0]:
                    new_query += "某日"
            else:
                new_query += w_p[0]
        else:
            judge = False
            this_n = w_p[1]
            name_str = w_p[0]
            # print(name_str, this_n)
            while j < len(corpus_pos[i]) - 1 and corpus_pos[i][j+1].split("_")[1] == this_n:
                # print("adding", corpus_pos[i][j+1].split("_")[0])
                j += 1
                hulue_list.append(j)
                name_str += corpus_pos[i][j].split("_")[0]
            if name_str not in name_dict:
                l = len(name_dict)
                subfix = ""
                if this_n == "ns":
                    subfix = "地"
                elif this_n == "ni":
                    subfix = "机构"
                Alpha = Alpha_list[l % 26]
                while l >= 26:
                    l = l // 26
                    Alpha = Alpha_list[l % 26] + Alpha
                name_dict[name_str] = Alpha + subfix
            new_query += name_dict[name_str]
    if not judge:
        count += 1
    print(new_query)
    data_xs[i]["query"] = new_query

print(f"count: {count}")
# 10334 for ms and 39000 for xs
# with open("/home/gaocheng/Law_data/data_xs/data_xs_new.json", "w") as f:
#     json.dump(data_xs, f, ensure_ascii=False, indent=4)