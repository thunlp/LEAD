import json
import random
from tqdm import tqdm
with open("/home/gaocheng/similar_case_added/data_pure_xs_clean_100060_95464_SCadded_30_train.json", "r") as f:
    data = json.load(f)

# with open("/home/gaocheng/similar_case_added/data_pure_xs_clean_100060_95464_SCadded_30_train.json", "r") as f:
#     data_dev = json.load(f)

# 从key中随机抽取一句话(句号分割), 作为新的query
new_data = []
count = 0
for d in tqdm(data):
    key = d["key"].strip()
    if key[-1] != "。":
        key += "。"
    key_list = [i for i in key.split("。") if i]
    if len(key_list) == 1:
        # 取中间0.3-0.7的部分
        query = key_list[0][int(0.3 * len(key_list[0])):int(0.7 * len(key_list[0]))]
        # new_key是其余的部分
        new_key = key_list[0][:int(0.3 * len(key_list[0]))] + key_list[0][int(0.7 * len(key_list[0])):]
        if query[-1] != "。":
            query += "。"
        if new_key[-1] != "。":
            new_key += "。"
    else:
        key_list = [x for x in key_list if len(x) > 10 and len(x) <= 0.5 * len(key)]
        if key_list == []:
            count += 1
            continue
        query = random.choice(key_list)
        if query[-1] != "。":
            query += "。"
        new_key = key.replace(query, "")
    d["query"] = query
    d["key"] = new_key
    new_data.append(d)
print(count)
    
with open("/home/gaocheng/similar_case_added/ruledata_pure_xs_clean_100060_95464_SCadded_30_train.json", "w") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)