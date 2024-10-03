import json
import os

result_dir = '/yeesuanAI06/thunlp/gaocheng/DPR/result/result_LeCard/results_Lawformer_cail2022_train_39epochs'
if "CAIL2022" in result_dir:
    dpr_rank_dir_name = 'DPR_rank_CAIL2022'
else:
    dpr_rank_dir_name = 'DPR_rank_LeCard'
# 遍历目录下的所有JSON文件
result_dict = {}
for filename in os.listdir(result_dir):
    if filename.endswith('.json'):
        file_path = os.path.join(result_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        ridx = filename[:-5]
        rank_list = []
        for question in data:
            for r in question["ctxs"]:
                rank_list.append(int(r["id"][7:])) # remove "lecard:"
        result_dict[ridx] = rank_list
# 取result_dir的最后一个目录名去掉results_，作为保存文件名

file_save = f"/yeesuanAI06/thunlp/gaocheng/DPR/DPR_rank/{dpr_rank_dir_name}/DPR_rank_{result_dir.split('/')[-1][8:]}" + ".json"
with open(file_save, 'w') as f:
    json.dump(result_dict, f)

print(f"{file_save} saved, length: {len(result_dict)}")
print("Maybe your next step is to run LeCaRD/metrics.py:")
print(f"python /yeesuanAI06/thunlp/gaocheng/LeCaRD/metrics.py --pred {file_save}")
os.system(f"python /yeesuanAI06/thunlp/gaocheng/LeCaRD/metrics.py --pred {file_save}")