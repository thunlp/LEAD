import json

with open("D:\gaocheng\科研\my_paper\LEAD\DPR_rank\DPR_rank_LeCard\DPR_rank_2048_xs30original_fp16_train_globalmask_1e-5_39epochs.json", "r") as f:
    data = json.load(f)

print(len(data))