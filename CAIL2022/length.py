import json

with open("/yeesuanAI06/thunlp/gaocheng/CAIL2022/cail2022_train.json", "r") as f:
    data = json.load(f)
print(len(data))