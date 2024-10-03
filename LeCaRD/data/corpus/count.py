import json

with open('document_path.json', 'r') as f:
    paths = json.load(f)

count = 0
for key, value in paths.items():
    print(key)
    count += len(value)

print(count)