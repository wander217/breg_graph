import json
import os

import numpy

data = {}
count = 0
data_path = r'C:\Users\Trinh_Thinh\Downloads\graph_data'
for dirname in os.listdir(data_path):
    data[dirname] = []
    for file in os.listdir(os.path.join(data_path, dirname)):
        with open(os.path.join(data_path, dirname, file), 'r', encoding='utf-8') as f:
            item_data = json.loads("".join(f.readlines()))
        data[dirname].append({
            "file_name": file.split(".")[0],
            "target": item_data
        })
        count += 1

label = {}
for key, value in data.items():
    for file in value:
        for item in file['target']:
            label[item['label']] = 1
print(len(label.keys()))
with open(r'D:\workspace\project\breg_graph\asset\breg\label.json','w', encoding='utf-8') as f:
    f.write(json.dumps(list(label.keys())))

# save_path = r'D:\workspace\project\breg_graph\fake_data'
# with open(os.path.join(save_path, "train.json"), 'r', encoding='utf-8') as f:
#     print(len(json.loads(f.readline())))
# with open(os.path.join(save_path, "valid.json"), 'r', encoding='utf-8') as f:
#     print(len(json.loads(f.readline())))