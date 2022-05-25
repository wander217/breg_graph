import glob
import json
import os

text_path = r'D:\text_doanh_nghiep_230522.json'
data_root = r'D:\labelme_dn'
texts = {}
error_path = []
with open(text_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = json.loads(line)
        texts[tmp['url'].split("/")[-1].split(".")[0]] = tmp['text']
empty_path = r'D:\python_project\breg_graph\empty.txt'
with open(empty_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = line.split(" ")
        texts[tmp[0].split(".")[0]] = " ".join(tmp[0:]).strip()

miss_path = r'D:\python_project\breg_graph\miss_file.txt'
with open(miss_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = line.split(" ")
        texts[tmp[0].split(".")[0]] = " ".join(tmp[0:]).strip()

# print(texts.keys())

new_data = []
count = 0
file_count = 0

missing_image = []
for data_dir in os.listdir(data_root):
    for data_path in os.listdir(os.path.join(data_root, data_dir)):
        if "json" not in data_path:
            continue
        item_path = os.path.join(data_root, data_dir, data_path)
        with open(item_path, 'r', encoding='utf-8') as f:
            data = json.loads("".join(f.readlines()))
        tmp = data['imagePath'].split("/")
        item = {
            "file_name": "-".join(tmp[-2:]),
            "size": [data['imageWidth'], data['imageHeight']],
            "target": []
        }
        # count += len([item for item in data['shapes'] if item['label'] != 'document'])
        for i, shape in enumerate(data['shapes']):
            if shape['label'] == 'document':
                continue
            key = "__".join([data_dir, str(i), data_path.split(".")[0]])
            try:
                item['target'].append({
                    "text": texts[key],
                    "label": shape['label'],
                    "bbox": shape['points']
                })
            except Exception as e:
                count += 1
                missing_image.append(key)
        # new_data.append(item)
print(len(missing_image))

lost_path = r'D:\python_project\breg_graph\lost_image.txt'
with open(lost_path, 'w', encoding='utf-8') as f:
    for item in missing_image:
        f.write(item)
        f.write("\n")
