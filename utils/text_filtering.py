import json
import os

target_file = r'D:\python_project\breg_graph\test.txt'
texts = {}
with open(target_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        key = tmp[0].split("/")[-1].split('.')[0]
        text = tmp[1]
        texts[key] = text

missing_file = []
target_dir = r'C:\Users\thinhtq\Downloads\drive-download-20220526T024823Z-001\text_crop_image_new\text_crop_image_new'
for item in os.listdir(target_dir):
    if item.split(".")[0] not in texts:
        missing_file.append(item)

save_path = r'D:\python_project\breg_graph\checked_data'
with open(os.path.join(save_path, "true.txt"), 'w', encoding='utf-8') as f:
    for item in texts.keys():
        f.write(item)
        f.write("\n")
with open(os.path.join(save_path, "wrong.txt"), 'w', encoding='utf-8') as f:
    for item in missing_file:
        f.write(item)
        f.write("\n")
