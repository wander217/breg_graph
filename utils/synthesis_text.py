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
        # if tmp['text'] == '<[EMPTY]>':
        #     error_path.append(tmp['url'].split("/")[-1])
        texts[tmp['url'].split("/")[-1]] = tmp['text']

# save_path = r'D:\python_project\breg_graph\error.txt'
# with open(save_path, 'w', encoding='utf-8') as f:
#     for item in error_path:
#         f.write(item)
#         f.write("\n")


new_data = []

# for data_dir in os.listdir(data_root):
#     for data_path in os.listdir(os.path.join(data_root, data_dir)):
#         if "json" not in data_path:
#             continue
#         item_path = os.path.join(data_root, data_dir, data_path)
#         with open(item_path, 'r', encoding='utf-8') as f:
#             data = json.loads("".join(f.readlines()))
#         # item = {
#         #     "file_name": data['imagePath'],
#         #     "size": [data['imageWidth'], data['imageHeight']],
#         #     "target": []
#         # }
#         for i, shape in enumerate(data['shapes']):
#             key = "__".join([data_dir, str(i), data_path.split(".")[0]])
#             print(key)
#             if key not in texts:
#                 error_path.append(key)
#         # new_data.append(item)
#
for item in os.listdir(r"D:\text_crop_image"):
    if item not in texts:
        error_path.append(item)
print(len(error_path))
save_path = r'D:\python_project\breg_graph\miss_file.txt'
with open(save_path, 'w', encoding='utf-8') as f:
    for item in error_path:
        f.write(item)
        f.write("\n")


