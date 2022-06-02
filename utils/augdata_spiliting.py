import json
import os

import numpy

# data = {}
# count = 0
# data_path = r'C:\Users\Trinh_Thinh\Downloads\graph_data'
# for dirname in os.listdir(data_path):
#     data[dirname] = []
#     for file in os.listdir(os.path.join(data_path, dirname)):
#         with open(os.path.join(data_path, dirname, file), 'r', encoding='utf-8') as f:
#             item_data = json.loads("".join(f.readlines()))
#         data[dirname].append({
#             "file_name": file.split(".")[0],
#             "target": item_data
#         })
#         count += 1
#         print(count)
#
# train = []
# valid = []
# for key, value in data.items():
#     indices = numpy.random.permutation(len(value))
#     training_idx, valid_idx = indices[:1520], indices[1520:]
#     for idx in training_idx:
#         train.append(value[idx])
#     for idx in valid_idx:
#         valid.append(value[idx])

save_path = r'D:\workspace\project\breg_graph\fake_data'
with open(os.path.join(save_path, "train.json"), 'r', encoding='utf-8') as f:
    print(len(json.loads(f.readline())))
with open(os.path.join(save_path, "valid.json"), 'r', encoding='utf-8') as f:
    print(len(json.loads(f.readline())))