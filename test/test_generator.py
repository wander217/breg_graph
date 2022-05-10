import json
import os
import random

import numpy as np

target_file = r'D:\python_project\breg_graph\data\gen\01F8010705_DKKD.json'
with open(target_file, 'r', encoding='utf-8') as f:
    data = json.loads(f.readline())
print(data[random.randint(0, len(data) - 1)])
for item in data[random.randint(0, len(data) - 1)]['target']:
    tmp = np.array(item['bbox'])
    print(item['text'], item['label'], np.max(tmp[:, 0]) - np.min(tmp[:, 0]))

# target_file = r'D:\python_project\dkkd_graph\data\total.json'
# with open(target_file, 'r', encoding='utf-8') as f:
#     data = json.loads(f.readline())
#
# for item in data:
#     if "41J8023212_GPKD_KA_LONG" in item["file_name"]:
#         print(item["target"])
