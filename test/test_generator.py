import json
import os
import random

target_file = r'D:\workspace\project\dkkd_graph\data\gen\41X8005336_GPKD_LE_THANG.json'
with open(target_file, 'r', encoding='utf-8') as f:
    data = json.loads(f.readline())
print(data[random.randint(0, len(data)-1)])
for item in data[random.randint(0, len(data)-1)]['target']:
    print(item['text'], item['label'])

# target_file = r'D:\python_project\dkkd_graph\data\total.json'
# with open(target_file, 'r', encoding='utf-8') as f:
#     data = json.loads(f.readline())
#
# for item in data:
#     if "41J8023212_GPKD_KA_LONG" in item["file_name"]:
#         print(item["target"])
