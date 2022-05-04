import json
import random

# metric_path = r'D:\workspace\project\dkkd_graph\test\metric.txt'
# with open(metric_path, 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     f1_score = 0
#     best = None
#     place = 0
#     for i, line in enumerate(lines):
#         tmp = json.loads(line)
#         if tmp['validation']['avg_f1'] > f1_score:
#             best = tmp
#             place = i
#     print(place)
#     for item in best["validation"]["metric"]:
#         print(item)

# with open(r"D:\workspace\project\dkkd_graph\data\train.json", "r", encoding="utf-8") as f:
#     data = json.loads(f.readline())
# for item in data[0]['target']:
#     print(item['text'],item['label'])

# with open(r"D:\workspace\project\dkkd_graph\data\train.json", 'r', encoding='utf-8') as f:
#     train_data: list = json.loads(f.readline())
# with open(r"D:\workspace\project\dkkd_graph\data\test.json", 'r', encoding='utf-8') as f:
#     test_data = json.loads(f.readline())
# train_data.extend(test_data)
# with open(r"D:\workspace\project\dkkd_graph\data\train.json", 'w', encoding='utf-8') as f:
#     f.write(json.dumps(train_data))

with open(r"D:\workspace\project\dkkd_graph\data\valid.json", 'r', encoding='utf-8') as f:
    valid_data: list = json.loads(f.readline())

test_data = valid_data[:51]
valid_data = valid_data[51:]
print(len(test_data), len(valid_data))
with open(r"/data/breg/test.json", 'w', encoding='utf-8') as f:
    f.write(json.dumps(test_data))

with open(r"D:\workspace\project\dkkd_graph\data\valid.json", 'w', encoding='utf-8') as f:
    f.write(json.dumps(valid_data))