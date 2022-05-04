import math
import os
import json
import numpy

test_label = ["01M012214_ĐKKD.json", "41S8004926_GPKD_T_N-1.jpg.json"]
test_data = []
new_data = []
gen_data = r'D:\python_project\dkkd_graph\data\gen'
save_dir = r'D:\python_project\dkkd_graph\data\breg'
count = 0
for i, file in enumerate(os.listdir(gen_data)):
    with open(os.path.join(gen_data, file), 'r', encoding='utf-8') as f:
        data = json.loads(f.readline())
        if file in test_label:
            test_data.extend(data)
        else:
            new_data.extend(data)
with open(os.path.join(save_dir, 'test.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(test_data))

indices = numpy.random.permutation(len(new_data))
training_idx, valid_idx = indices[:1520], indices[1520:]
train_data = []
for idx in training_idx:
    train_data.append(new_data[idx])
valid_data = []
for idx in valid_idx:
    valid_data.append(new_data[idx])
print(len(train_data), len(valid_data), len(test_data))
# for data in train_data:
#     for item in data['target']:
#         if len(item['text']) == 0:
#             print("-" * 44)
#             for item1 in data['target']:
#                 print(item1['text'], item1['label'])
#             print("-" * 44)
#             break


with open(os.path.join(save_dir, 'train.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(train_data))
with open(os.path.join(save_dir, 'valid.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(valid_data))

print(len(train_data))
