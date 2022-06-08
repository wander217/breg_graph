import json
import os

data_path = r'D:\python_project\breg_graph\tmp\valid_data.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.loads(f.readline())

length = [
    [40, 5, 5],
    [20, 3, 3],
    [41, 5, 5],
    [27, 3, 3],
    [15, 2, 2],
    [44, 6, 6],
    [26, 3, 3],
    [2, 1, 1],
    [49, 6, 6],
]

train, valid, test = [], [], []
count = 0
for key, value in data.items():
    if key == '-1':
        continue
    train.extend(value[:length[count][0]])
    valid.extend(value[length[count][0]:length[count][0] + length[count][1]])
    test.extend(value[length[count][0] + length[count][1]:])
    count += 1

dataset_path = r'D:\python_project\breg_graph\tmp\dataset'

with open(os.path.join(dataset_path, 'train.json'), 'w', encoding='utf-8') as f:
    print(len(train))
    f.write(json.dumps(train))

with open(os.path.join(dataset_path, 'valid.json'), 'w', encoding='utf-8') as f:
    print(len(valid))
    f.write(json.dumps(valid))

with open(os.path.join(dataset_path, 'test.json'), 'w', encoding='utf-8') as f:
    print(len(test))
    f.write(json.dumps(test))
