import json
import os

data_path = r'D:\python_project\breg_graph\tmp\valid_data.json'
# data_path = r'D:\python_project\breg_graph\tmp\ category\0.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.loads(f.read())

train, valid, test = [], [], []
count = 0
for key, value in data.items():
    if key == '-1':
        continue
    tmp = len(value)
    train_len = int(0.8 * tmp)
    valid_len = int(0.1 * tmp)
    train.extend(value[:train_len])
    valid.extend(value[train_len:train_len + valid_len])
    test.extend(value[train_len + valid_len:])
    count += 1

dataset_path = r'D:\python_project\breg_graph\tmp\dataset'


with open(os.path.join(dataset_path, 'train.json'), 'w', encoding='utf-8') as f:
    print(len(train))
    f.write(json.dumps(train, indent=4))

with open(os.path.join(dataset_path, 'valid.json'), 'w', encoding='utf-8') as f:
    print(len(valid))
    f.write(json.dumps(valid, indent=4))

with open(os.path.join(dataset_path, 'test.json'), 'w', encoding='utf-8') as f:
    print(len(test))
    f.write(json.dumps(test, indent=4))
