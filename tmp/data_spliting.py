import json
import os

# data_path = r'D:\python_project\breg_graph\tmp\valid_data.json'
data_path = r'D:\python_project\breg_graph\tmp\ category\0.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.loads("".join(f.readlines()))

train, valid, test = [], [], []
count = 0
# for item in data:
    # if key == '-1':
    #     continue
tmp = len(data)
train_len = int(0.8 * tmp)
valid_len = int(0.1 * tmp)
train.extend(data[:train_len])
valid.extend(data[train_len:train_len + valid_len])
test.extend(data[train_len + valid_len:])
# count += 1

dataset_path = r'D:\python_project\breg_graph\tmp\dataset'

with open(os.path.join(dataset_path, 'train_0.json'), 'w', encoding='utf-8') as f:
    print(len(train))
    f.write(json.dumps(train))

with open(os.path.join(dataset_path, 'valid_0.json'), 'w', encoding='utf-8') as f:
    print(len(valid))
    f.write(json.dumps(valid))

with open(os.path.join(dataset_path, 'test_0.json'), 'w', encoding='utf-8') as f:
    print(len(test))
    f.write(json.dumps(test))
