import json
import os
import random

train = []
valid = []
test = []
data_path = r'D:\python_project\breg_graph\TMP1'
for file in os.listdir(data_path):
    if file.endswith(".json"):
        with open(os.path.join(data_path, file)) as f:
            data = json.loads(f.read())
        random.shuffle(data)
        train_len = int(0.8 * len(data))
        valid_len = int(0.1 * len(data))
        train_item = data[:train_len]
        valid_item = data[train_len: train_len + valid_len]
        test_item = data[train_len+valid_len:]
        train.extend(train_item)
        valid.extend(valid_item)
        test.extend(test_item)

save_path = r'D:\python_project\breg_graph\TMP1\dataset'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

with open(os.path.join(save_path, "train.json"), 'w', encoding='utf-8') as f:
    print(len(train))
    f.write(json.dumps(train, indent=4))
with open(os.path.join(save_path, "valid.json"), 'w', encoding='utf-8') as f:
    print(len(valid))
    f.write(json.dumps(valid, indent=4))
with open(os.path.join(save_path, "test.json"), 'w', encoding='utf-8') as f:
    print(len(test))
    f.write(json.dumps(test, indent=4))
