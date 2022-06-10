import json
import os

import numpy as np

data_path = [
    r'D:\python_project\breg_graph\tmp\dataset\valid.json',
    r'D:\python_project\breg_graph\tmp\dataset1\valid.json',
    r'D:\python_project\breg_graph\tmp\dataset2\valid.json',
    r'D:\python_project\breg_graph\tmp\dataset3\valid.json',
]

new_data = []
for path in data_path:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.loads(f.readline())
    new_data.extend(data)

train_data = []
for item in new_data:
    item_data = {
        "dir": item['dir'],
        "folder": item['folder']
    }
    new_shape = np.array([0, 0])
    target = []
    for file in item['file']:
        shape = file['shape']
        for item1 in file['target']:
            target.append({
                "text": item1['text'],
                "label": item1['label'],
                'bbox': (np.array(item1['bbox']) + new_shape).tolist()
            })
        new_shape[1] = new_shape[1] + shape[1]
    item_data['target'] = target
    train_data.append(item_data)

save_path = r'D:\python_project\breg_graph\tmp\synthesize_dataset'
with open(os.path.join(save_path, "valid.json"), 'w', encoding='utf-8') as f:
    f.write(json.dumps(train_data))



