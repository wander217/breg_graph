import json
import os
import numpy as np

t = "valid"

data_path = [
    r'D:\python_project\breg_graph\tmp\dataset\{}.json'.format(t),
    r'D:\python_project\breg_graph\tmp\dataset1\{}.json'.format(t),
    r'D:\python_project\breg_graph\tmp\dataset2\{}.json'.format(t),
    r'D:\python_project\breg_graph\tmp\dataset3\{}.json'.format(t),
    r'D:\python_project\breg_graph\tmp\dataset4\{}.json'.format(t),
    r'D:\python_project\breg_graph\tmp\dataset5\{}.json'.format(t),
    r'D:\python_project\breg_graph\tmp\dataset6\{}.json'.format(t),
    r'D:\python_project\breg_graph\tmp\dataset7\{}.json'.format(t),
    r'D:\python_project\breg_graph\tmp\dataset8\{}.json'.format(t),
    r'D:\python_project\breg_graph\tmp\dataset9\{}.json'.format(t),
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
with open(os.path.join(save_path, "{}.json".format(t)), 'w', encoding='utf-8') as f:
    print(len(train_data))
    f.write(json.dumps(train_data))



