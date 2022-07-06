import json
import numpy as np

data_path = r'D:\python_project\breg_graph\tmp\aug_data_1\test.json'
# data_path = r'D:\python_project\breg_graph\tmp\dataset\valid.json'

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.loads(f.read())

new_data = []
for item in data:
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
    if len(item_data['target']) == 0:
        print(item_data['folder'])
        continue
    new_data.append(item_data)

save_path = r'D:\python_project\breg_graph\tmp\synthetic_data_1\test.json'
# save_path = r'D:\python_project\breg_graph\test_data\test.json'
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(new_data))
