import json
import copy
import numpy as np

data_path = r'D:\python_project\breg_graph\convert_data.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.loads(f.readline())
split_data = {}

for value in data:
    tmp = value['file_name'].split("_")
    name = "{}__{}".format(value['folder'], tmp[0])
    if name not in split_data:
        split_data[name] = [value]
    else:
        split_data[name].append(value)

# print(split_data.keys())
synth_data = []
for key, value in split_data.items():
    item = {
        "file_name": value[0]['file_name'].split("_")[0],
        "folder": value[0]['folder'],
        "target": []
    }
    adding = np.array([0, 0])
    for i in range(len(value)):
        for target in value[i]['target']:
            target['bbox'] = (np.array(target['bbox']) + adding).tolist()
            item['target'].append(target)
        adding = np.array(value[i]['shape']) + adding
    synth_data.append(item)
with open(r'D:\python_project\breg_graph\abc.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(synth_data))
