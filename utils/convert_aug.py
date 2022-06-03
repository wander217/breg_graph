import json
import os

data_path = r'D:\python_project\breg_graph\data\DATA_GENOCR_DKKD'
for dirname in os.listdir(data_path):
    os.mkdir(os.path.join(r'D:\python_project\breg_graph\graph_data', dirname))
    for item in os.listdir(os.path.join(data_path, dirname)):
        if "json" in item:
            with open(os.path.join(data_path, dirname, item), 'r', encoding='utf-8') as f:
                data = json.loads("".join(f.readlines()))
            with open(os.path.join(r'D:\python_project\breg_graph\graph_data', dirname, item), 'w', encoding='utf-8') as f:
                f.write(json.dumps(data))
