import json
import os

data_path = r'C:\Users\thinhtq\Downloads\DATA_GENOCR_DKKD\DATA_GENOCR_DKKD'
save_path = r'D:\python_project\breg_graph\tmp1\dataset'
for folder in os.listdir(data_path):
    new_data = []
    for file in os.listdir(os.path.join(data_path, folder)):
        if not file.endswith("json"):
            continue
        file_path = os.path.join(data_path, folder, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())
        new_data.append(data)
    with open(os.path.join(save_path, "{}.json".format(folder)), 'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))
