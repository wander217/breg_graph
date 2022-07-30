import json
import os

data_root = r'C:\Users\Trinh_Thinh\Downloads\data_graph_3007\test_data_json_graph'
data = []
for file in os.listdir(data_root):
    with open(os.path.join(data_root, file), 'r', encoding='utf-8') as f:
        tmp = json.loads(f.read())
        if len(tmp) > 0:
            continue
        data.append({
            "file_name": file,
            "target": tmp
        })

data_len = len(data)
valid = data[:data_len//2]
test = data[data_len//2:]

save_path = r'D:\workspace\project\breg_graph\data\valid.json'
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(valid, indent=True))

save_path = r'D:\workspace\project\breg_graph\data\test.json'
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(test, indent=True))