import json
import os

data_root = r'C:\Users\thinhtq\Downloads\data_json_graph\data_json_graph'
data = []
for file in os.listdir(data_root):
    with open(os.path.join(data_root, file), 'r', encoding='utf-8') as f:
        data.append({
            "file_name": file,
            "target": json.loads(f.read())
        })

save_path = r'D:\python_project\breg_graph\data\total_data'
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(data, indent=True))
