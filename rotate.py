import json
import os

rotate = {}
root = r'D:\labelme_dn'
for dirname, _, files in os.walk(root):
    name = dirname.split("\\")[-1]
    rotate[name] = []
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            rotate[name].append({file: 0})

save_path = r'D:\python_project\breg_graph\tmp\test.json'
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(rotate, indent=4))
