import json
import os

contract_type = [
    "Doanh nghiệp tư nhân",
    "Công ty trách nghiệm hữu hạn một thành viên",
    "Văn phòng đại diện",
    "Doanh nghiệp tư nhân",
    "Chi nhánh",
    "Công ty trách nghiệm hữu hạn hai thành viên trở lên",
    "Hộ kinh doanh",
    "Công ty hợp danh",
    "Công ty cổ phần"
]

category = {}
root = r'D:\python_project\breg_graph\tmp\convert_data'
for dirname, _, files in os.walk(root):
    name = dirname.split("\\")[-1]
    category[name] = {}
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            category[name][file.split("_")[0]] = 0

save_path = r'D:\python_project\breg_graph\tmp\category.json'
with open(save_path, 'r', encoding='utf-8') as f:
    data = json.loads("".join(f.readlines()))

for folder, value in data['file'].items():
    for file, file_type in value.items():
        if file_type == 3 or file_type == 0:
            value[file] = 0
        elif file_type == 2 or file_type == 4:
            value[file] = 2

save_path1 = r'D:\python_project\breg_graph\tmp\category1.json'
with open(save_path1, 'w', encoding='utf-8') as f:
    f.write(json.dumps(data, indent=4))