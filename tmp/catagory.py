import json
import os

contract_type = [
    "Doanh nghiệp tư nhân",  # 0
    "Công ty trách nghiệm hữu hạn một thành viên",  # 1
    "Văn phòng đại diện",  # 2
    "Doanh nghiệp tư nhân",  # 3
    "Chi nhánh",  # 4
    "Công ty trách nghiệm hữu hạn hai thành viên trở lên",  # 5
    "Hộ kinh doanh",  # 6
    "Công ty hợp danh",  # 7
    "Công ty cổ phần",  # 8
]

category = {}
root = r'D:\labelme_dn'
for dirname, _, files in os.walk(root):
    name = dirname.split("\\")[-1]
    category[name] = {}
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            category[name][file.split("_")[0]] = 0

save_path = r'D:\python_project\breg_graph\tmp\category.json'
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps({
        "type": contract_type,
        "file": category
    }, indent=4))
