import json

data_path = r'D:\python_project\breg_graph\tmp\category.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.loads("".join(f.readlines()))

convert_data = r'D:\python_project\breg_graph\convert_data.json'
with open(convert_data, 'r', encoding='utf-8') as f:
    data1 = json.loads(f.readline())
    for item in data1:
        tmp = data['file'][item['folder']][item['file_name'].split("_")[0]]
        item['type'] = tmp

rotate_data = r'D:\python_project\breg_graph\tmp\rotate.json'
with open(rotate_data, 'r', encoding='utf-8') as f:
    data2 = json.loads("".join(f.readlines()))
    for item in data1:
        tmp = data2[item['folder']]
        # item['rotate'] = 0
        for item1 in tmp:
            if item['file_name'] in item1:
                item['rotate'] = item1[item['file_name']]
                break

stat = {}
for folder, item in data['file'].items():
    for key, value in item.items():
        if value not in stat:
            stat[value] = [{
                "dir": folder,
                "folder": key,
                "file": []
            }]
        else:
            stat[value].append({
                "dir": folder,
                "folder": key,
                "file": []
            })

for item in data1:
    folder = item['file_name'].split("_")[0]
    dirname = item['folder']
    tmp = stat[item['type']]
    for item1 in tmp:
        if folder == item1['folder'] and dirname == item1['dir']:
            item1['file'].append(item)
            break

# for key, value in stat.items():
#     item = value[20]
#     for file in item['file']:
#         print(file)
#     break

with open("valid_data.json", 'w', encoding='utf-8') as f:
    f.write(json.dumps(stat))
