import json

data_path = r'C:\Users\thinhtq\Downloads\01JuneTextData.json'
filter_data = {}
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        item = json.loads(line)
        tmp = item['url'].strip().split("/")[-1].split(".")[0]
        if 'text_crop_image_2' in item['url']:
            filter_data[tmp] = item['text']

true_path = r'D:\python_project\breg_graph\checked_data\true.txt'
true_data = []
with open(true_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("/")[0].split(".")
        true_data.append(tmp[0])

old_data_path = r"C:\Users\thinhtq\Downloads\test (1).txt"
with open(old_data_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        name = tmp[0].split("/")[-1].split(".")[0]
        if name in true_data:
            filter_data[name] = tmp[1]

with open(r"D:\python_project\breg_graph\save_text.json", 'w', encoding='utf-8') as f:
    f.write(json.dumps(filter_data))
