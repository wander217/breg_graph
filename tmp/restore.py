import json
import imgaug.augmenters as iaa
import imgaug as ia


def convert_data(target):
    pass


data_path = r'D:\python_project\breg_graph\tmp\valid_data.json'

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.loads(f.readline())

for key, value in data.items():
    for folder in value:
        for file in folder['file']:
            file['target'] = convert_data(file['target'])
