import json
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage


def augment_data(file):
    aug = iaa.Affine(rotate=-file['rotate'])
    w, h = file['shape']
    if file['rotate'] == 90 or file['rotate'] == -90:
        file['shape'] = [h, w]
    new_target = []
    for target in file['target']:
        keypoint = KeypointsOnImage([
            Keypoint(x=point[0], y=point[1])
            for point in target['bbox']],
            shape=tuple(file['shape']))
        aug = aug.to_deterministic()
        new_keypoint = aug.augment_keypoints(keypoint).keypoints
        new_target.append({
            **target,
            'bbox': [(int(point.x), int(point.y))
                     for point in new_keypoint]
        })
    file['target'] = new_target
    return file


data_path = r'D:\python_project\breg_graph\tmp\category1.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.loads("".join(f.readlines()))

convert_data = r'D:\python_project\breg_graph\convert_data.json'
with open(convert_data, 'r', encoding='utf-8') as f:
    data1 = json.loads(f.readline())
    for item in data1:
        tmp = data['file'][item['folder']][item['file_name'].split("_")[0]]
        item['type'] = tmp

# rotate_data = r'D:\python_project\breg_graph\tmp\rotate.json'
# with open(rotate_data, 'r', encoding='utf-8') as f:
#     data2 = json.loads("".join(f.readlines()))
#     for item in data1:
#         tmp = data2[item['folder']]
#         # item['rotate'] = 0
#         for item1 in tmp:
#             if item['file_name'] in item1:
#                 item['rotate'] = item1[item['file_name']]
#                 break

for data in data1:
    data = augment_data(data)

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

for key, value in stat.items():
    item = value[20]
    for file in item['file']:
        print(file)
    break

# with open("valid_data.json", 'w', encoding='utf-8') as f:
#     f.write(json.dumps(stat))
