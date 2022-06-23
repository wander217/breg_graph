import json
import os.path

import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage


# def convert_label(old_label):
#     old_labels: dict = {
#         "LABEL_OWNER": "label_representative",
#         "OWNER_NAME": "representative_name",
#         "OWNER_SEX": "representative_sex",
#         "OWNER_BIRTHDAY": "representative_birthday",
#         "OWNER_ETHNICITY": "representative_ethnicity",
#         "OWNER_NATION": "representative_nation",
#         "OWNER_IDCARD_TYPE": "representative_type",
#         "OWNER_IDCARD_NUMBER": "representative_idcard_number",
#         "OWNER_IDCARD_DATE": "representative_idcard_date",
#         "OWNER_IDCARD_PLACE": "representative_idcard_place",
#         "OWNER_RESIDENCE_PERMANENT": "representative_residence_permanent",
#         "OWNER_LIVING_PLACE": "representative_living_place",
#         "LABEL_REPRESENTATIVE_OFFICE": "label_business_place",
#         "REPRESENTATIVE_COMPANY_NAME": "business_place_name",
#         "REPRESENTATIVE_COMPANY_ADDRESS": "business_place_address",
#         "REPRESENTATIVE_COMPANY_CODE": "business_place_code",
#         "LABEL_BRANCH": "label_business_place",
#         "BRANCH_COMPANY_NAME": "business_place_name",
#         "BRANCH_COMPANY_ADDRESS": "business_place_address",
#         "BRANCH_COMPANY_CODE": "business_place_code",
#         "AUTHORITY_COMPANY_NAME": "business_place_name",
#         "AUTHORITY_COMPANY_CODE": "business_place_code",
#         "AUTHORITY_COMPANY_ADDRESS": "business_place_address",
#         "LABEL_SHAREHOLDER": "other"
#     }
#     return old_labels.get(old_label, old_label)


def augment_data(file):
    aug = iaa.Affine(rotate=-file['rotate'])
    # w, h = file['shape']
    # if file['rotate'] == 90 or file['rotate'] == -90:
    #     file['shape'] = [h, w]
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
            'label': target['label'],
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
    data1 = json.loads("".join(f.readlines()))
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

# for data in data1:
#     data = augment_data(data)

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

del stat[-1]
save_path = r'D:\python_project\breg_graph\tmp\ category'
for key, value in stat.items():
    print(len(value))
    # label = {}
    # with open(os.path.join(save_path, "{}.json".format(key)), 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(value))
    # for file in value:
    #     for target in file['file']:
    #         for bbox in target['target']:
    #             label[bbox['label']] = 1
    # del label['other']
    # with open(os.path.join(save_path, "label_{}.json".format(key)), 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(list(label.keys())))
    # item = value[20]
    # for file in item['file']:
    #     print(file)
    # break

with open("valid_data.json", 'w', encoding='utf-8') as f:
    f.write(json.dumps(stat, indent=4))
