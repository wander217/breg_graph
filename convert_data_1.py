import json

import cv2
import numpy as np
import os
import pandas as pd
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

# all_item = []
# data_dir = r'C:\Users\thinhtq\Downloads\SROIE_2019-20220511T021937Z-001\SROIE_2019\interim'
# for data_path in os.listdir(data_dir):
#     data = pd.read_csv(os.path.join(data_dir, data_path))
#     data = data.fillna("other")
#     new_data = []
#     for i, row in data.iterrows():
#         tmp = row.tolist()
#         new_data.append({
#             "bbox": np.array(tmp[:8], dtype=np.float32).reshape((-1, 2)).tolist(),
#             "text": tmp[8],
#             "label": tmp[9]
#         })
#     all_item.append({
#         "file_name": data_path.split(".")[0] + ".jpg",
#         "target": new_data
#     })
# save_path = r'D:\python_project\breg_graph\data\data3\total.json'
# with open(save_path, 'w', encoding='utf-8') as f:
#     f.write(json.dumps(all_item))

# data_path = r'D:\python_project\breg_graph\data\data3\total.json'
# with open(data_path, 'r', encoding='utf-8') as f:
#     data = json.loads(f.readline())
# train_len = int(0.9 * len(data))
# train_data = data[:train_len]
# test_data = data[train_len:]
# save_path = r"D:\python_project\breg_graph\data\data3"
# with open(os.path.join(save_path, "train.json"), 'w', encoding='utf-8') as f:
#     f.write(json.dumps(train_data))
# with open(os.path.join(save_path, "test.json"), 'w', encoding='utf-8') as f:
#     f.write(json.dumps(test_data))

# train_data_path = r'D:\python_project\breg_graph\data\data3\test.json'
# with open(train_data_path, 'r', encoding='utf-8') as f:
#     data = json.loads(f.readline())
#
#
# def augment(datas: list, shape):
#     aug = iaa.Affine(scale=(0.5, 2),
#                      translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
#                      rotate=(-5, 5),
#                      shear=(-5, 5),
#                      fit_output=True).to_deterministic()
#     for item in datas:
#         key_points = [Keypoint(point[0], point[1])
#                       for point in item['bbox']]
#         new_key_points = aug.augment_keypoints([
#             KeypointsOnImage(key_points, shape=tuple(shape))
#         ])[0].keypoints
#         new_bbox = [(float(key_point.x), float(key_point.y))
#                     for key_point in new_key_points]
#         item['bbox'] = new_bbox
#     return datas
#
#
# new_data = []
# print(len(data))
# image_path = r'C:\Users\thinhtq\Downloads\SROIE_2019-20220511T021937Z-001\SROIE_2019\raw\img'
# for item in data:
#     try:
#         image = cv2.imread(os.path.join(image_path, item['file_name']))
#         if image is None:
#             continue
#     except Exception as e:
#         continue
#     new_data.append(item)
#     for i in range(5):
#         new_data.append({
#             "file_name": item['file_name'],
#             "target": augment(item['target'], image.shape)
#         })
# save_data = r'D:\python_project\breg_graph\data\data3\test1.json'
# with open(save_data, 'w', encoding='utf-8') as f:
#     f.write(json.dumps(new_data))

# train1 = r'D:\python_project\breg_graph\data\data3\train1.json'
# with open(train1, 'r', encoding='utf-8') as f:
#     data = json.loads(f.readline())
# train_len = 2976
# train_data = data[:train_len]
# valid_data = data[train_len:]
# receipt_path = r'D:\python_project\breg_graph\data\receipt_data'
# with open(os.path.join(receipt_path, "train.json"), 'w', encoding='utf-8') as f:
#     f.write(json.dumps(train_data))
# with open(os.path.join(receipt_path, "valid.json"), 'w', encoding='utf-8') as f:
#     f.write(json.dumps(valid_data))

train_path = r'D:\python_project\breg_graph\data\receipt_data\train.json'
with open(train_path, 'r', encoding='utf-8') as f:
    datas = json.loads(f.readline())

label = {}
for data in datas:
    for item in data['target']:
        label[item['label']] = 1

print(label.keys())