import os
import json

import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage


def augment_data(file):
    aug = iaa.Affine(rotate=(-30, 30),
                     shear=(-30, 30),
                     fit_output=True)
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


def convert24point(bbox):
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


data_path = r'C:\Users\thinhtq\Downloads\DATA_GENOCR_DKKD\DATA_GENOCR_DKKD'
for folder in os.listdir(data_path):
    data = []
    for file in os.listdir(os.path.join(data_path, folder)):
        if "json" in file:
            file_name = file.split(".")[0]
            w, h = 0, 0
            for item in os.listdir(os.path.join(data_path, folder, file_name)):
                image = cv2.imread(os.path.join(data_path, folder, file_name, item))
                h1, w1, c1 = image.shape
                w += w1
                h += h1
            with open(os.path.join(data_path, folder, file), 'r', encoding='utf-8') as f:
                item_data = json.loads(f.read())
            for item in item_data:
                item['bbox'] = convert24point(item['bbox'])

            data.append({
                "file_name": file,
                "shape": (h, w),
                "target": item_data
            })
    with open(os.path.join("{}.json".format(folder)), 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=4))
