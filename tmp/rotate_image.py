import json
import os
import cv2 as cv
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

rotate_path = r'D:\workspace\project\breg_graph\tmp\rotate.json'
with open(rotate_path, 'r', encoding='utf-8') as f:
    rotate = json.loads("".join(f.readlines()))

data_path = r'D:\labelme_dn'
save_path = r'D:\python_project\breg_graph\tmp\convert_data'

def augment_data(file, aug, rotate):
    new_target = []
    for target in file['shapes']:
        keypoint = KeypointsOnImage([
            Keypoint(x=point[0], y=point[1])
            for point in target['points']],
            shape=(file['imageHeight'], file['imageWidth']))
        aug = aug.to_deterministic()
        new_keypoint = aug.augment_keypoints(keypoint).keypoints
        new_target.append({
            **target,
            'points': [(int(point.x), int(point.y))
                     for point in new_keypoint]
        })
    file['shapes'] = new_target
    if rotate == 90 or rotate == -90:
        file['imageWidth'], file['imageHeight'] = file['imageHeight'], file['imageWidth']
    return file


for folder in os.listdir(data_path):
    for file in os.listdir(os.path.join(data_path, folder)):
        if file.endswith("png") or file.endswith("jpg"):
            image = cv.imread(os.path.join(data_path, folder, file))
            tmp = os.path.join(data_path, folder, file.split(".")[0] + ".json")
            with open(tmp, 'r', encoding='utf-8') as f:
                data = json.loads("".join(f.readlines()))
            angle = 0
            for item in rotate[folder]:
                if file in item:
                    angle = item[file]
            aug = iaa.Sequential(iaa.Affine(rotate=-angle))
            aug = aug.to_deterministic()
            new_image = aug.augment_image(image)
            data = augment_data(data, aug, angle)
            if not os.path.isdir(os.path.join(save_path, folder)):
                os.mkdir(os.path.join(save_path, folder))
            cv.imwrite(os.path.join(save_path, folder, file), new_image)
            with open(os.path.join(save_path, folder, file.split(".")[0] + ".json"), 'w', encoding='utf-8') as f:
                f.write(json.dumps(data))
