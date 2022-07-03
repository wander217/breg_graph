import json
import os
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

root = r'D:\python_project\breg_graph\TMP1\dataset'
aug_data = r'D:\python_project\breg_graph\TMP1\aug_data'


def augment_data(file, aug):
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


for item in os.listdir(root):
    new_data = []
    aug = iaa.Affine(scale=(0.5, 1),
                     rotate=(-5, 5),
                     shear=(-5, 5),
                     fit_output=True)
    for i in range(20):
        with open(os.path.join(root, item), 'r', encoding='utf-8') as f:
            data = json.loads(f.read())
        for item1 in data:
            if i != 0:
                new_data.append(augment_data(item1, aug))
            else:
                new_data.append(item1)
    with open(os.path.join(aug_data, item), 'w', encoding='utf-8') as f:
        print(len(new_data))
        f.write(json.dumps(new_data, indent=4))
