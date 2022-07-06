import copy
import json
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage


def augment_data(file):
    aug = iaa.Affine(scale=(0.5, 1),
                     rotate=(-30, 30),
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


for t in ["train", "valid", "test"]:
    train_data = r'D:\python_project\breg_graph\tmp\dataset\{}.json'.format(t)
    save_data = r'D:\python_project\breg_graph\tmp\aug_data_1\{}.json'.format(t)

    with open(train_data, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    aug_data = copy.deepcopy(data)
    for i in range(0):
        new_data = copy.deepcopy(data)
        for item in new_data:
            for j in range(len(item['file'])):
                item['file'][j] = augment_data(item['file'][j])
        aug_data.extend(new_data)

    with open(save_data, 'w', encoding='utf-8') as f:
        print(len(aug_data))
        f.write(json.dumps(aug_data))
