import json
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

t = "test"


def augment_data(file):
    aug = iaa.Affine(scale=(0.5, 3),
                     translate_percent=(-0.05, 0.05),
                     rotate=(-45, 45),
                     shear=(-20, 20),
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


train_data = r'D:\python_project\breg_graph\tmp\dataset\{}.json'.format(t)
save_data = r'D:\python_project\breg_graph\tmp\dataset3\{}.json'.format(t)

with open(train_data, 'r', encoding='utf-8') as f:
    data = json.loads("".join(f.readline()))

for item in data:
    for i in range(len(item['file'])):
        item['file'][i] = augment_data(item['file'][i])

with open(save_data, 'w', encoding='utf-8') as f:
    f.write(json.dumps(data))
