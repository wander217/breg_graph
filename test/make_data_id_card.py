import copy
import json
import os.path

import numpy as np

from utils.generator import make_label, convert, translate_coordinate

# make_label([
#     "NUMBER",
#     "FULLNAME",
#     "BIRTHDAY",
#     "FULLNAME",
#     "ORIGINAL_RESIDENCE",
#     "REGISTRATION_RESIDENCE"
# ], r'D:\workspace\project\dkkd_graph\asset\idcard\label.json')
convert(r"D:\workspace\project\dkkd_graph\data\idcard\data.json",
        r"D:\workspace\project\dkkd_graph\data\idcard\total.json")
data_path = r"D:\workspace\project\dkkd_graph\data\idcard\total.json"
save_path = r"D:\workspace\project\dkkd_graph\data\idcard\split"
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.loads(f.readline())
train = data[:30]
with open(os.path.join(save_path, "train.json"), 'w', encoding='utf-8') as f:
    for i in range(len(train)):
        for bbox in train[i]['target']:
            tmp = np.array(bbox["bbox"])
            x_min = np.min(tmp[:, 0])
            x_max = np.max(tmp[:, 0])
            y_min = np.min(tmp[:, 1])
            y_max = np.max(tmp[:, 1])
            new_bbox = [
                [x_min, y_min], [x_max, y_min],
                [x_max, y_max], [x_min, y_max]
            ]
            bbox['bbox'] = new_bbox
        new_item = copy.deepcopy(train[i])
        new_item["target"] = translate_coordinate(new_item["target"])
        train.append(new_item)
    print(len(train))
    f.write(json.dumps(train))
valid = data[30:40]
with open(os.path.join(save_path, "valid.json"), 'w', encoding='utf-8') as f:
    for i in range(len(valid)):
        for bbox in valid[i]['target']:
            tmp = np.array(bbox["bbox"])
            x_min = np.min(tmp[:, 0])
            x_max = np.max(tmp[:, 0])
            y_min = np.min(tmp[:, 1])
            y_max = np.max(tmp[:, 1])
            new_bbox = [
                [x_min, y_min], [x_max, y_min],
                [x_max, y_max], [x_min, y_max]
            ]
            bbox['bbox'] = new_bbox
        new_item = copy.deepcopy(valid[i])
        new_item["target"] = translate_coordinate(new_item["target"])
        valid.append(new_item)
    print(len(valid))
    f.write(json.dumps(valid))
test = data[40:]
with open(os.path.join(save_path, "test.json"), 'w', encoding='utf-8') as f:
    for i in range(len(test)):
        for bbox in test[i]['target']:
            tmp = np.array(bbox["bbox"])
            x_min = np.min(tmp[:, 0])
            x_max = np.max(tmp[:, 0])
            y_min = np.min(tmp[:, 1])
            y_max = np.max(tmp[:, 1])
            new_bbox = [
                [x_min, y_min], [x_max, y_min],
                [x_max, y_max], [x_min, y_max]
            ]
            bbox['bbox'] = new_bbox
        new_item = copy.deepcopy(test[i])
        new_item["target"] = translate_coordinate(new_item["target"])
        test.append(new_item)
    print(len(test))
    f.write(json.dumps(test))
