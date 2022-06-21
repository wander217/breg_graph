import json
import os
import numpy as np
import cv2 as cv

data_path = r'D:\python_project\breg_graph\save_text.json'
with open(data_path, 'r', encoding='utf-8') as f:
    new_data = json.loads(f.readline())

remove_empty_path = r'C:\Users\thinhtq\Downloads\test_result_data_empty.txt'
with open(remove_empty_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        path, text, _ = line.split("\t")
        key = path.split("/")[-1].split(".")[0]
        new_data[key] = text

# with open(r"D:\python_project\breg_graph\save_text1.json", 'w', encoding='utf-8') as f:
#     f.write(json.dumps(new_data, indent=4))

count = 0
save_data = []
object_path = r'D:\python_project\breg_graph\tmp\convert_data'
for dirname in os.listdir(object_path):
    for file in os.listdir(os.path.join(object_path, dirname)):
        tmp = os.path.join(object_path, dirname, file)
        if "json" in file:
            with open(tmp, 'r', encoding='utf-8') as f:
                data = json.loads("".join(f.readlines()))
            item_data = {
                "file_name": data['imagePath'].split("/")[-1],
                "folder": data['imagePath'].split("/")[-2],
                "shape": [data['imageWidth'], data['imageHeight']],
                "target": []
            }
            document = None
            for i, item in enumerate(data['shapes']):
                if item['label'] == 'document':
                    document = np.array(item['points']).astype(np.int32).reshape((-1, 2))
            x_min = np.min(document[:, 0])
            x_max = np.max(document[:, 0])
            y_min = np.min(document[:, 1])
            y_max = np.max(document[:, 1])
            item_data['shape'] = [int(x_max - x_min + 1), int(y_max - y_min + 1)]
            for i, item in enumerate(data['shapes']):
                if item['label'] == 'document':
                    continue
                text = "{}__{}__{}".format(dirname, i, file.split(".")[0])
                item_data['target'].append({
                    "bbox": (cv.boxPoints(cv.minAreaRect(np.array(item['points']).astype(np.int32).reshape((-1, 2)))) - np.array([x_min, y_min])).tolist(),
                    "label": item['label'],
                    "text": new_data[text]
                })
            save_data.append(item_data)
with open(r"D:\python_project\breg_graph\convert_data.json", 'w', encoding='utf-8') as f:
    f.write(json.dumps(save_data, indent=4))
