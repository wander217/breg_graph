import json
import os
import cv2 as cv
import numpy as np

data_path = r'D:\python_project\breg_graph\tmp\convert_data'
save_path = r'D:\python_project\breg_graph\tmp\rotate_check_data'

for folder in os.listdir(data_path):
    for file in os.listdir(os.path.join(data_path, folder)):
        if file.endswith("png") or file.endswith('jpg'):
            image = cv.imread(os.path.join(data_path, folder, file))
            tmp = os.path.join(data_path, folder, file.split('.')[0]+'.json')
            with open(tmp, 'r', encoding='utf-8') as f:
                data = json.loads(f.readline())
            for item in data['shapes']:
                cv.polylines(image, [np.array(item['points']).astype(np.int32)],True, (255, 0, 0))
            cv.imwrite(os.path.join(save_path, file), image)
            # cv.imshow("abc", image)
            # cv.waitKey(0)
