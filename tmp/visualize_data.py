import json
import os.path

import cv2 as cv
import numpy as np

data_path = r"C:\Users\thinhtq\Downloads\test_result_data_empty.txt"
image_dir = r'D:\labelme_dn'
data = []
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        file, text, _ = line.strip().split("\t")
        print(file)
        file = file.split("/")[-1].split(".")[0]
        dir_name, id_code, image_name = file.split("__")
        image_path = os.path.join(image_dir, dir_name, image_name + ".jpg")
        if not os.path.isfile(image_path):
            image_path = os.path.join(image_dir, dir_name, image_name + ".png")
        image = cv.imread(image_path)
        with open(os.path.join(image_dir, dir_name, image_name + ".json")) as f:
            data = json.loads(f.read())
        polygon = data['shapes'][int(id_code)]['points']
        cv.polylines(image, [np.array(polygon).astype(np.int32)], True, (0, 255, 0), 2)
        cv.imwrite(os.path.join(r"D:\python_project\breg_graph\checked_text", "{}.jpg".format(i)), image)
        # cv.namedWindow("abc", cv.WINDOW_NORMAL)
        # cv.imshow("abc", image)
        # cv.waitKey(0)
