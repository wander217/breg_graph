import os
import cv2 as cv

image_path = r'D:\text_crop_image'
wrong_path = r'D:\python_project\breg_graph\checked_data\wrong.txt'
save_path = r'D:\python_project\breg_graph\checked_data\wrong_image'

data = []
with open(wrong_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        data.append(line.strip())
for item in data:
    try:
        image = cv.imread(os.path.join(image_path, item))
        cv.imwrite(os.path.join(save_path, item), image)
    except Exception as e:
        print(item)



