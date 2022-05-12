# import cv2 as cv
# import os
#
# start_1 = 2906
# end_1 = 3190
#
# start_2 = 2691
#
# image_path = r'C:\Users\thinhtq\Downloads\image\image'
#
# for i in range(start_1, end_1 + 1):
#     image = cv.imread(os.path.join(image_path, "{}_1.jpg".format(i)))
#     cv.imwrite(os.path.join(image_path, "{}_2.jpg".format(start_2)), image)
#     start_2 += 1

import json

target_path = r'D:\\data1.json'
new_data = []
with open(target_path, 'r', encoding='utf-8') as f:
    datas = json.loads(f.readline())
    for data in datas:
        anno_list = []
        print(data['file_upload'])
        for anno in data['annotations']:
            results = anno['result']
            filter = {}
            for result in results:
                if not result['id'] in filter:
                    filter[result['id']] = [result['value']]
                else:
                    filter[result['id']].append(result['value'])
            for key, values in filter.items():
                item_data = {}
                for value in values:
                    if 'labels' in value:
                        item_data['label'] = value['labels'][0]
                    elif 'text' in value:
                        item_data['text'] = value['text'][0]
                    else:
                        item_data['bbox'] = value['points']
                anno_list.append(item_data)
        new_data.append({
            "file_name": data['file_upload'].split("-")[1],
            "target_file": anno_list
        })
print(len(new_data))
save_path = r'D:\python_project\breg_graph\data\data2\data.json'
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(new_data))
