import copy
import json
import math
import os.path
import random
import numpy as np
from tqdm import tqdm
from typing import List
from utils import remove_space
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage


def convert(data_path, save_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        datas = json.loads(f.readline())
    new_data = []
    for i in tqdm(range(len(datas))):
        data = datas[i]
        width, height = 0, 0
        file_name = '-'.join(data['file_upload'].split("-")[1:])
        polygons = []
        results = data['annotations'][0]['result']
        points, texts, labels = [], [], []
        for i in range(len(results)):
            value = results[i]['value']
            width = max([width, results[i]['original_width']])
            height = max([height, results[i]['original_height']])
            if 'labels' not in value and 'text' not in value:
                points.append(value['points'])
            if 'labels' in value:
                labels.append(value['labels'])
            if 'text' in value:
                texts.append(value['text'])
        for point, label, text in zip(points, labels, texts):
            polygons.append({
                "bbox": point,
                "label": label[0],
                "text": text[0]
            })
        if len(polygons) == 0:
            continue
        new_data.append({
            "file_name": file_name,
            "shape": [width, height],
            "target": polygons
        })
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def make_label(label: List, save_path: str):
    print("Total label:", len(label))
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(label))


def convert_label(label: str):
    if label == "Ngành nghề kinh doanh":
        return "BUSINESS_KIND"
    if label == "Loại hợp đồng":
        return "CONTRACT_TYPE"
    if label == "Số hợp đồng":
        return "CONTRACT_NUMBER"
    if label == "Tên hộ kinh doanh":
        return "COMPANY_NAME"
    if label == "Địa điểm kinh doanh":
        return "COMPANY_ADDRESS"
    if label == "Điện thoại":
        return "COMPANY_PHONE"
    if label == "Fax":
        return "COMPANY_FAX"
    if label == "Email/Website":
        return "COMPANY_EMAIL/WEBSITE"
    if label == "Vốn kinh doanh":
        return "BUSINESS_CAPITAL"
    if label == "Họ và tên chủ hộ":
        return "REPRESENTATIVE_NAME"
    if label == "Giới tính chủ hộ":
        return "REPRESENTATIVE_SEX"
    if label == "Ngày sinh chủ hộ":
        return "REPRESENTATIVE_BIRTHDAY"
    if label == "Dân tộc chủ hộ":
        return "REPRESENTATIVE_MAJORITY"
    if label == "Quốc tịch của chủ hộ":
        return "REPRESENTATIVE_NATIONAL"
    if label == "Số CCCD/ chứng minh thư của chủ hộ":
        return "REPRESENTATIVE_IDCARD_NUMBER"
    if label == "Ngày cấp CDCD/CMT của chủ hộ":
        return "REPRESENTATIVE_IDCARD_DATE"
    if label == "Nơi cấp CCCD/CMT của chủ hộ":
        return "REPRESENTATIVE_IDCARD_PLACE"
    if label == "Nơi đăng ký hộ khẩu thường trú của chủ hộ":
        return "REPRESENTATIVE_PERMANENT_RESIDENCE"
    if label == "Chỗ ở hiện tại của chủ hộ":
        return "REPRESENTATIVE_LIVING_PLACE"
    # if label == "Thành viên góp vốn":
    #     return "MEMBER_INFORMATION"
    return "OTHER"


def bbox_sorted(bbox: List):
    bbox_matrix: np.ndarray = np.array(bbox)
    x_min: int = np.min(bbox_matrix[:, 0])
    x_max: int = np.max(bbox_matrix[:, 0])
    y_min: int = np.min(bbox_matrix[:, 1])
    y_max: int = np.max(bbox_matrix[:, 1])
    return np.array([
        [x_min, y_min], [x_max, y_min],
        [x_max, y_max], [x_min, y_max]
    ]).astype(np.int32)


def check(path: str, save_path: str):
    with open(path, 'r', encoding='utf-8') as f:
        datas = json.loads(f.readline())
    datas = sorted(datas, key=lambda x: x['file_name'])
    data_filer = dict()
    for i in range(len(datas)):
        name = datas[i]['file_name'].split(".")[0].split("-")[0]
        if name == "41S8004926_GPKD_T_N":
            data_filer[datas[i]['file_name']] = [i]
        elif name in data_filer:
            data_filer[name].append(i)
        else:
            data_filer[name] = [i]
    for item in datas:
        print(item['file_name'])
    print("")
    new_datas = []
    labels = {}
    for key, value in data_filer.items():
        shape = np.zeros((2,))
        targets = []
        for j, id in enumerate(value):
            data = datas[id]
            for target in data['target']:
                bbox: np.ndarray = bbox_sorted(target["bbox"]) + np.array([0, shape[1]])
                targets.append({
                    "bbox": bbox.tolist(),
                    "label": convert_label(target["label"]),
                    "text": target["text"]
                })
                labels[target['label']] = 1
            shape = shape + np.array([0 if j > 0 else data['shape'][0], data['shape'][1]])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        new_datas.append({
            "file_name": key,
            "shape": shape.tolist(),
            "target": targets
        })
    with open(save_path, 'w', encoding="utf-8") as f:
        f.write(json.dumps(new_datas))
    for item in new_datas:
        print(item['file_name'])
    print(len(new_datas))


def convert_date(timestamp):
    date = timestamp.strftime("%d/%m/%Y")
    return date


def generate_contract_number():
    result = ""
    for _ in range(2):
        result += str(random.randint(0, 9))
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result += random.choice(alphabet)
    for _ in range(7):
        result += str(random.randint(0, 10))
    return result


def cut_company(name: str):
    tmp = name
    tmp = tmp.split('"')
    if len(tmp) != 0:
        tmp = tmp[0]
    tmp = tmp.split("(")
    if len(tmp) != 0:
        tmp = tmp[0]
    return remove_space(tmp)


def group_data(datas):
    groups = {}
    for data in datas:
        label = data['label']
        if label in groups:
            groups[label].append(data)
        else:
            groups[label] = [data]
    return groups


def text_split(text: str, length: int):
    words = text.split(" ")
    first, second = "", ""
    for word in words:
        if len(first) < length:
            first = first + " " + word
        else:
            second = second + " " + word
    return first, second


def random_captial():
    number = ["220.000.000,00",
              "40.000.000,00",
              "55.000.000,00",
              "111.000.000,00",
              "77.000.000,00",
              "835.000.000,00",
              "5.000.000,00",
              "6.000.000,00",
              "13.000.000",
              "22.000.000,00",
              "221.000.000,00",
              "205.000.000",
              "111.000.000,00",
              "897.000.000,00",
              "1.000.000.000,00",
              "2.000.000.000,00",
              "2.300.000.000,00",
              "10.000.000"]
    character = ["Hai trăm hai mươi triệu đồng",
                 "Bốn mươi triệu chẵn",
                 "Năm năm triệu đồng",
                 "Một trăm mười một triệu đồng",
                 "Bảy bảy triệu đồng",
                 "Tám trăm ba mươi năm triệu",
                 "Năm triệu đồng chẵn",
                 "Sáu triệu đồng chẵn",
                 "Mười ba triệu đồng",
                 "Hai mươi hai triệu đồng",
                 "Hai trăm hai mươi mốt triệu đồng",
                 "Hai trăm linh năm triệu đồng",
                 "Một trăm mười một triệu đồng",
                 "Tám trăm chín bảy triệu",
                 "Một tỷ đồng chẵn",
                 "Hai tỷ đồng chẵn",
                 "Hai tỷ ba trăm triệu đồng",
                 "Mười triệu đồng"]
    number_id = random.randint(0, len(number) - 1)
    return number[number_id], character[number_id]


def translate_coordinate(datas: List, shape):
    aug = iaa.Affine(scale=(0.5, 2),
                     translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                     rotate=(-5, 5),
                     shear=(-5, 5),
                     fit_output=True).to_deterministic()
    for item in datas:
        key_points = [Keypoint(point[0], point[1])
                      for point in item['bbox']]
        new_key_points = aug.augment_keypoints([
            KeypointsOnImage(key_points, shape=tuple(shape))
        ])[0].keypoints
        new_bbox = [(float(key_point.x), float(key_point.y))
                    for key_point in new_key_points]
        item['bbox'] = new_bbox
    return datas


def get_char_len(bbox: np.ndarray, text: str):
    tmp = bbox.reshape((-1, 2))[:, 0]
    x_min, x_max = np.min(tmp), np.max(tmp)
    w = x_max - x_min
    text_len = len(text)
    char_len = w / text_len
    return char_len


def get_bbox(text: str, char_len: int, bbox: np.ndarray):
    tmp = bbox.reshape((-1, 2))[:, 0]
    x_min, x_max = np.min(tmp), np.max(tmp)
    tmp = bbox.reshape((-1, 2))[:, 1]
    y_min, y_max = np.min(tmp), np.max(tmp)
    h = y_max - y_min
    w = char_len * len(text)
    return [
        [x_min, y_min],
        [x_min + w - 1, y_min],
        [x_min + w - 1, y_min + h - 1],
        [x_min, y_min + h - 1]
    ]


def process_1(original_data, addition_data, save_path):
    new_data = []
    data_id: int = 0
    print(original_data[data_id]["file_name"])
    shape = original_data[data_id]["shape"]
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["REPRESENTATIVE_BIRTHDAY"])
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số: " + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][1]["text"] = selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        first, second = text_split(selected_data["COMPANY_ADDRESS"],
                                   len(copied_group["COMPANY_ADDRESS"][2]["text"]))
        targets.append(copied_group["COMPANY_ADDRESS"][0])
        char_len = get_char_len(np.array(copied_group["COMPANY_ADDRESS"][2]["bbox"]),
                                copied_group["COMPANY_ADDRESS"][2]["text"])
        copied_group["COMPANY_ADDRESS"][2]["text"] = first
        targets.append(copied_group["COMPANY_ADDRESS"][2])
        if len(second) != 0:
            copied_group["COMPANY_ADDRESS"][1]["text"] = second
            copied_group["COMPANY_ADDRESS"][1]["bbox"] = get_bbox(second, char_len,
                                                                  np.array(copied_group["COMPANY_ADDRESS"][1]["bbox"]))
            targets.append(copied_group["COMPANY_ADDRESS"][1])
        # Company phone
        copied_group["COMPANY_PHONE"][0]["text"] = "Điện thoại:" + selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Company fax
        copied_group["COMPANY_FAX"][0]["text"] = "Fax:" + selected_data["COMPANY_FAX"]
        targets.extend(copied_group["COMPANY_FAX"])
        # Company email/website
        copied_group["COMPANY_EMAIL/WEBSITE"][0]["text"] = "Email:" + selected_data["COMPANY_EMAIL/WEBSITE"]
        targets.extend(copied_group["COMPANY_EMAIL/WEBSITE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]["text"] = selected_data["BUSINESS_KIND"]
        copied_group["BUSINESS_KIND"][0]["label"] = "OTHER"
        targets.extend(copied_group["BUSINESS_KIND"])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][0]["text"] = "Vốn kinh doanh: {}./.".format(number)
        copied_group["BUSINESS_CAPITAL"][1]["text"] = character
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][1]["text"] = selected_data["REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "Giới tính:" + selected_data["REPRESENTATIVE_SEX"]
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Sinh ngày:" + selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative national
        copied_group["REPRESENTATIVE_NATIONAL"][0]["text"] = "Quốc tịch:" + selected_data["REPRESENTATIVE_NATIONAL"]
        targets.extend(copied_group["REPRESENTATIVE_NATIONAL"])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][1]["text"] = selected_data["REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][0]["text"] = "Ngày cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_DATE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][0]["text"] = "Nơi cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent resistance
        first, second = text_split(selected_data["REPRESENTATIVE_PERMANENT_RESIDENCE"],
                                   len(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"]))
        char_len = get_char_len(np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["bbox"]),
                                copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"])
        targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0])
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = first
        targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1])
        if len(second) != 0:
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][2]["text"] = second
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][2]["bbox"] = get_bbox(second, char_len,
                                                                                     np.array(copied_group[
                                                                                                  "REPRESENTATIVE_PERMANENT_RESIDENCE"][
                                                                                                  2]["bbox"]))
            targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][2])
        # Representative living palace
        first, second = text_split(selected_data["REPRESENTATIVE_LIVING_PLACE"],
                                   len(copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["text"]))
        char_len = get_char_len(np.array(copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["bbox"]),
                                copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["text"])
        targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][0])
        copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["text"] = first
        targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][1])
        if len(second) != 0:
            copied_group["REPRESENTATIVE_LIVING_PLACE"][2]["text"] = second
            copied_group["REPRESENTATIVE_LIVING_PLACE"][2]["bbox"] = get_bbox(second, char_len,
                                                                              np.array(copied_group[
                                                                                           "REPRESENTATIVE_LIVING_PLACE"][
                                                                                           2]["bbox"]))
            targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][2])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def process_2(original_data, addition_data, save_path):
    new_data = []
    data_id: int = 1
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    shape = original_data[data_id]['shape']
    print(groups["COMPANY_NAME"])
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số:" + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][1]["text"] = selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        first, second = text_split(selected_data["COMPANY_ADDRESS"],
                                   len(copied_group["COMPANY_ADDRESS"][2]["text"]))
        char_len = get_char_len(np.array(copied_group["COMPANY_ADDRESS"][2]["bbox"]),
                                copied_group["COMPANY_ADDRESS"][2]["text"])
        targets.append(copied_group["COMPANY_ADDRESS"][0])
        copied_group["COMPANY_ADDRESS"][2]["text"] = first
        targets.append(copied_group["COMPANY_ADDRESS"][2])
        if len(second) != 0:
            copied_group["COMPANY_ADDRESS"][1]["text"] = second
            copied_group["COMPANY_ADDRESS"][1]["bbox"] = get_bbox(second, char_len,
                                                                  np.array(copied_group["COMPANY_ADDRESS"][1]["bbox"]))
            targets.append(copied_group["COMPANY_ADDRESS"][1])
        # Company phone
        copied_group["COMPANY_PHONE"][0]["text"] = "Điện thoại:" + selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Company fax
        copied_group["COMPANY_FAX"][0]["text"] = "Fax:" + selected_data["COMPANY_FAX"]
        targets.extend(copied_group["COMPANY_FAX"])
        # Company email/website
        copied_group["COMPANY_EMAIL/WEBSITE"][0]["text"] = "Email:" + selected_data["COMPANY_EMAIL/WEBSITE"]
        targets.extend(copied_group["COMPANY_EMAIL/WEBSITE"])
        # Business kind
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][0]["text"] = "Vốn kinh doanh: {}./.".format(number)
        copied_group["BUSINESS_CAPITAL"][1]["text"] = character
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][0]["text"] = "5. Họ và tên đại diện hộ kinh doanh:  " + selected_data[
            "REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "Giới tính:" + selected_data["REPRESENTATIVE_SEX"]
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Sinh ngày:" + selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative national
        copied_group["REPRESENTATIVE_NATIONAL"][0]["text"] = "Quốc tịch:" + selected_data["REPRESENTATIVE_NATIONAL"]
        targets.extend(copied_group["REPRESENTATIVE_NATIONAL"])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][1]["text"] = selected_data["REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][0]["text"] = "Ngày cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_DATE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][0]["text"] = "Nơi cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent resistance
        first, second = text_split(selected_data["REPRESENTATIVE_PERMANENT_RESIDENCE"], 53)
        char_len = get_char_len(np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["bbox"]),
                                "-" * 53)
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["text"] = "Nơi đăng ký HKTT:" + first
        targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0])
        if len(second) != 0:
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = second
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["bbox"] = get_bbox(second, char_len,
                                                                                     np.array(copied_group[
                                                                                                  "REPRESENTATIVE_PERMANENT_RESIDENCE"
                                                                                              ][1]["bbox"]))
            targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1])
        # Representative living palace
        first, second = text_split(selected_data["REPRESENTATIVE_LIVING_PLACE"], 50)
        char_len = get_char_len(np.array(copied_group[
                                             "REPRESENTATIVE_PERMANENT_RESIDENCE"
                                         ][0]["bbox"]),
                                "-" * 50)
        copied_group["REPRESENTATIVE_LIVING_PLACE"][0]["text"] = "Chỗ ở hiện tại: " + first
        targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][0])
        if len(second) != 0:
            copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["text"] = second
            copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["bbox"] = get_bbox(second, char_len,
                                                                              np.array(copied_group[
                                                                                           "REPRESENTATIVE_LIVING_PLACE"
                                                                                       ][1]["bbox"]))
            targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][1])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[1]["file_name"] + ".json"),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def process_3(original_data, addition_data, save_path):
    new_data = []
    data_id: int = 2
    shape = original_data[data_id]["shape"]
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["REPRESENTATIVE_BIRTHDAY"])
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][1]['text'] = generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][1]["text"] = selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        first, second = text_split(selected_data["COMPANY_ADDRESS"],
                                   len(copied_group["COMPANY_ADDRESS"][2]["text"]))
        char_len = get_char_len(np.array(copied_group["COMPANY_ADDRESS"][2]["bbox"]),
                                copied_group["COMPANY_ADDRESS"][2]["text"])
        targets.append(copied_group["COMPANY_ADDRESS"][0])
        copied_group["COMPANY_ADDRESS"][2]["text"] = first
        targets.append(copied_group["COMPANY_ADDRESS"][2])
        if len(second) != 0:
            copied_group["COMPANY_ADDRESS"][1]["text"] = second
            copied_group["COMPANY_ADDRESS"][1]["bbox"] = get_bbox(second, char_len,
                                                                  np.array(copied_group["COMPANY_ADDRESS"][1]["bbox"]))
            targets.append(copied_group["COMPANY_ADDRESS"][1])
        # Company phone
        copied_group["COMPANY_PHONE"][1]["text"] = selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Company fax
        copied_group["COMPANY_FAX"][0]["text"] = "Fax:" + selected_data["COMPANY_FAX"]
        targets.extend(copied_group["COMPANY_FAX"])
        # Company email/website
        copied_group["COMPANY_EMAIL/WEBSITE"][0]["text"] = "Email:" + selected_data["COMPANY_EMAIL/WEBSITE"]
        targets.extend(copied_group["COMPANY_EMAIL/WEBSITE"])
        # Business kind
        first, second = text_split(selected_data["BUSINESS_KIND"],
                                   len(copied_group["BUSINESS_KIND"][0]["text"]))
        char_len = get_char_len(np.array(copied_group["BUSINESS_KIND"][0]["bbox"]),
                                copied_group["BUSINESS_KIND"][0]["text"])
        copied_group["BUSINESS_KIND"][0]["text"] = first
        copied_group["BUSINESS_KIND"][0]["label"] = "OTHER"
        targets.append(copied_group["BUSINESS_KIND"][0])
        if len(second) != 0:
            copied_group["BUSINESS_KIND"][1]["text"] = second
            copied_group["BUSINESS_KIND"][1]["bbox"] = get_bbox(second, char_len,
                                                                np.array(copied_group["BUSINESS_KIND"][1]["bbox"]))
            copied_group["BUSINESS_KIND"][1]["label"] = "OTHER"
            targets.append(copied_group["BUSINESS_KIND"][1])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][1]["text"] = "{} đồng".format(number)
        copied_group["BUSINESS_CAPITAL"][2]["text"] = "(Bằng chữ: {}./.)".format(character)
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][1]["text"] = selected_data["REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][1]["text"] = selected_data["REPRESENTATIVE_SEX"]
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][1]["text"] = selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][1]["text"] = selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative national
        copied_group["REPRESENTATIVE_NATIONAL"][1]["text"] = selected_data["REPRESENTATIVE_NATIONAL"]
        targets.extend(copied_group["REPRESENTATIVE_NATIONAL"])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][1]["text"] = selected_data["REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][1]["text"] = selected_data["REPRESENTATIVE_IDCARD_DATE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][1]["text"] = selected_data["REPRESENTATIVE_IDCARD_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent resistance
        first, second = text_split(selected_data["REPRESENTATIVE_PERMANENT_RESIDENCE"],
                                   len(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"]))
        char_len = get_char_len(np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["bbox"]),
                                copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"])
        targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0])
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = first
        targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1])
        if len(second) != 0:
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][2]["text"] = second
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][2]["bbox"] = get_bbox(second, char_len,
                                                                                     np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][2]["bbox"]))
            targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][2])
        # Representative living palace
        first, second = text_split(selected_data["REPRESENTATIVE_LIVING_PLACE"],
                                   len(copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["text"]))
        char_len = get_char_len(np.array(copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["bbox"]),
                                copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["text"])
        targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][0])
        copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["text"] = first
        targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][1])
        if len(second) != 0:
            copied_group["REPRESENTATIVE_LIVING_PLACE"][2]["text"] = second
            copied_group["REPRESENTATIVE_LIVING_PLACE"][2]["bbox"] = get_bbox(second, char_len,
                                                                              np.array(copied_group["REPRESENTATIVE_LIVING_PLACE"][2]["bbox"]))
            targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][2])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def process_4(original_data, addition_data, save_path):
    new_data = []
    data_id = 3
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["CONTRACT_NUMBER"])
    shape = original_data[data_id]['shape']
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số:" + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][1]["text"] = selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        first, second = text_split(selected_data["COMPANY_ADDRESS"], 55)
        char_len = get_char_len(np.array(copied_group["COMPANY_ADDRESS"][0]["bbox"]),
                                copied_group["COMPANY_ADDRESS"][0]["text"])
        copied_group["COMPANY_ADDRESS"][0]["text"] = "2.Địa điểm kinh doanh:" + first
        targets.append(copied_group["COMPANY_ADDRESS"][0])
        if len(second) != 0:
            copied_group["COMPANY_ADDRESS"][1]["text"] = second
            copied_group["COMPANY_ADDRESS"][1]["bbox"] = get_bbox(second, char_len,
                                                                  np.array(copied_group["COMPANY_ADDRESS"][1]["bbox"]))
            targets.append(copied_group["COMPANY_ADDRESS"][1])
        # Company phone
        copied_group["COMPANY_PHONE"][0]["text"] = "Số điện thoại:" + selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Company email/website
        copied_group["COMPANY_EMAIL/WEBSITE"][0]["text"] = "Email/Website:" + selected_data["COMPANY_EMAIL/WEBSITE"]
        targets.extend(copied_group["COMPANY_EMAIL/WEBSITE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]['text'] = "3.Ngành nghề kinh doanh: {}.".format(selected_data["BUSINESS_KIND"])
        copied_group["BUSINESS_KIND"][0]['label'] = "OTHER"
        targets.extend(copied_group["BUSINESS_KIND"])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][0]["text"] = "Vốn kinh doanh: {} đồng".format(number)
        copied_group["BUSINESS_CAPITAL"][1]["text"] = "({})".format(character)
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][1]["text"] = selected_data["REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "Giới tính:" + selected_data["REPRESENTATIVE_SEX"]
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Ngày sinh:" + selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative national
        copied_group["REPRESENTATIVE_NATIONAL"][0]["text"] = "Quốc tịch:" + selected_data["REPRESENTATIVE_NATIONAL"]
        targets.extend(copied_group["REPRESENTATIVE_NATIONAL"])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][1]["text"] = "Số hộ khẩu/hộ chiếu:" + selected_data[
            "REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][0]["text"] = "Ngày cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_DATE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][0]["text"] = "Nơi cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent resistance
        first, second = text_split(selected_data["REPRESENTATIVE_PERMANENT_RESIDENCE"], 50)
        char_len = get_char_len(np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["bbox"]),
                                "-" * 50)
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["text"] = "Nơi đăng ký hộ khẩu thường trú: " + first
        targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0])
        if len(second) != 0:
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = second
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["bbox"] = get_bbox(second, char_len,
                                                                                     np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["bbox"]))
            targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1])
        # Representative living palace
        copied_group["REPRESENTATIVE_LIVING_PLACE"][0]["text"] = "Chỗ ở hiện tại:" + first
        targets.extend(copied_group["REPRESENTATIVE_LIVING_PLACE"])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def process_5(original_data, addition_data, save_path):
    new_data = []
    data_id = 4
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["CONTRACT_NUMBER"])
    shape = original_data[data_id]['shape']
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số:" + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][1]["text"] = selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        copied_group["COMPANY_ADDRESS"][0]["text"] = "2.Địa điểm kinh doanh:" + selected_data["COMPANY_ADDRESS"]
        targets.extend(copied_group["COMPANY_ADDRESS"])
        # Company phone
        copied_group["COMPANY_PHONE"][1]["text"] = selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Company fax
        copied_group["COMPANY_FAX"][0]["text"] = "Fax:" + selected_data["COMPANY_FAX"]
        targets.extend(copied_group["COMPANY_FAX"])
        # Company email/website
        copied_group["COMPANY_EMAIL/WEBSITE"][0]["text"] = "Email:" + selected_data["COMPANY_EMAIL/WEBSITE"]
        targets.extend(copied_group["COMPANY_EMAIL/WEBSITE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]['text'] = "3.Ngành nghề kinh doanh: {}.".format(selected_data["BUSINESS_KIND"])
        copied_group["BUSINESS_KIND"][0]['label'] = "OTHER"
        targets.extend(copied_group["BUSINESS_KIND"])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][0]["text"] = "Vốn kinh doanh: {} VND".format(number)
        copied_group["BUSINESS_CAPITAL"][1]["text"] = "({})".format(character)
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][0]["text"] = "5. Họ và tên đại diện hộ kinh doanh: " + selected_data[
            "REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][1]["text"] = selected_data["REPRESENTATIVE_SEX"]
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][1]["text"] = selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][1]["text"] = selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative national
        copied_group["REPRESENTATIVE_NATIONAL"][1]["text"] = selected_data["REPRESENTATIVE_NATIONAL"]
        targets.extend(copied_group["REPRESENTATIVE_NATIONAL"])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][1]["text"] = selected_data["REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][1]["text"] = selected_data["REPRESENTATIVE_IDCARD_DATE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][1]["text"] = selected_data["REPRESENTATIVE_IDCARD_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent resistance
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["text"] = "Nơi đăng ký hộ khẩu thường trú: " + \
                                                                        selected_data[
                                                                            "REPRESENTATIVE_PERMANENT_RESIDENCE"]
        targets.extend(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"])
        # Representative living palace
        copied_group["REPRESENTATIVE_LIVING_PLACE"][0]["text"] = "Chỗ ở hiện tại:" + selected_data[
            "REPRESENTATIVE_LIVING_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_LIVING_PLACE"])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def process_6(original_data, addition_data, save_path):
    new_data = []
    data_id = 5
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    # print(groups["MEMBER_INFORMATION"])
    shape = original_data[data_id]['shape']
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số:" + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][0]["text"] = "1. Tên hộ kinh doanh: " + selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        first, second = text_split(selected_data["COMPANY_ADDRESS"],
                                   len(copied_group["COMPANY_ADDRESS"][0]["text"]))
        char_len = get_char_len(np.array(copied_group["COMPANY_ADDRESS"][0]["bbox"]),
                                copied_group["COMPANY_ADDRESS"][0]["text"])
        copied_group["COMPANY_ADDRESS"][0]["text"] = "2. Địa điểm kinh doanh: " + first
        targets.append(copied_group["COMPANY_ADDRESS"][0])
        if len(second) != 0:
            copied_group["COMPANY_ADDRESS"][1]["text"] = second
            copied_group["COMPANY_ADDRESS"][1]["bbox"] = get_bbox(second, char_len,
                                                                  np.array(copied_group["COMPANY_ADDRESS"][1]["bbox"]))
            targets.append(copied_group["COMPANY_ADDRESS"][1])
        # Company phone
        copied_group["COMPANY_PHONE"][0]["text"] = "Điện thoại:" + selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Company fax
        copied_group["COMPANY_FAX"][0]["text"] = "Fax:" + selected_data["COMPANY_FAX"]
        targets.extend(copied_group["COMPANY_FAX"])
        # Company email/website
        copied_group["COMPANY_EMAIL/WEBSITE"][0]["text"] = "Email:" + selected_data["COMPANY_EMAIL/WEBSITE"]
        targets.extend(copied_group["COMPANY_EMAIL/WEBSITE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]['text'] = selected_data["BUSINESS_KIND"]
        copied_group["BUSINESS_KIND"][0]['label'] = "OTHER"
        targets.extend(copied_group["BUSINESS_KIND"])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][0]["text"] = "Vốn kinh doanh: {} đ ({})".format(number, character)
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][1]["text"] = selected_data["REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "Giới tính: " + selected_data["REPRESENTATIVE_SEX"]
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Sinh ngày:" + selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative national
        copied_group["REPRESENTATIVE_NATIONAL"][0]["text"] = "Quốc tịch:" + selected_data["REPRESENTATIVE_NATIONAL"]
        targets.extend(copied_group["REPRESENTATIVE_NATIONAL"])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][1]["text"] = selected_data["REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][0]["text"] = "Ngày cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_DATE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][0]["text"] = "Cơ quan cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent resistance
        first, second = text_split(selected_data["REPRESENTATIVE_PERMANENT_RESIDENCE"],
                                   len(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["text"]))
        char_len = get_char_len(np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["bbox"]),
                                copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["text"])
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["text"] = "Nơi đăng ký hộ khẩu thường trú: " + first
        targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0])
        if len(second) != 0:
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = second
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["bbox"] = get_bbox(second, char_len,
                                                                                     np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["bbox"]))
            targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1])
        # Representative living palace
        copied_group["REPRESENTATIVE_LIVING_PLACE"][0]["text"] = "Chỗ ở hiện tại:" + selected_data[
            "REPRESENTATIVE_LIVING_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_LIVING_PLACE"])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def process_7(original_data, addition_data, save_path):
    new_data = []
    data_id = 6
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["REPRESENTATIVE_IDCARD_NUMBER"])
    shape = original_data[data_id]['shape']
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số:" + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][1]["text"] = selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        copied_group["COMPANY_ADDRESS"][1]["text"] = selected_data["COMPANY_ADDRESS"]
        targets.extend(copied_group["COMPANY_ADDRESS"])
        # Company phone
        copied_group["COMPANY_PHONE"][1]["text"] = selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Company fax
        copied_group["COMPANY_FAX"][0]["text"] = "Fax:" + selected_data["COMPANY_FAX"]
        targets.extend(copied_group["COMPANY_FAX"])
        # Company email/website
        copied_group["COMPANY_EMAIL/WEBSITE"][0]["text"] = "Email:" + selected_data["COMPANY_EMAIL/WEBSITE"]
        targets.extend(copied_group["COMPANY_EMAIL/WEBSITE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]['text'] = selected_data["BUSINESS_KIND"]
        copied_group["BUSINESS_KIND"][0]['label'] = "OTHER"
        targets.extend(copied_group["BUSINESS_KIND"])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][1]["text"] = number + " đồng"
        copied_group["BUSINESS_CAPITAL"][2]["text"] = "({})".format(character)
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][1]["text"] = selected_data["REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "Giới tính: " + selected_data["REPRESENTATIVE_SEX"]
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Sinh ngày:" + selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative national
        copied_group["REPRESENTATIVE_NATIONAL"][0]["text"] = "Quốc tịch:" + selected_data["REPRESENTATIVE_NATIONAL"]
        targets.extend(copied_group["REPRESENTATIVE_NATIONAL"])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][2]["text"] = selected_data["REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][1]["text"] = selected_data["REPRESENTATIVE_IDCARD_DATE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][1]["text"] = selected_data["REPRESENTATIVE_IDCARD_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent resistance
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = selected_data[
            "REPRESENTATIVE_PERMANENT_RESIDENCE"]
        targets.extend(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"])
        # Representative living palace
        copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["text"] = selected_data["REPRESENTATIVE_LIVING_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_LIVING_PLACE"])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def process_8(original_data, addition_data, save_path):
    new_data = []
    data_id = 7
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["REPRESENTATIVE_IDCARD_NUMBER"])
    shape = original_data[data_id]['shape']
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số:" + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][1]["text"] = selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company fax
        copied_group["COMPANY_FAX"][0]["text"] = "Fax: " + selected_data["COMPANY_FAX"]
        targets.extend(copied_group["COMPANY_FAX"])
        # Company address
        copied_group["COMPANY_ADDRESS"][1]["text"] = selected_data["COMPANY_ADDRESS"]
        targets.extend(copied_group["COMPANY_ADDRESS"])
        # Company phone
        copied_group["COMPANY_PHONE"][1]["text"] = selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Company email
        copied_group["COMPANY_EMAIL/WEBSITE"][0]["text"] = "Email: " + selected_data["COMPANY_EMAIL/WEBSITE"]
        targets.extend(copied_group["COMPANY_EMAIL/WEBSITE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]["text"] = selected_data["BUSINESS_KIND"]
        copied_group["BUSINESS_KIND"][0]["label"] = "OTHER"
        targets.extend(copied_group["BUSINESS_KIND"])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][1]["text"] = number + " đồng"
        copied_group["BUSINESS_CAPITAL"][2]["text"] = "({})".format(character)
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][1]["text"] = selected_data["REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "-" + selected_data["REPRESENTATIVE_SEX"]
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Sinh ngày:" + selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative idcard number
        idcard_type = ["Căn cước công dân", "Chứng minh thư nhân dân"]
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][0]["text"] = "Loại giấy chứng thực cá nhân: {}".format(
            random.choice(idcard_type))
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][2]["text"] = selected_data["REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        tmp = selected_data["REPRESENTATIVE_IDCARD_DATE"]
        copied_group["REPRESENTATIVE_IDCARD_DATE"][1]["text"] = tmp
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][1]["text"] = selected_data["REPRESENTATIVE_IDCARD_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent resistance
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = selected_data[
            "REPRESENTATIVE_PERMANENT_RESIDENCE"]
        targets.extend(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"])
        # Representative living palace
        copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["text"] = selected_data["REPRESENTATIVE_LIVING_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_LIVING_PLACE"])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"),
              'w', encoding='utf-8') as f: \
            f.write(json.dumps(new_data))


def process_9(original_data, addition_data, save_path):
    new_data = []
    data_id = 8
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["REPRESENTATIVE_IDCARD_PLACE"])
    shape = original_data[data_id]['shape']
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số:" + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][1]["text"] = selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        copied_group["COMPANY_ADDRESS"][0]["text"] = "2. Địa điểm kinh doanh: " + selected_data["COMPANY_ADDRESS"]
        targets.extend(copied_group["COMPANY_ADDRESS"])
        # Company phone
        copied_group["COMPANY_PHONE"][0]["text"] = "Điện thoại:" + selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]["text"] = selected_data["BUSINESS_KIND"] + "(phải thực hiện đúng theo"
        copied_group["BUSINESS_KIND"][0]["label"] = "OTHER"
        targets.append(copied_group["BUSINESS_KIND"][0])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][0]["text"] = "4. Vốn kinh doanh: {} đồng ({})".format(number, character)
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][1]["text"] = selected_data["REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "-" + selected_data["REPRESENTATIVE_SEX"]
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        tmp = selected_data["REPRESENTATIVE_BIRTHDAY"].split("/")
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Sinh ngày:" + '-'.join(tmp)
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][0]["text"] = "Chứng minh nhân dân số:" + selected_data[
            "REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        tmp = selected_data["REPRESENTATIVE_IDCARD_DATE"].split("/")
        copied_group["REPRESENTATIVE_IDCARD_DATE"][0]["text"] = "Ngày cấp:" + "-".join(tmp)
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][0]["text"] = "Nơi cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent resistance
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = selected_data[
            "REPRESENTATIVE_PERMANENT_RESIDENCE"]
        targets.extend(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"])
        # Representative living palace
        copied_group["REPRESENTATIVE_LIVING_PLACE"][0]["text"] = "Chỗ ở hiện tại:" + selected_data[
            "REPRESENTATIVE_LIVING_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_LIVING_PLACE"])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def process_10(original_data, addition_data, save_path):
    new_data = []
    data_id = 9
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["REPRESENTATIVE_IDCARD_PLACE"])
    shape = original_data[data_id]['shape']
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số:" + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][0]["text"] = "1.Tên hộ kinh doanh:" + selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        copied_group["COMPANY_ADDRESS"][0]["text"] = "2. Địa điểm kinh doanh: " + selected_data["COMPANY_ADDRESS"]
        targets.extend(copied_group["COMPANY_ADDRESS"])
        # Company phone
        copied_group["COMPANY_PHONE"][0]["text"] = "Điện thoại:" + selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]["text"] = "3.Ngành nghề kinh doanh: {} (phải thực hiện đúng các quy".format(
            selected_data["BUSINESS_KIND"])
        copied_group["BUSINESS_KIND"][0]["label"] = "OTHER"
        targets.extend(copied_group["BUSINESS_KIND"])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][0]["text"] = "4. Vốn kinh doanh: {} đồng ({})".format(number, character)
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][1]["text"] = selected_data["REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "-" + selected_data["REPRESENTATIVE_SEX"]
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Sinh ngày:" + selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][0]["text"] = "Chứng minh nhân dân số:" + selected_data[
            "REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][0]["text"] = "Ngày cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_DATE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][0]["text"] = "Nơi cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent resistance
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = selected_data[
            "REPRESENTATIVE_PERMANENT_RESIDENCE"]
        targets.extend(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"])
        # Representative living palace
        copied_group["REPRESENTATIVE_LIVING_PLACE"][0]["text"] = "Chỗ ở hiện tại:" + selected_data[
            "REPRESENTATIVE_LIVING_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_LIVING_PLACE"])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def process_12(original_data, addition_data, save_path):
    new_data = []
    data_id = 11
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["REPRESENTATIVE_IDCARD_PLACE"])
    shape = original_data[data_id]['shape']
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số:" + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][1]["text"] = selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        copied_group["COMPANY_ADDRESS"][0]["text"] = "2. Địa điểm kinh doanh: " + selected_data["COMPANY_ADDRESS"]
        targets.extend(copied_group["COMPANY_ADDRESS"])
        # Company phone
        copied_group["COMPANY_PHONE"][0]["text"] = "Điện thoại:" + selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]["text"] = "3.Ngành nghề kinh doanh: " + selected_data[
            "BUSINESS_KIND"] + " (Kinh doanh không lấn chiếm lòng"
        copied_group["BUSINESS_KIND"][0]["label"] = "OTHER"
        targets.append(copied_group["BUSINESS_KIND"][0])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][0]["text"] = "4. Vốn kinh doanh: {} VNĐ  ({})".format(number, character)
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][1]["text"] = selected_data["REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "Giới tính:" + selected_data["REPRESENTATIVE_SEX"]
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Sinh ngày:" + selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][0]["text"] = "Chứng minh nhân dân số:" + selected_data[
            "REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][0]["text"] = "Ngày cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_DATE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][0]["text"] = "Nơi cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent resistance
        first, second = text_split(selected_data["REPRESENTATIVE_PERMANENT_RESIDENCE"], 45)
        char_len = get_char_len(np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["bbox"]),
                                "-" * 45)
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["text"] = "Nơi đăng ký hộ khẩu thường trú:" + first
        targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0])
        if len(second) != 0:
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = second
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["bbox"] = get_bbox(second, char_len,
                                                                                     np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["bbox"]))
            targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1])
        # Representative living palace
        first, second = text_split(selected_data["REPRESENTATIVE_LIVING_PLACE"], 40)
        copied_group["REPRESENTATIVE_LIVING_PLACE"][0]["text"] = "Chỗ ở hiện tại:" + first
        targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][0])
        if len(second) != 0:
            copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["text"] = second
            targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][1])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"), 'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def process_11(original_data, addition_data, save_path):
    new_data = []
    data_id = 10
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["REPRESENTATIVE_IDCARD_PLACE"])
    shape = original_data[data_id]['shape']
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số:" + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][0]["text"] = "1.Tên hộ kinh doanh:" + selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        copied_group["COMPANY_ADDRESS"][0]["text"] = "2. Địa điểm kinh doanh: " + selected_data["COMPANY_ADDRESS"]
        targets.extend(copied_group["COMPANY_ADDRESS"])
        # Company phone
        copied_group["COMPANY_PHONE"][0]["text"] = "Điện thoại:" + selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]["text"] = "3.Ngành,nghề kinh doanh: {} (phải thực hiện đúng các".format(
            selected_data["BUSINESS_KIND"])
        copied_group["BUSINESS_KIND"][0]["label"] = "OTHER"
        targets.extend(copied_group["BUSINESS_KIND"])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][0]["text"] = "4. Vốn kinh doanh: {} đồng ({})".format(number, character)
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][1]["text"] = selected_data["REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "-" + selected_data["REPRESENTATIVE_SEX"]
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Sinh ngày:" + selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][0]["text"] = "Chứng minh nhân dân số:" + selected_data[
            "REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][0]["text"] = "Ngày cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_DATE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][0]["text"] = "Nơi cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent resistance
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = selected_data[
            "REPRESENTATIVE_PERMANENT_RESIDENCE"]
        targets.extend(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"])
        # Representative living palace
        copied_group["REPRESENTATIVE_LIVING_PLACE"][0]["text"] = "Chỗ ở hiện tại:" + selected_data[
            "REPRESENTATIVE_LIVING_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_LIVING_PLACE"])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[10]["file_name"] + ".json"),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def process_13(original_data, addition_data, save_path):
    new_data = []
    data_id = 12
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["COMPANY_ADDRESS"])
    shape = original_data[data_id]['shape']
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số:" + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][0]['text'] = "1. Tên hộ kinh doanh:" + selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        copied_group["COMPANY_ADDRESS"][0]['text'] = "2. Địa điểm kinh doanh: " + selected_data["COMPANY_ADDRESS"]
        targets.extend(copied_group["COMPANY_ADDRESS"])
        # Company phone
        copied_group["COMPANY_PHONE"][0]['text'] = "Điện thoại: " + selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]["text"] = "3. Ngành,nghề kinh doanh: {} (chủ hộ kinh doanh phải thực".format(
            selected_data["BUSINESS_KIND"])
        copied_group["BUSINESS_KIND"][0]["label"] = "OTHER"
        targets.extend(copied_group["BUSINESS_KIND"])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][0]["text"] = "4. Vốn kinh doanh: {} đồng ({})".format(number, character)
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][0]["text"] = "5. Họ và tên hộ kinh doanh:"
        copied_group["REPRESENTATIVE_NAME"][1]["text"] = selected_data["REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "Giới tính:" + selected_data["REPRESENTATIVE_SEX"]
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Sinh ngày:" + selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative national
        copied_group["REPRESENTATIVE_NATIONAL"][0]["text"] = "Quốc tịch:" + selected_data["REPRESENTATIVE_NATIONAL"]
        targets.extend(copied_group["REPRESENTATIVE_NATIONAL"])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][0]["text"] = "Căn cước công dân số:" + selected_data[
            "REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][0]["text"] = "Ngày cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_DATE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][0]["text"] = "Nơi cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent resistance
        first, second = text_split(selected_data["REPRESENTATIVE_PERMANENT_RESIDENCE"], 40)
        char_len = get_char_len(np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["bbox"]),
                                "-" * 50)
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["text"] = "Nơi đăng ký hộ khẩu thường trú:" + first
        targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0])
        if len(second) != 0:
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = second
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["bbox"] = get_bbox(second, char_len,
                                                                                     np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["bbox"]))
            targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1])
        # Representative living palace
        first, second = text_split(selected_data["REPRESENTATIVE_LIVING_PLACE"], 40)
        char_len = get_char_len(np.array(copied_group["REPRESENTATIVE_LIVING_PLACE"][0]["bbox"]),
                                "-" * 40)
        copied_group["REPRESENTATIVE_LIVING_PLACE"][0]["text"] = "Chỗ ở hiện tại:" + first
        targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][0])
        if len(second) != 0:
            copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["text"] = second
            copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["bbox"] = get_bbox(second, char_len,
                                                                              np.array(copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["bbox"]))
            targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][1])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def process_14(original_data, addition_data, save_path):
    new_data = []
    data_id = 13
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["COMPANY_ADDRESS"])
    shape = original_data[data_id]['shape']
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số:" + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][0]["text"] = "1. Tên hộ kinh doanh:" + selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        copied_group["COMPANY_ADDRESS"][0]["text"] = "2. Địa điểm kinh doanh: " + selected_data["COMPANY_ADDRESS"]
        targets.extend(copied_group["COMPANY_ADDRESS"])
        # Company phone
        copied_group["COMPANY_PHONE"][0]["text"] = "Điện thoại:" + selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]["text"] = "3. Ngành,nghề kinh doanh: {} (chủ hộ phải thực hiện".format(
            selected_data["BUSINESS_KIND"])
        copied_group["BUSINESS_KIND"][0]["label"] = "OTHER"
        targets.extend(copied_group["BUSINESS_KIND"])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][0]["text"] = "4. Vốn kinh doanh: {} VNĐ ({})".format(number, character)
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][0]["text"] = "5. Họ và tên hộ kinh doanh:"
        copied_group["REPRESENTATIVE_NAME"][1]["text"] = selected_data["REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "Giới tính:" + selected_data["REPRESENTATIVE_SEX"]
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Sinh ngày:" + selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative national
        copied_group["REPRESENTATIVE_NATIONAL"][0]["text"] = "Quốc tịch:" + selected_data["REPRESENTATIVE_NATIONAL"]
        targets.extend(copied_group["REPRESENTATIVE_NATIONAL"])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][0]["text"] = "Chứng minh nhân dân số:" + selected_data[
            "REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][0]["text"] = "Ngày cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_DATE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][0]["text"] = "Nơi cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent resistance
        first, second = text_split(selected_data["REPRESENTATIVE_PERMANENT_RESIDENCE"], 40)
        char_len = get_char_len(np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["bbox"]),
                                "-" * 40)
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["text"] = "Nơi đăng ký hộ khẩu thường trú:" + first
        targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0])
        if len(second) != 0:
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = second
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["bbox"] = get_bbox(second, char_len,
                                                                                     np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["bbox"]))
            targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1])
        # Representative living palace
        first, second = text_split(selected_data["REPRESENTATIVE_LIVING_PLACE"], 40)
        char_len = get_char_len(np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["bbox"]),
                                "-" * 40)
        copied_group["REPRESENTATIVE_LIVING_PLACE"][0]["text"] = "Chỗ ở hiện tại:" + first
        targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][0])
        if len(second) != 0:
            copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["text"] = second
            copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["bbox"] = get_bbox(second, char_len,
                                                                              np.array(copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["bbox"]))
            targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][1])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def process_15(original_data, addition_data, save_path):
    new_data = []
    data_id = 14
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["COMPANY_PHONE"])
    shape = original_data[data_id]['shape']
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số: " + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][1]["text"] = selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        first, second = text_split(selected_data["COMPANY_ADDRESS"], 45)
        copied_group["COMPANY_ADDRESS"][0]["text"] = "2.Địa điểm kinh doanh:" + first
        targets.append(copied_group["COMPANY_ADDRESS"][0])
        if len(second) != 0:
            copied_group["COMPANY_ADDRESS"][1]["text"] = second
            targets.append(copied_group["COMPANY_ADDRESS"][1])
        # Company phone
        copied_group["COMPANY_PHONE"][0]["text"] = selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]["text"] = "3. Ngành nghề kinh doanh: " + selected_data[
            "BUSINESS_KIND"] + ".Bổ sung:"
        copied_group["BUSINESS_KIND"][0]["label"] = "OTHER"
        targets.extend(copied_group["BUSINESS_KIND"])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][0]["text"] = "4. Vốn kinh doanh: {} VNĐ ({})".format(number, character)
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][0]["text"] = "5. Họ và tên hộ kinh doanh:"
        copied_group["REPRESENTATIVE_NAME"][1]["text"] = selected_data["REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "Giới tính:" + selected_data["REPRESENTATIVE_SEX"]
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Sinh ngày:" + selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative national
        copied_group["REPRESENTATIVE_NATIONAL"][0]["text"] = "Quốc tịch:" + selected_data["REPRESENTATIVE_NATIONAL"]
        targets.extend(copied_group["REPRESENTATIVE_NATIONAL"])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][1]["text"] = selected_data[
            "REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][0]["text"] = "Ngày cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_DATE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][0]["text"] = "Nơi cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent resistance
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["text"] = "Nơi đăng ký hộ khẩu thường trú:" + \
                                                                        selected_data[
                                                                            "REPRESENTATIVE_PERMANENT_RESIDENCE"]
        targets.extend(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"])
        # Representative living palace
        first, second = text_split(selected_data["REPRESENTATIVE_LIVING_PLACE"], 40)
        char_len = get_char_len(np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["bbox"]),
                                "-" * 40)
        copied_group["REPRESENTATIVE_LIVING_PLACE"][0]["text"] = "Chỗ ở hiện tại:" + first
        targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][0])
        if len(second) != 0:
            copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["text"] = second
            copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["bbox"] = get_bbox(second, char_len,
                                                                              np.array(copied_group[
                                                                                           "REPRESENTATIVE_LIVING_PLACE"][
                                                                                           1]["bbox"]))
            targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][1])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def process_16(original_data, addition_data, save_path):
    new_data = []
    data_id = 15
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["COMPANY_ADDRESS"])
    shape = original_data[data_id]['shape']
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số:" + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][1]["text"] = selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        copied_group["COMPANY_ADDRESS"][0]["text"] = "2. Địa điểm kinh doanh: " + selected_data["COMPANY_ADDRESS"]
        targets.extend(copied_group["COMPANY_ADDRESS"])
        # Company phone
        copied_group["COMPANY_PHONE"][0]["text"] = "Điện thoại:" + selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]["text"] = "3. Ngành,nghề kinh doanh: {} (chủ hộ phải thực hiện".format(
            selected_data["BUSINESS_KIND"].capitalize())
        copied_group["BUSINESS_KIND"][0]["label"] = "OTHER"
        targets.extend(copied_group["BUSINESS_KIND"])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][0]["text"] = "4. Vốn kinh doanh: {} VNĐ ({})".format(number, character)
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][0]["text"] = "5. Họ và tên hộ kinh doanh:"
        copied_group["REPRESENTATIVE_NAME"][1]["text"] = selected_data["REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "Giới tính:" + selected_data["REPRESENTATIVE_SEX"]
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Sinh ngày:" + selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative national
        copied_group["REPRESENTATIVE_NATIONAL"][0]["text"] = "Quốc tịch:" + selected_data["REPRESENTATIVE_NATIONAL"]
        targets.extend(copied_group["REPRESENTATIVE_NATIONAL"])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][0]["text"] = "Chứng minh nhân dân số:" + selected_data[
            "REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][0]["text"] = "Ngày cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_DATE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][0]["text"] = "Nơi cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent resistance
        first, second = text_split(selected_data["REPRESENTATIVE_PERMANENT_RESIDENCE"], 40)
        char_len = get_char_len(np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["bbox"]),
                                "-" * 40)
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["text"] = "Nơi đăng ký hộ khẩu thường trú:" + first
        targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0])
        if len(second) != 0:
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = second
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["bbox"] = get_bbox(second, char_len,
                                                                                     np.array(copied_group[
                                                                                                  "REPRESENTATIVE_PERMANENT_RESIDENCE"][
                                                                                                  1]["bbox"]))
            targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1])
        # Representative living palace
        first, second = text_split(selected_data["REPRESENTATIVE_LIVING_PLACE"], 40)
        char_len = get_char_len(np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["bbox"]),
                                "-" * 40)
        copied_group["REPRESENTATIVE_LIVING_PLACE"][0]["text"] = "Chỗ ở hiện tại:" + first
        targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][0])
        if len(second) != 0:
            copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["text"] = second
            copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["bbox"] = get_bbox(second, char_len,
                                                                              np.array(copied_group[
                                                                                           "REPRESENTATIVE_LIVING_PLACE"][
                                                                                           1]["bbox"]))
            targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][1])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def process_17(original_data, addition_data, save_path):
    new_data = []
    data_id = 16
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["REPRESENTATIVE_PERMANENT_RESIDENCE"])
    shape = original_data[data_id]['shape']
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số:" + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][1]["text"] = "(ghi bằng chữ in hoa) " + selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        copied_group["COMPANY_ADDRESS"][1]["text"] = selected_data["COMPANY_ADDRESS"]
        targets.extend(copied_group["COMPANY_ADDRESS"])
        # Company phone
        copied_group["COMPANY_PHONE"][0]["text"] = "Điện thoại:" + selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Company fax
        copied_group["COMPANY_FAX"][0]["text"] = "Fax:" + selected_data["COMPANY_FAX"]
        targets.extend(copied_group["COMPANY_FAX"])
        # Company email
        copied_group["COMPANY_EMAIL/WEBSITE"][0]["text"] = "Email:" + selected_data["COMPANY_EMAIL/WEBSITE"]
        targets.extend(copied_group["COMPANY_EMAIL/WEBSITE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]["text"] = selected_data["BUSINESS_KIND"]
        copied_group["BUSINESS_KIND"][0]["label"] = "OTHER"
        targets.extend(copied_group["BUSINESS_KIND"])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][0]["text"] = "4. Vốn kinh doanh: {} đồng ({})".format(number, character)
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][1]["text"] = selected_data["REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "Giới tính:" + selected_data["REPRESENTATIVE_SEX"]
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Sinh ngày:" + selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative national
        copied_group["REPRESENTATIVE_NATIONAL"][0]["text"] = "Quốc tịch:" + selected_data["REPRESENTATIVE_NATIONAL"]
        targets.extend(copied_group["REPRESENTATIVE_NATIONAL"])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][2]["text"] = selected_data["REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][0]["text"] = "Ngày cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_DATE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][0]["text"] = "Nơi cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent resistance
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = selected_data[
            "REPRESENTATIVE_PERMANENT_RESIDENCE"]
        targets.extend(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"])
        # Representative living palace
        copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["text"] = selected_data["REPRESENTATIVE_LIVING_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_LIVING_PLACE"])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def process_18(original_data, addition_data, save_path):
    new_data = []
    data_id = 17
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["REPRESENTATIVE_PERMANENT_RESIDENCE"])
    shape = original_data[data_id]['shape']
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số:" + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][1]["text"] = "(ghi chữ in hoa) " + selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        copied_group["COMPANY_ADDRESS"][1]["text"] = selected_data["COMPANY_ADDRESS"]
        targets.extend(copied_group["COMPANY_ADDRESS"])
        # Company phone
        copied_group["COMPANY_PHONE"][0]["text"] = "Điện thoại:" + selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Company fax
        copied_group["COMPANY_FAX"][0]["text"] = "Fax:" + selected_data["COMPANY_FAX"]
        targets.extend(copied_group["COMPANY_FAX"])
        # Company email
        copied_group["COMPANY_EMAIL/WEBSITE"][0]["text"] = "Email:" + selected_data["COMPANY_EMAIL/WEBSITE"]
        targets.extend(copied_group["COMPANY_EMAIL/WEBSITE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]["text"] = selected_data["BUSINESS_KIND"]
        copied_group["BUSINESS_KIND"][0]["label"] = "OTHER"
        targets.extend(copied_group["BUSINESS_KIND"])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][0]["text"] = "4. Vốn kinh doanh: {} đồng ({})".format(number, character)
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][1]["text"] = selected_data["REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "Giới tính:" + selected_data["REPRESENTATIVE_SEX"]
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Sinh ngày:" + selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative national
        copied_group["REPRESENTATIVE_NATIONAL"][0]["text"] = "Quốc tịch:" + selected_data["REPRESENTATIVE_NATIONAL"]
        targets.extend(copied_group["REPRESENTATIVE_NATIONAL"])
        # Representative idcard number
        tmp = ["Loại giấy tờ chứng thực cá nhân: Giấy chứng minh thư nhân dân",
               "Loại giấy tờ chứng thực cá nhân: Căn cước công dân"]
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][0]["text"] = random.choice(tmp)
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][2]["text"] = selected_data["REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][0]["text"] = "Ngày cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_DATE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][0]["text"] = "Nơi cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent resistance
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = selected_data[
            "REPRESENTATIVE_PERMANENT_RESIDENCE"]
        targets.extend(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"])
        # Representative living palace
        copied_group["REPRESENTATIVE_LIVING_PLACE"][1]["text"] = selected_data["REPRESENTATIVE_LIVING_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_LIVING_PLACE"])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def process_19(original_data, addition_data, save_path):
    new_data = []
    data_id = 18
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["REPRESENTATIVE_PERMANENT_RESIDENCE"])
    shape = original_data[data_id]['shape']
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số:" + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][1]["text"] = selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        copied_group["COMPANY_ADDRESS"][0]["text"] = "2. Địa điểm kinh doanh:" + selected_data["COMPANY_ADDRESS"]
        targets.append(copied_group["COMPANY_ADDRESS"][0])
        # Company phone
        copied_group["COMPANY_PHONE"][0]["text"] = "Điện thoại:" + selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Company fax
        copied_group["COMPANY_FAX"][0]["text"] = "Fax:" + selected_data["COMPANY_FAX"]
        targets.extend(copied_group["COMPANY_FAX"])
        # Company email
        copied_group["COMPANY_EMAIL/WEBSITE"][1]["text"] = "Email: " + selected_data["COMPANY_EMAIL/WEBSITE"]
        targets.extend(copied_group["COMPANY_EMAIL/WEBSITE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]["text"] = "3. Ngành nghề kinh doanh:" + selected_data["BUSINESS_KIND"]
        copied_group["BUSINESS_KIND"][0]["label"] = "OTHER"
        targets.append(copied_group["BUSINESS_KIND"][0])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][0]["text"] = "4. Vốn kinh doanh: {}     đồng".format(number)
        targets.append(copied_group["BUSINESS_CAPITAL"][0])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][0]["text"] = "5. Họ và tên cá nhân: " + selected_data["REPRESENTATIVE_NAME"]
        targets.append(copied_group["REPRESENTATIVE_NAME"][0])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "({})".format(selected_data["REPRESENTATIVE_SEX"])
        targets.append(copied_group["REPRESENTATIVE_SEX"][0])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Sinh ngày:" + selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.append(copied_group["REPRESENTATIVE_BIRTHDAY"][0])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.append(copied_group["REPRESENTATIVE_MAJORITY"][0])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][1]["text"] = selected_data["REPRESENTATIVE_IDCARD_NUMBER"]
        targets.append(copied_group["REPRESENTATIVE_IDCARD_NUMBER"][0])
        targets.append(copied_group["REPRESENTATIVE_IDCARD_NUMBER"][1])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][0]["text"] = "Ngày cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_DATE"]
        targets.append(copied_group["REPRESENTATIVE_IDCARD_DATE"][0])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][0]["text"] = "Nơi cấp:" + selected_data[
            "REPRESENTATIVE_IDCARD_PLACE"]
        targets.append(copied_group["REPRESENTATIVE_IDCARD_PLACE"][0])
        # Representative permanent resistance
        first, second = text_split(selected_data["REPRESENTATIVE_PERMANENT_RESIDENCE"], 40)
        char_len = get_char_len(np.array(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["bbox"]),
                                "-" * 40)
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0]["text"] = "Nơi đăng ký HKTT:" + first
        targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][0])
        if len(second) != 0:
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = second
            copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["bbox"] = get_bbox(second, char_len,
                                                                                     np.array(copied_group[
                                                                                                  "REPRESENTATIVE_PERMANENT_RESIDENCE"][
                                                                                                  1]["bbox"]))
            targets.append(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1])
        # Representative living palace
        copied_group["REPRESENTATIVE_LIVING_PLACE"][0]["text"] = "Chỗ ở hiện tại: " + selected_data[
            "REPRESENTATIVE_LIVING_PLACE"]
        targets.append(copied_group["REPRESENTATIVE_LIVING_PLACE"][0])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"),
              'w', encoding='utf-8') as f: \
            f.write(json.dumps(new_data))


def process_20(original_data, addition_data, save_path):
    new_data = []
    data_id = 19
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["REPRESENTATIVE_PERMANENT_RESIDENCE"])
    shape = original_data[data_id]['shape']
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số:" + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][1]["text"] = selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        copied_group["COMPANY_ADDRESS"][0]["text"] = "2. Địa điểm kinh doanh: " + selected_data["COMPANY_ADDRESS"]
        targets.extend(copied_group["COMPANY_ADDRESS"])
        # Company phone
        copied_group["COMPANY_PHONE"][0]["text"] = "Điện thoại: " + selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]["text"] = "3. Ngành nghề kinh doanh: {} (phải thực hiện đúng theo".format(
            selected_data["BUSINESS_KIND"].capitalize())
        copied_group["BUSINESS_KIND"][0]["label"] = "OTHER"
        targets.extend(copied_group["BUSINESS_KIND"])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][0]["text"] = "4. Vốn kinh doanh: {} đồng ({})".format(number, character)
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][1]["text"] = selected_data["REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "-{}".format(selected_data["REPRESENTATIVE_SEX"])
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Sinh ngày:" + selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][0]["text"] = "Chứng minh nhân dân số: " + selected_data[
            "REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][0]["text"] = "Ngày cấp: " + selected_data[
            "REPRESENTATIVE_IDCARD_DATE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][0]["text"] = "Nơi cấp: " + selected_data[
            'REPRESENTATIVE_IDCARD_PLACE']
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent residence
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = selected_data[
            "REPRESENTATIVE_PERMANENT_RESIDENCE"]
        targets.extend(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"])
        # Representative living palace
        copied_group["REPRESENTATIVE_LIVING_PLACE"][0]["text"] = "Chỗ ở hiên tại: " + selected_data[
            "REPRESENTATIVE_LIVING_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_LIVING_PLACE"])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def process_21(original_data, addition_data, save_path):
    new_data = []
    data_id = 20
    print(original_data[data_id]["file_name"])
    new_data.append(original_data[data_id])
    groups = group_data(original_data[data_id]['target'])
    print(groups["REPRESENTATIVE_PERMANENT_RESIDENCE"])
    shape = original_data[data_id]['shape']
    for i in range(99):
        id: int = random.randint(0, len(addition_data) - 1)
        selected_data: dict = addition_data[id]
        save_data: dict = copy.deepcopy(original_data[data_id])
        copied_group = copy.deepcopy(groups)
        targets = []
        # Other
        targets.extend(copied_group["OTHER"])
        # Contract type
        targets.extend(copied_group["CONTRACT_TYPE"])
        # Contract number
        copied_group["CONTRACT_NUMBER"][0]['text'] = "Số:" + generate_contract_number()
        targets.extend(copied_group["CONTRACT_NUMBER"])
        # Company name
        copied_group["COMPANY_NAME"][0]["text"] = "1. Hộ kinh doanh: " + selected_data["COMPANY_NAME"]
        targets.extend(copied_group["COMPANY_NAME"])
        # Company address
        copied_group["COMPANY_ADDRESS"][0]["text"] = "2. Địa điểm kinh doanh: " + selected_data["COMPANY_ADDRESS"]
        targets.extend(copied_group["COMPANY_ADDRESS"])
        # Company phone
        copied_group["COMPANY_PHONE"][0]["text"] = "Điện thoại: " + selected_data["COMPANY_PHONE"]
        targets.extend(copied_group["COMPANY_PHONE"])
        # Business kind
        copied_group["BUSINESS_KIND"][0]["text"] = "3. Ngành nghề kinh doanh: {} (hàng hóa mua".format(
            selected_data["BUSINESS_KIND"].capitalize())
        copied_group["BUSINESS_KIND"][0]["label"] = "OTHER"
        targets.extend(copied_group["BUSINESS_KIND"])
        # Business capital
        number, character = random_captial()
        copied_group["BUSINESS_CAPITAL"][0]["text"] = "4. Vốn kinh doanh: {}".format(number)
        targets.extend(copied_group["BUSINESS_CAPITAL"])
        # Representative name
        copied_group["REPRESENTATIVE_NAME"][1]["text"] = selected_data["REPRESENTATIVE_NAME"]
        targets.extend(copied_group["REPRESENTATIVE_NAME"])
        # Representative sex
        copied_group["REPRESENTATIVE_SEX"][0]["text"] = "Nam/Nữ: ".format(selected_data["REPRESENTATIVE_SEX"])
        targets.extend(copied_group["REPRESENTATIVE_SEX"])
        # Representative birthday
        copied_group["REPRESENTATIVE_BIRTHDAY"][0]["text"] = "Sinh ngày:" + selected_data["REPRESENTATIVE_BIRTHDAY"]
        targets.extend(copied_group["REPRESENTATIVE_BIRTHDAY"])
        # Representative people
        copied_group["REPRESENTATIVE_MAJORITY"][0]["text"] = "Dân tộc:" + selected_data["REPRESENTATIVE_MAJORITY"]
        targets.extend(copied_group["REPRESENTATIVE_MAJORITY"])
        # Representative idcard number
        copied_group["REPRESENTATIVE_IDCARD_NUMBER"][0]["text"] = "Chứng minh nhân dân số: " + selected_data[
            "REPRESENTATIVE_IDCARD_NUMBER"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_NUMBER"])
        # Representative idcard date
        copied_group["REPRESENTATIVE_IDCARD_DATE"][0]["text"] = "Ngày cấp: " + selected_data[
            "REPRESENTATIVE_IDCARD_DATE"]
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_DATE"])
        # Representative idcard place
        copied_group["REPRESENTATIVE_IDCARD_PLACE"][0]["text"] = "Nơi cấp: " + selected_data[
            'REPRESENTATIVE_IDCARD_PLACE']
        targets.extend(copied_group["REPRESENTATIVE_IDCARD_PLACE"])
        # Representative permanent residence
        copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"][1]["text"] = selected_data[
            "REPRESENTATIVE_PERMANENT_RESIDENCE"]
        targets.extend(copied_group["REPRESENTATIVE_PERMANENT_RESIDENCE"])
        # Representative living palace
        copied_group["REPRESENTATIVE_LIVING_PLACE"][0]["text"] = "Chỗ ở hiên tại: " + selected_data[
            "REPRESENTATIVE_LIVING_PLACE"]
        targets.extend(copied_group["REPRESENTATIVE_LIVING_PLACE"])
        targets = sorted(targets, key=lambda x: x['bbox'][0][1])
        save_data['target'] = translate_coordinate(targets, shape)
        new_data.append(save_data)
    for item in new_data[1]['target']:
        print(item['text'], item['label'])
    with open(os.path.join(save_path, original_data[data_id]["file_name"] + ".json"),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))


def generate_data(original_path: str, addition_path: str, save_path: str):
    with open(original_path, 'r', encoding='utf-8') as f:
        original_data: List = json.loads(f.readline())
    with open(addition_path, 'r', encoding='utf-8') as f:
        addition_data: List = json.loads(f.readline())
    # Processing sample 1
    process_1(original_data, addition_data, save_path)
    # Processing sample 2
    process_2(original_data, addition_data, save_path)
    # Processing sample 3
    process_3(original_data, addition_data, save_path)
    # Processing sample 4
    process_4(original_data, addition_data, save_path)
    # Processing sample 5
    process_5(original_data, addition_data, save_path)
    # Processing sample 6
    process_6(original_data, addition_data, save_path)
    # Processing sample 7
    process_7(original_data, addition_data, save_path)
    # Processing sample 8
    process_8(original_data, addition_data, save_path)
    # Processing sample 9
    process_9(original_data, addition_data, save_path)
    # Processing sample 10
    process_10(original_data, addition_data, save_path)
    # Processing sample 11
    process_11(original_data, addition_data, save_path)
    # Processing sample 12
    process_12(original_data, addition_data, save_path)
    # Processing sample 13
    process_13(original_data, addition_data, save_path)
    # Processing sample 14
    process_14(original_data, addition_data, save_path)
    # Processing sample 15
    process_15(original_data, addition_data, save_path)
    # Processing sample 16
    process_16(original_data, addition_data, save_path)
    # Processing sample 17
    process_17(original_data, addition_data, save_path)
    # Processing sample 18
    process_18(original_data, addition_data, save_path)
    # Processing sample 19
    process_19(original_data, addition_data, save_path)
    # Processing sample 20
    process_20(original_data, addition_data, save_path)
    # Processing sample 21
    process_21(original_data, addition_data, save_path)


if __name__ == '__main__':
    # make_label([
    #     "CONTRACT_TYPE",
    #     "CONTRACT_NUMBER",
    #     "COMPANY_NAME",
    #     "COMPANY_ADDRESS",
    #     "COMPANY_PHONE",
    #     "COMPANY_FAX",
    #     "COMPANY_EMAIL/WEBSITE",
    #     "BUSINESS_CAPITAL",
    #     "REPRESENTATIVE_NAME",
    #     "REPRESENTATIVE_SEX",
    #     "REPRESENTATIVE_BIRTHDAY",
    #     "REPRESENTATIVE_MAJORITY",
    #     "REPRESENTATIVE_NATIONAL",
    #     "REPRESENTATIVE_IDCARD_NUMBER",
    #     "REPRESENTATIVE_IDCARD_DATE",
    #     "REPRESENTATIVE_IDCARD_PLACE",
    #     "REPRESENTATIVE_PERMANENT_RESIDENCE",
    #     "REPRESENTATIVE_LIVING_PLACE"
    # ], r'D:\python_project\dkkd_graph\asset\breg\label.json')
    # convert(r"D:\python_project\dkkd_graph\data\data.json",
    #         r"D:\python_project\dkkd_graph\data\total.json")
    # check(r"D:\python_project\dkkd_graph\data\total.json",
    #       r"D:\python_project\dkkd_graph\data\checked.json")
    # convert_excel(r"D:\python_project\dkkd_graph\data\hokd2.xls",
    #               "2300 CTY MOI THANH LAP HN 2010",
    #               r"D:\python_project\dkkd_graph\data\company_info.json",
    #               r"D:\python_project\dkkd_graph\data\hokd1.xls")
    # check_company(r"D:\python_project\dkkd_graph\data\company_info.json")
    generate_data(r"/data/data1/checked.json",
                  r"D:\python_project\breg_graph\data\info\new_data.json",
                  r"D:\python_project\breg_graph\data\gen")
