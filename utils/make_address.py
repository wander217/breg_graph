import copy
import json
import os.path
import random
import pandas as pd
import numpy as np


def remove_space(txt: str):
    return txt.strip().strip("\r\t").strip("\n")


def cut_company(name: str):
    tmp = name
    tmp = tmp.split('"')
    if len(tmp) != 0:
        tmp = tmp[0]
    tmp = tmp.split("(")
    if len(tmp) != 0:
        tmp = tmp[0]
    return remove_space(tmp)


def convert_date(timestamp):
    date = timestamp.strftime("%d/%m/%Y")
    return date


data: pd.DataFrame = pd.read_excel(r'D:\workspace\project\dkkd_graph\data\info.xls',
                                   sheet_name='2300 CTY MOI THANH LAP HN 2010')
data = data.replace(np.nan, "\\")
data = data.replace("\\", "")
with open(r"D:\workspace\project\dkkd_graph\data\info\business.json", 'r', encoding='utf-8') as f:
    business_kind = json.loads(f.readline())

new_data = []
for id, row in data.iterrows():
    try:
        new_data.append({
            "COMPANY_NAME": cut_company(row["Tên doanh nghiệp"]),
            "COMPANY_ADDRESS": remove_space(row["Địa chỉ"]),
            "COMPANY_PHONE": remove_space(row["Điện thọai"]),
            "COMPANY_FAX": remove_space(row["Fax"]),
            "COMPANY_EMAIL/WEBSITE": remove_space(row["Email"]),
            "BUSINESS_KIND": remove_space(random.choice(business_kind)),
            "REPRESENTATIVE_NAME": remove_space(row["Người đại diện"]),
            "REPRESENTATIVE_SEX": remove_space(row["Giới tính"]),
            "REPRESENTATIVE_MAJORITY": remove_space(row["Dân tộc"]),
            "REPRESENTATIVE_NATIONAL": remove_space(row["Quốc tịch"]),
            "REPRESENTATIVE_IDCARD_NUMER": remove_space(row["Số CMTND/Hộ chiếu"]),
            "REPRESENTATIVE_BIRTHDAY": remove_space(convert_date(row["Ngày sinh"])),
            "REPRESENTATIVE_IDCARD_NUMBER": remove_space(row["Số CMTND/Hộ chiếu"]),
            "REPRESENTATIVE_IDCARD_DATE": remove_space(convert_date(row["Ngày cấp CMTND/HC"])),
            "REPRESENTATIVE_IDCARD_PLACE": remove_space(row["Nơi cấp CMTND/HC"]),
            "REPRESENTATIVE_PERMANENT_RESIDENCE": remove_space(row["Hộ khẩu người đại diện"]),
            "REPRESENTATIVE_LIVING_PLACE": remove_space(row["Chổ ở hiện tại người đại diện"])
        })
    except Exception as e:
        print(e)
for i in range(2080 - len(new_data)):
    item = random.choice(new_data)
    tmp = copy.deepcopy(item)
    tmp["COMPANY_NAME"] = "HỘ KINH DOANH " + tmp["REPRESENTATIVE_NAME"]
    tmp["REPRESENTATIVE_LIVING_PLACE"] = tmp["COMPANY_ADDRESS"]
    new_data.append(tmp)
data_path = r'D:\workspace\project\dkkd_graph\data\info'
with open(os.path.join(data_path, "new_data.json"), 'w', encoding='utf-8') as f:
    f.write(json.dumps(new_data))
print(len(new_data))
