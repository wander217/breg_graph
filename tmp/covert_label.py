import copy
import json
import os


def change_label(label):
    changes = {
        # COMPANY_NAME
        "label_company_name": "COMPANY_NAME",
        "company_vietnamese_name": "COMPANY_NAME",
        "company_english_name": "COMPANY_NAME",
        "company_short_name": "COMPANY_NAME",
        # COMPANY_DETAIL
        "label_company_address": "COMPANY_DETAIL",
        "company_address": "COMPANY_DETAIL",
        "company_phone": "COMPANY_DETAIL",
        "company_fax": "COMPANY_DETAIL",
        "company_website/email": "COMPANY_DETAIL",
        # DOCUMENT
        "document": "DOCUMENT",
        # CONTRACT_DETAIL
        "contract_type": "CONTRACT_DETAIL",
        "company_code": "CONTRACT_DETAIL",
        "register_date": "CONTRACT_DETAIL",
        # BUSINESS_TYPE
        "business_type": "BUSINESS_TYPE",
        # OTHER
        "other": "OTHER",
        # SHAREHOLDER
        "shareholder": "SHAREHOLDER",
        "label_shareholder": "SHAREHOLDER",
        # REPRESENTATIVE/OWNER
        "label_representative": "REPRESENTATIVE/OWNER",
        "representative_name": "REPRESENTATIVE/OWNER",
        "representative_sex": "REPRESENTATIVE/OWNER",
        "representative_position": "REPRESENTATIVE/OWNER",
        "representative_birthday": "REPRESENTATIVE/OWNER",
        "representative_ethnicity": "REPRESENTATIVE/OWNER",
        "representative_nation": "REPRESENTATIVE/OWNER",
        "representative_idcard_type": "REPRESENTATIVE/OWNER",
        "representative_idcard_number": "REPRESENTATIVE/OWNER",
        "representative_idcard_date": "REPRESENTATIVE/OWNER",
        "representative_idcard_place": "REPRESENTATIVE/OWNER",
        "representative_residence_permanent": "REPRESENTATIVE/OWNER",
        "representative_living_place": "REPRESENTATIVE/OWNER",

        "label_owner": "REPRESENTATIVE/OWNER",
        "owner_name": "REPRESENTATIVE/OWNER",
        "owner_sex": "REPRESENTATIVE/OWNER",
        "owner_position": "REPRESENTATIVE/OWNER",
        "owner_birthday": "REPRESENTATIVE/OWNER",
        "owner_ethnicity": "REPRESENTATIVE/OWNER",
        "owner_nation": "REPRESENTATIVE/OWNER",
        "owner_idcard_type": "REPRESENTATIVE/OWNER",
        "owner_idcard_number": "REPRESENTATIVE/OWNER",
        "owner_idcard_date": "REPRESENTATIVE/OWNER",
        "owner_idcard_place": "REPRESENTATIVE/OWNER",
        "owner_residence_permanent": "REPRESENTATIVE/OWNER",
        "owner_living_place": "REPRESENTATIVE/OWNER",
        # OWNER_TYPE
        "owner_type": "OWNER_TYPE",
        # BRANCH_DETAIL
        "label_branch": "BRANCH_DETAIL",
        "branch_company_name": "BRANCH_DETAIL",
        "branch_company_code": "BRANCH_DETAIL",
        "branch_company_address": "BRANCH_DETAIL",
        # OFFICE_DETAIL
        "label_representative_office": "OFFICE_DETAIL",
        "representative_company_name": "OFFICE_DETAIL",
        "representative_company_code": "OFFICE_DETAIL",
        "representative_company_address": "OFFICE_DETAIL",
        # BUSINESS_DETAIL
        "label_business_place": "BUSINESS_DETAIL",
        "business_place_name": "BUSINESS_DETAIL",
        "business_place_code": "BUSINESS_DETAIL",
        "business_place_address": "BUSINESS_DETAIL",
        # BUSINESS_CAPITAL
        "label_business_capital": "BUSINESS_CAPITAL",
        "business_capital": "BUSINESS_CAPITAL",
        "business_par_value_share": "BUSINESS_CAPITAL",
        "business_total_share": "BUSINESS_CAPITAL",
        "number_of_saled_share": "BUSINESS_CAPITAL",
        # AUTHORITY
        "label_authority": "AUTHORITY",
        "authority_company_name": "AUTHORITY",
        "authority_company_code": "AUTHORITY",
        "authority_company_address": "AUTHORITY",
        # LEGAL_CAPITAL
        "legal_capital": "LEGAL_CAPITAL"
    }
    print(len(changes))
    return changes[label]


def convert_label(data):
    new_data = copy.deepcopy(data)
    for target in new_data['shapes']:
        target['label'] = change_label(target['label'])
    return new_data


save_path = r'D:\python_project\breg_graph\tmp\clustering_data'
data_path = r'D:\python_project\breg_graph\tmp\convert_data'
for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    os.mkdir(os.path.join(save_path, folder))
    for file in os.listdir(folder_path):
        if file.endswith("json"):
            with open(os.path.join(folder_path, file)) as f:
                data = json.loads(f.read())
            new_data = convert_label(data)
            with open(os.path.join(save_path, folder, file), 'w', encoding='utf-8') as f:
                f.write(json.dumps(new_data, indent=4))
