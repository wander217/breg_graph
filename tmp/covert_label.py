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
        "label_company_address": "COMPANY_ADDRESS",
        "company_address": "COMPANY_ADDRESS",
        "company_phone": "COMPANY_PHONE/FAX",
        "company_fax": "COMPANY_PHONE/FAX",
        "company_website/email": "COMPANY_WEBSITE/EMAIL",
        # DOCUMENT
        "document": "DOCUMENT",
        # CONTRACT_DETAIL
        "contract_type": "CONTRACT_TYPE",
        "company_code": "COMPANY_CODE",
        "register_date": "REGISTER_DATE",
        # BUSINESS_TYPE
        "business_type": "BUSINESS_TYPE",
        # OTHER
        "other": "OTHER",
        # SHAREHOLDER
        "shareholder": "OTHER",
        "label_shareholder": "OTHER",
        # REPRESENTATIVE/OWNER
        "label_representative": "OWNER_REPRESENTATIVE_NAME",
        "representative_name": "OWNER_REPRESENTATIVE_NAME",
        "representative_sex": "OWNER_REPRESENTATIVE_SEX",
        "representative_position": "OWNER_REPRESENTATIVE_POSITION",
        "representative_birthday": "OWNER_REPRESENTATIVE_BIRTHDAY",
        "representative_ethnicity": "OWNER_REPRESENTATIVE_ETHNICITY",
        "representative_nation": "OWNER_REPRESENTATIVE_NATION",
        "representative_idcard_type": "OWNER_REPRESENTATIVE_IDCARD_TYPE",
        "representative_idcard_number": "OWNER_REPRESENTATIVE_IDCARD_NUMBER/CODE",
        "representative_idcard_date": "OWNER_REPRESENTATIVE_IDCARD_DATE",
        "representative_idcard_place": "OWNER_REPRESENTATIVE_IDCARD_PLACE",
        "representative_residence_permanent": "OWNER_REPRESENTATIVE_RESIDENCE_PERMANENT",
        "representative_living_place": "OWNER_REPRESENTATIVE_LIVING_PLACE",

        "label_owner": "OWNER_REPRESENTATIVE_NAME",
        "owner_name": "OWNER_REPRESENTATIVE_NAME",
        "owner_sex": "OWNER_REPRESENTATIVE_SEX",
        "owner_position": "OWNER_REPRESENTATIVE_POSITION",
        "owner_birthday": "OWNER_REPRESENTATIVE_BIRTHDAY",
        "owner_ethnicity": "OWNER_REPRESENTATIVE_ETHNICITY",
        "owner_nation": "OWNER_REPRESENTATIVE_NATION",
        "owner_idcard_type": "OWNER_REPRESENTATIVE_IDCARD_TYPE",
        "owner_idcard_number": "OWNER_REPRESENTATIVE_IDCARD_NUMBER/CODE",
        "owner_idcard_date": "OWNER_REPRESENTATIVE_IDCARD_DATE",
        "owner_idcard_place": "OWNER_REPRESENTATIVE_IDCARD_PLACE",
        "owner_residence_permanent": "OWNER_REPRESENTATIVE_RESIDENCE_PERMANENT",
        "owner_living_place": "OWNER_REPRESENTATIVE_LIVING_PLACE",
        # OWNER_TYPE
        "owner_type": "OWNER_TYPE",
        # AUTHORITY
        "label_branch": "BRANCH_COMPANY_NAME",
        "branch_company_name": "BRANCH_COMPANY_NAME",
        "branch_company_code": "BRANCH_COMPANY_CODE",
        "branch_company_address": "BRANCH_COMPANY_ADDRESS",

        "label_representative_office": "BRANCH_COMPANY_NAME",
        "representative_company_name": "BRANCH_COMPANY_NAME",
        "representative_company_code": "BRANCH_COMPANY_CODE",
        "representative_company_address": "BRANCH_COMPANY_ADDRESS",

        "label_business_place": "BRANCH_COMPANY_NAME",
        "business_place_name": "BRANCH_COMPANY_NAME",
        "business_place_code": "BRANCH_COMPANY_CODE",
        "business_place_address": "BRANCH_COMPANY_ADDRESS",

        "label_authority": "BRANCH_COMPANY_NAME",
        "authority_company_name": "BRANCH_COMPANY_NAME",
        "authority_company_code": "BRANCH_COMPANY_CODE",
        "authority_company_address": "BRANCH_COMPANY_ADDRESS",
        # BUSINESS_CAPITAL
        "label_business_capital": "BUSINESS_CAPITAL",
        "business_capital": "BUSINESS_CAPITAL",
        "business_par_value_share": "BUSINESS_CAPITAL",
        "business_total_share": "BUSINESS_CAPITAL",
        "number_of_saled_share": "BUSINESS_CAPITAL",
        # LEGAL_CAPITAL
        "legal_capital": "BUSINESS_CAPITAL"
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
