import os
import pandas as pd

import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from datetime import datetime
import re

load_dotenv()

llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')

def merge(llm:object, query: str, structured_response: str, unstructured_response: str):
    prompt_template = prompt_template_merge_extraction(query, structured_response, unstructured_response)
    res = llm.invoke(prompt_template)
    return res.content


def prompt_template_merge_extraction(query: str, structured_response: str, unstructured_response: str):
    return f"""
    Given the context in the structured response, provide the 
    final response with respect to the unstructured query. If both responses do not have any 
    information within it, inform that you are unable to provide an answer as the 
    information is not available.

    Here's the query: {query}

    Sturctured responses: {structured_response}
    Unstructures responses: {unstructured_response}
    Response:
    """


def final_report(csv01, folder_path, doc_code, category_name):
    match_name = "Match/No_Match"
    df = pd.read_csv(os.path.join(csv01))

    report = {}
    label_names = []
    label_counts = []
    label_detected = []
    detection_accuracy = []
    average_accuracy = []
    matched_labels = []
    matched_detected = []
    total_match_percentage = []
    g = df.groupby("label_name")
    for name, name_df in g:
        label_names.append(name)
        label_counts.append(len(name_df.index))
        average_accuracy.append(round(name_df["Accuracy"].mean(), 2))
        matched = sum(list(name_df[match_name]))
        print(f'matched numbers: {matched}')
        matched_labels.append(matched)
        total_match_percentage.append(round((matched / len(name_df.index)) * 100, 2))
        not_detected = name_df["filter_prediction"].isnull().sum()
        detected = len(name_df.index) - not_detected
        print(f'detected numbers: {detected}')
        label_detected.append(detected)
        matched_detected.append(matched / detected)
        print(matched_detected)
        detection_accuracy.append((detected / len(name_df.index)) * 100)
    data = {'Label_Name': label_names, 'Label_Count': label_counts, "Labels_Detected": label_detected,
            "Detection_Accuracy": detection_accuracy, "Fuzzy_Match_Percentage": average_accuracy,
            "Complete_Match_Count_Detected": matched_detected, "Complete_Match_Count": matched_labels,
            "Complete_Match_Percentage": total_match_percentage}

    # Just to calculate detection accuracy
    detected = sum(label_detected)
    all_matched = sum(matched_labels)
    avg_matched_detected = all_matched / detected
    all_labels = sum(label_counts)
    overall_detection_accuracy = (detected / all_labels) * 100
    # dataframe and csv file generation
    report = pd.DataFrame(data)
    li = ["OVERALL", sum(list(report["Label_Count"])), sum(list(report["Labels_Detected"])), overall_detection_accuracy,
        df["Accuracy"].mean(), avg_matched_detected, sum(list(report["Complete_Match_Count"])),
        sum(list(report["Complete_Match_Count"])) / sum(list(report["Label_Count"]))]
    report.loc[len(report.index)] = li

    if not os.path.exists(os.path.join(folder_path, 'result_path', category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}")):
        os.makedirs(os.path.join(folder_path, 'result_path', category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}"))


    name = f"final_report_{doc_code}_pre_{str(datetime.now())}_after_fuzzy_match_change.csv"

    file_path_csv2 = os.path.join(folder_path, 'result_path', category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}", name)
    report.to_csv(file_path_csv2)

    # txt file generation
    name = f"final_report_{datetime.now()}" + ".txt"
    name_path = os.path.join(folder_path, 'result_path',category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}",  name)
    text_report = {
        row["Label_Name"]: {
            "Label_Count": row["Label_Count"],
            "Detection_Accuracy": row["Detection_Accuracy"],
            "Fuzzy_Match_Percentage": row["Fuzzy_Match_Percentage"],
            "Complete_Match_Count_Detected": row["Complete_Match_Count_Detected"],
            "Complete_Match_Count": row["Complete_Match_Count"],
            "Complete_Match_Percentage": row["Complete_Match_Percentage"],
        }
        for index, row in report.iterrows()
    }
    with open(name_path, "w") as f:
        json.dump(text_report, f)
    f.close()

    return file_path_csv2

def extract_key_value_pairs(input_text):
    pairs = {}
    lines = input_text.strip().split('\n')
    current_key = None
    current_value = ""

    for line in lines:
        line = line.strip().strip('"')
        match = re.match(r'(.+?):\s*(.+?)$', line)

        if match:
            if current_key:
                pairs[current_key] = current_value.strip()
            current_key, current_value = match.groups()
            current_key = current_key.strip('"')
            current_value = current_value.strip('"').rstrip(',')
        elif current_key:
            current_value += " " + line.strip('"').rstrip(',')

    if current_key:
        pairs[current_key] = current_value.strip()

    return pairs

def extract_text_from_word_coords(word_coordinates: list):
    all_text = ' '.join([item['word'] for item in word_coordinates])
    return all_text


def get_low_match_labels(csv_file, threshold_val = 75, skip_keys = []):
    # Step 1: Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Step 2: Initialize an empty list to store the labels
    labels_below_threshold = []

    # Step 3: Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Check if the "Complete_Match_Percentage" is less than 70
        if row['Label_Name'] not in skip_keys:
            if row['Complete_Match_Percentage'] < threshold_val:
                # Append the corresponding "Label_Name" to the list
                labels_below_threshold.append(row['Label_Name'])
    
    # Step 4: Return the list of labels
    return labels_below_threshold


def open_text_file(file_path):
    # Open the file in read mode ('r')
    with open(file_path, 'r') as file:
        # Read the entire content of the file
        content = file.read()
    return content

def remove_special_chars_check(input_text):
    characters_to_remove = ".'·!'|:()/-%;'*,"
    cleaned_text = re.sub(f'^[{re.escape(characters_to_remove)}\s]+|[{re.escape(characters_to_remove)}\s]+$', '', str(input_text))
    return cleaned_text


def verify_prediction(actual_val, merged_result = None, mistral_key_value = None, mistral_flag = False, gpt_flag = False, both_results = False):
    if mistral_flag:
        if actual_val.lower() in str(merged_result).lower():
            return True
        else:
            return False
    elif gpt_flag:
        if actual_val.lower() in str(mistral_key_value).lower():
            return True
        else:
            return False
    else:
        if actual_val.lower() in str(merged_result).lower() or actual_val.lower() in str(mistral_key_value).lower():
            return True
        else:
            return False

if __name__ == '__main__':
    
    # final_report('/home/ntlpt19/itf_results_field_wise_report/CI/filtered_file_sep3.csv', '/home/ntlpt19/itf_results_field_wise_report/CI', 'CI', 'label_wise')
    # exit('OK')
    mistral_results = '/home/ntlpt19/LLM_training/EVAL/CI/results/text_files'
    csv_file_path = '/home/ntlpt19/itf_results_field_wise_report/CI/result_path/label_wise/CI_results/final_reprt_ci_label.csv'
    label_wise_keys = '/home/ntlpt19/itf_results_field_wise_report/CI/result_path/label_wise/ci_analysis_pre_valid_after_fuzzy_match_post_processing_latest.csv'
    ocr_folder = '/home/ntlpt19/itf_results_field_wise_report/CI/OCR'
    previous_gpt_results = '/home/ntlpt19/itf_results_field_wise_report/CI/filtered_file_gpt.csv'
    gpt_df = pd.read_csv(previous_gpt_results)
    
    only_mistral_flag = True
    only_gpt_flag = False
    both_results = False
    executed_json_file = ''
    if os.path.exists(executed_json_file):
        with open(executed_json_file, 'r') as json_file:
            executed_json_file_data = json.load(json_file)
    else:
        executed_json_file_data = {}
    root_folder = os.path.dirname(ocr_folder)
    master_exception_keys = ['signature', 'OVERALL', 'stamp']
    exception_keys = {
        'CI': ['signature']
    }
    threshold_ = 70
    document_type = 'CI'
    executed_file = {}
    df = pd.read_csv(label_wise_keys)
    
    if both_results or only_mistral_flag:
        df['mistral_result'] = None
    if both_results or only_gpt_flag:
        df['gpt_result'] = None

    df['filter_prediction'] = df['predicted']
    low_match_labels = get_low_match_labels(csv_file_path, threshold_val = threshold_, skip_keys = exception_keys.get(document_type, [])+ master_exception_keys)
    print(low_match_labels)
    for req_keys in low_match_labels:
        print(req_keys)
        filtered_df = df[df['label_name'] == req_keys]
        for img_name in filtered_df['File_Name']:
            filename_without_extension = os.path.splitext(img_name)[0]
            print(filename_without_extension) 
            match_no_Match = df[(df['File_Name'] == img_name) & (df['label_name'] == req_keys)]['Match/No_Match'].iloc[0]
            if match_no_Match == 0 and img_name+'_'+req_keys not in executed_json_file_data:
                ocr_file = os.path.join(ocr_folder, filename_without_extension + '_text.txt')
                print(ocr_file)
                if os.path.exists(ocr_file):
                    all_text = extract_text_from_word_coords(eval(open_text_file(ocr_file)).get('word_coordinates', []))
                    print(all_text)
                else:
                    print(f'File {ocr_file} does not exist.')
                preprocessed_key = req_keys.replace('_', ' ').replace('-', ' ')
                query = f'Find the ocr information for the {img_name}and extract the {preprocessed_key} ?'
                structured_response = all_text
                unstructured_response = f'extract the {preprocessed_key}'
                if os.path.exists(os.path.join(mistral_results, filename_without_extension+'.txt')):
                    mistral_sample_result = open_text_file(os.path.join(mistral_results, filename_without_extension+'.txt'))
                    print(mistral_sample_result)
                    mistral_sample_result_json = extract_key_value_pairs(mistral_sample_result)
                    mistral_key_value = mistral_sample_result_json.get(req_keys, '') + mistral_sample_result_json.get(preprocessed_key, '')
                    print(mistral_key_value)
                else:
                    mistral_key_value = ''
                # Find the ocr information for the document name pdf10_4.png and extract the your order no?
                merged_result = merge(llm, query, structured_response, unstructured_response)
                exit('OK')
                merged_result = gpt_df[(gpt_df['File_Name'] == img_name) & (gpt_df['label_name'] == req_keys)]['predicted'].iloc[0]
                
                executed_file[img_name+'_'+req_keys] = str(merged_result)
                
                actual_val = df[(df['File_Name'] == img_name) & (df['label_name'] == req_keys)]['actual'].iloc[0]
                actual_val = str(remove_special_chars_check(actual_val))
                if not pd.isna(actual_val):
                    if both_results or only_gpt_flag:
                        df.loc[(df['File_Name'] == img_name) & (df['label_name'] == req_keys), 'gpt_result'] = merged_result
                    if both_results or only_mistral_flag:
                        df.loc[(df['File_Name'] == img_name) & (df['label_name'] == req_keys), 'mistral_result'] = mistral_key_value
                        
                    if only_mistral_flag:
                        if mistral_key_value:
                            df.loc[(df['File_Name'] == img_name) & (df['label_name'] == req_keys), 'filter_prediction'] = mistral_key_value
                    else:
                        df.loc[(df['File_Name'] == img_name) & (df['label_name'] == req_keys), 'filter_prediction'] = merged_result
                    # if verify_prediction(actual_val, merged_result = merged_result, mistral_key_value = mistral_key_value, mistral_flag = only_mistral_flag, gpt_flag = only_gpt_flag, both_results = both_results):
                    # if actual_val in str(merged_result) or actual_val in str(mistral_key_value):
                    if actual_val.lower() in str(mistral_key_value).lower():
                        df.loc[(df['File_Name'] == img_name) & (df['label_name'] == req_keys), 'Accuracy'] = 100
                        df.loc[(df['File_Name'] == img_name) & (df['label_name'] == req_keys), 'Match/No_Match'] = 1  
    df.to_csv(os.path.join(root_folder, 'filtered_file_sep3.csv'), index=False)
    
    with open('executed_file1.json', 'w') as json_file:
        json.dump(executed_file, json_file, indent=4)  




