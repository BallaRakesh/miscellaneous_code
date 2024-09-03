import os
import json
import csv
from statistics import mean
from fuzzywuzzy import fuzz


def read_json(datajson_file):
    with open(datajson_file, 'r') as json_file:
        data = json.load(json_file)
    return data

def extract_keys_and_values(data):
    keys_values = {}
    # for page, elements in data.items():
    for key, value_list in data.items():
        merged_value = ' '.join([v[0] for v in value_list])
        if len(value_list) > 1:
            avg_confidence = mean([v[2] for v in value_list])
        else:
            avg_confidence = (value_list[0][2])
        keys_values[key] = (merged_value, avg_confidence)
    return keys_values

def extract_key_val(data):
    keys_values = {}
    # for page, elements in data.items():
    for key, value_list in data.items():
        merged_value = ' '.join([v[0] for v in value_list])
        keys_values[key] = merged_value
    return keys_values

def process_files(gt_folder, result1_folder, result2_folder, output_csv):
    gt_files = [f for f in os.listdir(gt_folder) if f.endswith('_labels.txt')]
    result1_files = [f for f in os.listdir(result1_folder) if f.endswith('.txt')]
    result2_files = [f for f in os.listdir(result2_folder) if f.endswith('.txt')]
    
    # Make sure the files are the same in all folders
    common_files = set(gt_files).intersection(result1_files,result2_files)
    
    results = []

    for file_name in common_files:
        gt_data = read_json(os.path.join(gt_folder, file_name))
        result1_data = read_json(os.path.join(result1_folder, file_name))
        result2_data = read_json(os.path.join(result2_folder, file_name))
        
        gt_keys_values = extract_key_val(gt_data)
        result1_keys_values = extract_keys_and_values(result1_data['keys_extraction'].get('Page Number 1', {}))
        result2_keys_values = extract_keys_and_values(result2_data['keys_extraction'].get('Page Number 1', {}))
        
        combined_keys = sorted(set(gt_keys_values.keys()).union(result1_keys_values.keys(), result2_keys_values.keys()))

        for key in combined_keys:
            gt_val = gt_keys_values.get(key, "")
            result1_val = result1_keys_values.get(key, ("", 0.0))
            result2_val = result2_keys_values.get(key, ("", 0.0))
            
            gt_value = gt_val
            result1_value, result1_conf = result1_val
            result2_value, result2_conf = result2_val
            print(gt_value)
            fuzzy_match_result1 = fuzz.ratio(gt_value.lower(), result1_value.lower())
            fuzzy_match_result2 = fuzz.ratio(gt_value.lower(), result2_value.lower())
            fuzzy_match_result3 = fuzz.ratio(result1_value.lower(), result2_value.lower())

            results.append({
                'image_name': file_name,
                'key': key,
                'gt_value': gt_value,
                'gt_conf': 0.0,
                'result1_value': result1_value,
                'result1_conf': result1_conf,
                'FUZZY MATCH GT VS LMV2': fuzzy_match_result1,
                'result2_value': result2_value,
                'result2_conf': result2_conf,
                'FUZZY MATCH GT VS TITRON': fuzzy_match_result2,
                'FUZZY MATCH LMV2 VS TIRON': fuzzy_match_result3
            })
    
    # Write results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'key', 'gt_value', 'gt_conf', 'result1_value', 'result1_conf', 'FUZZY MATCH GT VS LMV2', 'result2_value', 'result2_conf', 'FUZZY MATCH GT VS TITRON','FUZZY MATCH LMV2 VS TIRON']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)


gt_folder = '/home/ntlpt19/LLM_training/EVAL/PO/Master_Data'

# for file in os.listdir(gt_folder):
#     if file.endswith('_text.txt'):
#         file_path = os.path.join(gt_folder, file)
#         new_file_name = file[:-9] + '.txt'
#         new_file_path = os.path.join(gt_folder, new_file_name)
#         os.rename(file_path, new_file_path)

result1_folder = '/home/ntlpt19/LLM_training/LMV2_RESULTS/PO/RESULTS'


for file in os.listdir(result1_folder):
    file_path = os.path.join(result1_folder,file)
    new_file_name = file[:-4] + '_labels.txt'
    new_file_path = os.path.join(result1_folder,new_file_name)
    os.rename(file_path,new_file_path)

result2_folder = '/home/ntlpt19/LLM_training/TRITON_RESULTS/PO/RESULTS'


for file in os.listdir(result2_folder):
    file_path = os.path.join(result2_folder,file)
    new_file_name = file[:-4] + '_labels.txt'
    new_file_path = os.path.join(result2_folder,new_file_name)
    os.rename(file_path,new_file_path)

output_csv = '/home/ntlpt19/LLM_training/results/output_PO.csv'

process_files(gt_folder, result1_folder, result2_folder, output_csv)