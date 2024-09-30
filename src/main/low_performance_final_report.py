
import pandas as pd
from low_performance_key_v2 import final_report
import os


def read_file_into_list(file_name):  
    try:  
        # Open the file in read mode  
        with open(file_name, 'r') as file:  
            # Read all lines into a list  
            lines = [line.strip() for line in file]  
            return lines  
    except FileNotFoundError:  
        print(f"The file {file_name} does not exist.")  
        return []  
    except Exception as e:  
        print(f"An error occurred: {e}")  
        return [] 

def get_report():
    df = pd.read_csv(updated_sheet)
        
    data_frames = [df]
    sheet_types = ['gpt_updated_sheet']
    low_match_labels = read_file_into_list(keys_file)
    print(low_match_labels)

    for data_frame, sheet_type in zip(data_frames, sheet_types):
        out_put_df = final_report('', os.path.join(root_path, 'Reports'), document_class_com, 'label_wise', sheet_type, pandas_data=data_frame)
        filtered_df = out_put_df[out_put_df['Label_Name'].isin(low_match_labels)]
        filtered_df = filtered_df[['Label_Name', 'Label_Count', 'Labels_Detected', 'Complete_Match_Count', 'Complete_Match_Percentage']]
        # filtered_dfs.append(filtered_df)
        # Save combined DataFrame to CSV
    combined_csv_path = os.path.join(root_path, 'qwen_final_report_updated_sheet.csv')
    filtered_df.to_csv(combined_csv_path, index=False)

if __name__ == '__main__':
    root_path =  '/home/ntlpt19/itf_results_field_wise_report/PI/results_qwen2_7B_v1/V2'
    updated_sheet = os.path.join(root_path,'filtered_file_gpt_manual.csv')
    # updated_sheet = '/home/ntlpt19/itf_results_field_wise_report/BOE/results_qwen2_7b_v2/V2/filtered_file_gpt_manual.csv'
    document_class_com = 'Perfoma Invoice'
    keys_file = os.path.join(root_path, 'Perfoma_Invoice_low_match_labels.txt')
    get_report()