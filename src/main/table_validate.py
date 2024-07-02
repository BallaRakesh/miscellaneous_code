import os
import pandas as pd
import json

def process_csv(csv_path, text_folder_path, output_csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize a list to store the status of 'cells'
    cells_status = []
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        image_path = row['Image Path']
        image_name = os.path.basename(image_path)
        
        img_base = os.path.splitext(image_name)[0] 
        if os.path.exists(os.path.join(text_folder_path, img_base + '.jpg'+".txt")):
            json_file_path = os.path.join(text_folder_path, img_base + '.jpg'+".txt")
            
        elif os.path.exists(os.path.join(text_folder_path, img_base + '.png'+".txt")):
            json_file_path = os.path.join(text_folder_path, img_base + '.png'+".txt")

        else:
            json_file_path = ''
        # json_file_path = os.path.join(text_folder_path, image_name+".txt")
        
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
                if 'table_res' in data and 'cells' in data['table_res']:
                    if len(data['table_res']['cells']) > 0:
                        cells_status.append('cells_exists')
                    else:
                        cells_status.append('not_exists')
                else:
                    cells_status.append('not_exists')
        else:
            cells_status.append('not_exists')
    
    # Add the new column to the DataFrame
    df['cells_status'] = cells_status
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to {output_csv_path}")

# Example usage
csv_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/image_predictions.csv'
text_folder_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/out_put/ROOT/results'
output_csv_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/updated_pred.csv'

process_csv(csv_path, text_folder_path, output_csv_path)
