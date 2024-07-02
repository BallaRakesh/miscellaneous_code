import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os
import re
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
from get_ocr_gv_pytesseract_function import get_ocr_vision_api_charConfi


from typing import List, Tuple

from typing import List, Dict, Tuple



from typing import List, Tuple

def validate(filtered_bboxes: List[Tuple[Tuple[int, int, int, int], str]], min_word_length: int = 5, min_word_count: int = 3, vertical_tolerance: int = 10) -> List[Tuple[Tuple[int, int, int, int], str]]:
    """
    Validate the bounding boxes based on the given conditions.

    :param filtered_bboxes: List of tuples containing bounding box coordinates and description.
    :param min_word_length: Minimum length of words to be considered for the validation.
    :param min_word_count: Minimum number of words with min_word_length to be present in the description.
    :param vertical_tolerance: Minimum vertical distance between bounding boxes.
    :return: List of validated bounding boxes.
    """
    validated_bboxes = []

    # Function to check the number of words meeting the minimum length requirement
    def has_min_words(description: str, min_length: int, min_count: int) -> bool:
        words = description.split()
        long_words = [word for word in words if len(word) >= min_length]
        return len(long_words) >= min_count

    # Sort bounding boxes based on their top (y1) coordinate to check vertical distance
    sorted_bboxes = sorted(filtered_bboxes, key=lambda x: x[0][1])

    # Validate each bounding box
    for i, (bbox, description) in enumerate(sorted_bboxes):
        if has_min_words(description, min_word_length, min_word_count):
            # Check vertical distance with all previous validated bounding boxes
            valid = True
            for prev_bbox, _ in validated_bboxes:
                if abs(bbox[1] - prev_bbox[3]) < vertical_tolerance:
                    valid = False
                    break
            if valid:
                validated_bboxes.append((bbox, description))

    return validated_bboxes

def get_intersection_percentage(bb1, bb2):
    """
    Finds the percentage of intersection  with a smaller box. (what percernt of smaller box is in larger box)
    """
    # assert bb1['x1'] < bb1['x2']
    # assert bb1['y1'] < bb1['y2']
    # assert bb2['x1'] < bb2['x2']
    # assert bb2['y1'] < bb2['y2']


    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    # min_area = min(bb1_area,bb2_area)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
 
    if bb1_area > bb2_area:
        intersection_percent = intersection_area / bb2_area
    else:
        intersection_percent = intersection_area / bb1_area
        if intersection_percent<0.5:
            intersection_percent=1  # if ocr bounding box is big then we need to consider the entire token if the intersection is less than o.5 also 
            
    assert intersection_percent >= 0.0
    assert intersection_percent <= 1.0
    
    # print(f"Intersection Percentage: {intersection_percent}")
    return intersection_percent


# Filter bounding boxes
def finl_count_column_wise(count_column, words):
    filtered_bboxes = []
    for bbox, value in count_column.items():
        content = ''
        for word in words:
            if get_intersection_percentage(bbox, [word['x1'], word['y1'], word['x2'], word['y2']]) >= 0.40:
                content += word['word'] + " "
        filtered_bboxes.append((bbox, content))
    final_filter_values = validate(filtered_bboxes)
    return final_filter_values
    


def merge_nearby_bboxes(bboxes: List[List[int]], tolerance: int = 20) -> List[List[int]]:
    """
    Merge nearby bounding boxes into sentence-wise bounding boxes.
    
    :param bboxes: List of bounding boxes for each line.
    :param tolerance: Maximum distance between bboxes to be considered part of the same sentence.
    :return: List of merged sentence-wise bounding boxes.
    """
    merged_bboxes = []
    for line_bboxes in bboxes:
        if not line_bboxes:
            continue
        current_bbox = line_bboxes[0]
        for bbox in line_bboxes[1:]:
            if bbox[0] - current_bbox[2] <= tolerance:
                # Merge bboxes
                current_bbox[2] = max(current_bbox[2], bbox[2])
                current_bbox[3] = max(current_bbox[3], bbox[3])
            else:
                merged_bboxes.append(current_bbox)
                current_bbox = bbox
        merged_bboxes.append(current_bbox)
    return merged_bboxes


def count_column_intersections(sentence_bboxes: List[List[int]], col_lines: List[int], line_height_tolerance: int = 3) -> Dict[Tuple[int, int, int, int], int]:
    """
    Count how many column lines intersect each sentence bounding box.
    
    :param sentence_bboxes: List of sentence-wise bounding boxes.
    :param col_lines: List of x-coordinates representing column lines.
    :param line_height_tolerance: Tolerance for considering a line passing through a bbox.
    :return: Dictionary with sentence bounding boxes as keys and counts of column lines intersecting them as values.
    """

    intersections = {}
    for bbox in sentence_bboxes:
        count = 0
        for col_line in col_lines:
            print(col_line)
            print(bbox)
            if bbox[0] + line_height_tolerance <= col_line[0] <= bbox[2] - line_height_tolerance:
                count += 1
        intersections[tuple(bbox)] = count
    return intersections


def are_on_same_line(bbox1, bbox2, min_distance=0, tolerance=10):
    # Check if the vertical distance between the bottom of bbox1 and the top of bbox2 is within the tolerance
    # and if the overall distance is at least min_distance
    return (
        abs(bbox2[1] - bbox1[1]) <= tolerance
        and abs(bbox2[0] - bbox1[2]) >= min_distance
    )

def remove_duplicates_and_nearby(bboxes: List[List[int]], tolerance_val: int = 20) -> List[List[int]]:
    """
    Remove duplicate bounding boxes and filter out those that are too close to each other.
    
    :param bboxes: List of bounding boxes.
    :param tolerance: Maximum distance to consider bounding boxes as nearby.
    :return: Filtered list of bounding boxes.
    """
    # Remove duplicate bounding boxes
    unique_bboxes = list(map(list, set(map(tuple, bboxes))))
    
    # Sort bounding boxes based on their x1 coordinate
    sorted_bboxes = sorted(unique_bboxes, key=lambda x: x[0])
    
    filtered_bboxes = []
    
    for bbox in sorted_bboxes:
        if not filtered_bboxes:
            filtered_bboxes.append(bbox)
        else:
            last_bbox = filtered_bboxes[-1]
            # Check if the current bbox is too close to the last added bbox
            if abs(bbox[0] - last_bbox[0]) > tolerance_val:
                filtered_bboxes.append(bbox)
    
    return filtered_bboxes

def group_tokens_by_line(bbox_data, line_tolerance=15):
    all_bboxes_ = []
    line_wise_index = {}
    initial_bbox = []
    # Create sublists of OCR data in the same line with horizontal tolerance
    idx = 0
    for master_bbox in bbox_data:
        check_flag = False
        bbox = master_bbox
        if bbox not in all_bboxes_:
            all_bboxes_.append(bbox)
            line_wise_index[idx] = [bbox]
            check_flag = True

        if check_flag:
            prev_bbox = line_wise_index[idx][0]
            print("prev_bbox >>>>>>>>>>>", prev_bbox)
            for single_bbox in bbox_data:
                print('single_bbox>>>>>>>>>>>>>', single_bbox)
                if are_on_same_line(prev_bbox, single_bbox, tolerance=line_tolerance):
                    if single_bbox not in all_bboxes_:
                        line_wise_index[idx].append(single_bbox)
                        all_bboxes_.append(single_bbox)
            line_wise_index[idx].sort(key=lambda x: x[0])
            initial_bbox.append(line_wise_index[idx][0])
            idx += 1
    print(initial_bbox)
    initial_bbox.sort(key=lambda x: x[1])
    print(initial_bbox)
    updated_line_wise_data = []
    for bx in initial_bbox:
        for idx, values in line_wise_index.items():
            if bx in values:
                updated_line_wise_data.append(values)
    
    
    return updated_line_wise_data


def words_on_line_pre(words: List[Dict], row_line: List[int], col_line: List[int], line_height_tolerance: int = 5) -> List[Dict]:
    tokens_on_row_line = []
    tokens_on_col_line = []
    for word in words:
        # Check if the word's bounding box intersects with the row line (considering some tolerance for height)
        for row_lines_ in row_line:
            if word['y1'] - line_height_tolerance <= row_lines_[1] <= word['y2'] + line_height_tolerance:
                tokens_on_row_line.append(word['word'])

        for col_lines_ in col_line:
            if word['x1'] - line_height_tolerance <= col_lines_[0] <= word['x2'] + line_height_tolerance:
                tokens_on_col_line.append(word['word'])
        
    return tokens_on_row_line, tokens_on_col_line


def get_sentence_wise_col_line_overlapping(ocr_word_codinates, column_bboxes):
    modified_ocr_bbox = []
    print(ocr_word_codinates)
    for wc_ in ocr_word_codinates:
        modified_ocr_bbox.append([wc_['x1'], wc_['y1'], wc_['x2'], wc_['y2']])
    print(modified_ocr_bbox)
    line_wise_data = group_tokens_by_line(modified_ocr_bbox, line_tolerance=15)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(line_wise_data)
    merged_bboxes = merge_nearby_bboxes(line_wise_data, tolerance = 25)
    print('???????????????????????????????????????')
    print(merged_bboxes)
    filter_col_bboxes = remove_duplicates_and_nearby(column_bboxes, tolerance_val=5)
    col_dict = count_column_intersections(merged_bboxes, filter_col_bboxes)
    print(col_dict)
    filtered_col_dict = {bbox: value for bbox, value in col_dict.items() if value != 0}
    print(filtered_col_dict)
    count_column_wise = finl_count_column_wise(filtered_col_dict, ocr_word_codinates)
    print('>>>>>>>>')
    print(count_column_wise)
    if not filtered_col_dict:
        filtered_col_dict_value = 0
        # or raise an exception, or return a default value
    else:
        filtered_col_dict_value = max(filtered_col_dict.values())
    return len(count_column_wise), filtered_col_dict_value


class IdentifyMiscellaneous:
    
    def __init__(self, row_info, column_info, croped_img_ocr_path, table_res):
        self.croped_img_ocr_path = croped_img_ocr_path
        self.row_info = row_info
        self.column_info = column_info
        self.table_res = table_res
        
        self.max_token_thresold = 40
        self.max_tokens_mismatch_thresold = 6
        self.row_wise_grand_average_hresold = 0.8 #{'row_averages': [0.125, 0.875, 0.125], 'overall_average': 0.375}
        self.column_wise_max_ratio_thresold = 0.8
        self.ocr_data, _ = get_ocr_vision_api_charConfi(croped_img_ocr_path)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(self.ocr_data)
        
        
        # with open(self.croped_img_ocr_path , 'r') as file:
        #     self.ocr_data = json.load(file)
        # print(self.ocr_data)

    def tokenize(self, content):
        return content.split()

        # Initialize variables to keep track of the cell with the highest token count
    def extract_column_bboxes(self):
        # Extract single_line_elements and column_seperators
        single_line_elements = self.column_info['single_line_elements']
        column_seperators = self.column_info['column_seperators']
        # Initialize the list to store bboxes
        bboxes = []
        # Loop through column_seperators to get bounding boxes
        for item in column_seperators:
            # Each item could be a list or tuple, so normalize it
            if isinstance(item, list) or isinstance(item, tuple):
                bbox = item[0]  # The bbox is always the first element
                bboxes.append(bbox)
        return bboxes
        
        
    def extract_row_bboxes(self):
        row_bboxs = [row_bbox.get('coords', []) for row_bbox in self.row_info]
        return row_bboxs
        
    def get_column_wise_data(self):
        # Initialize a dictionary to store column-wise data
        columns = {}

        # Iterate through each item in the list
        for item in self.table_res['cells']:
            # Extract column number and content
            column_num = item['column_nums'][0]
            content = item['content']

            # If the column number is not already a key in the dictionary, add it
            if column_num not in columns:
                columns[column_num] = []

            # Append the content to the appropriate column list
            if len(content)>0:
                columns[column_num].extend(self.tokenize(content))
        
        return columns
        
        
    def calculate_token_ratios(self, column_data):
        # Calculate the total number of tokens
        total_tokens = sum(len(tokens) for tokens in column_data.values())

        # Calculate the ratio of tokens in each column to the total number of tokens
        token_ratios = {column: len(tokens) / total_tokens for column, tokens in column_data.items()}

        return token_ratios
        
        
    def calculate_ratio_column_data(self):
        # Get column-wise data
        columns = self.get_column_wise_data()
        print('columns')
        print(columns)
        column_wise_ratio = self.calculate_token_ratios(columns)
        print('column_wise_ratio')
        print(column_wise_ratio)
        return max(column_wise_ratio.values())
        
        
    def calculate_row_wise_average(self):
        total_rows = max(item['row_nums'][0] for item in self.table_res['cells']) + 1
        print('total_rows', total_rows)
        row_content_counts = [0] * total_rows
        row_non_empty_counts = [0] * total_rows

        for item in self.table_res['cells']:
            row_num = item['row_nums'][0]
            content = item['content']
            row_content_counts[row_num] += 1
            if content:
                row_non_empty_counts[row_num] += 1
        print('row_content_counts', row_content_counts)
        print('row_non_empty_counts', row_non_empty_counts)
        row_averages = []
        for i in range(total_rows):
            if row_non_empty_counts[i] == 0:
                row_averages.append(0)
            else:
                row_averages.append(row_non_empty_counts[i] / row_content_counts[i])
                
        print('row_averages', row_averages)
        overall_average = sum(row_averages) / total_rows
        return overall_average
        
    def get_count(self, cells):
        max_tokens = 0
        # Analyze each cell
        for cell in cells:
            content = cell['content']
            # Tokenize the content
            tokens = self.tokenize(content)
            num_tokens = len(tokens)
            
            if num_tokens > max_tokens:
                max_tokens = num_tokens
        return max_tokens
            # Output the cell with the highest number of tokens


    def words_on_line(self, words: List[Dict], row_line: List[int], col_line: List[int], line_height_tolerance: int = 3) -> List[Dict]:
        tokens_on_row_line = []
        tokens_on_col_line = []
        words_on_middle_line = []
        # for key, word in words.items():
        # # for word in words:
        #     # Check if the word's bounding box intersects with the row line (considering some tolerance for height)
        #     for row_lines_ in row_line:
        #         if word['bbox'][1] - line_height_tolerance <= row_lines_[1] <= word['bbox'][3] + line_height_tolerance:
        #             tokens_on_row_line.append(word['text'])

        #     for col_lines_ in col_line:
        #         # if word['bbox'][0] - line_height_tolerance <= col_lines_[0] <= word['bbox'][2] + line_height_tolerance:
        #         if word['bbox'][0] + line_height_tolerance <= col_lines_[0] <= word['bbox'][2] - line_height_tolerance:
        #             tokens_on_col_line.append(word['text'])
      
        for word in words:
            # Check if the word's bounding box intersects with the row line (considering some tolerance for height)
            for row_lines_ in row_line:
                if word['y1'] - line_height_tolerance <= row_lines_[1] <= word['y2'] + line_height_tolerance:
                    tokens_on_row_line.append(word['word'])

            for col_lines_ in col_line:
                if word['x1'] + line_height_tolerance <= col_lines_[0] <= word['x2'] - line_height_tolerance:
                    tokens_on_col_line.append(word['word'])
                    
                    len_width = word['x2'] - word['x1']
                    middle_width = word['x1'] + len_width/2
                    middle_tolerance = 3
                    if ((middle_width - col_lines_[0]) <= middle_tolerance) and (len(word['word'])>2):
                        words_on_middle_line.append(word['word'])
                    
        return tokens_on_row_line, tokens_on_col_line, words_on_middle_line
    
    
    def main(self):
        
        if not len(self.table_res.get('cells', {})):
            return True
        else:
            self.row_bboxes = self.extract_row_bboxes()
            self.column_bboxes = self.extract_column_bboxes()
            
            _, tokens_on_col_line, words_on_middle_line = self.words_on_line(self.ocr_data, self.row_bboxes , self.column_bboxes)
            
            print(words_on_middle_line)
            print(len(tokens_on_col_line))
            print(tokens_on_col_line)
            max_val_sentence_threshold = 3
            max_sentence_overlapping_val_threshold = 2
            
            max_sentence_overlapping_val, max_val_sentence = get_sentence_wise_col_line_overlapping( self.ocr_data, self.column_bboxes)
            print(max_sentence_overlapping_val)
            row_wise_grand_average = self.calculate_row_wise_average()
            print('row_wise_grand_average', row_wise_grand_average)
            
            column_wise_max_ratio = self.calculate_ratio_column_data()
            print('column_wise_max_ratio', column_wise_max_ratio)

            # max column_wise_max_ratio meaning most of hte values are included in single column
            # lower the row_wise_grand_average meaning not all row wise blocks are filled
            if (column_wise_max_ratio > self.column_wise_max_ratio_thresold and row_wise_grand_average < self.column_wise_max_ratio_thresold) \
                or len(tokens_on_col_line) > self.max_tokens_mismatch_thresold or \
                    (max_val_sentence >= max_val_sentence_threshold or max_sentence_overlapping_val >= max_sentence_overlapping_val_threshold):
                return True
            else:
                return False



from PIL import Image
import os

def crop_image(image_path, bbox, output_path):
    """
    Crop the image at image_path using the provided bbox and save the cropped image to output_path.
    
    :param image_path: Path to the input image
    :param bbox: Bounding box as a tuple (left, upper, right, lower)
    :param output_path: Path to save the cropped image
    """
    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Crop the image using the bounding box
            cropped_img = img.crop(bbox)
            # Save the cropped image
            cropped_img.save(output_path)
            print(f"Cropped image saved to {output_path}")
    except Exception as e:
        print(f"Error: {e}")



import pandas as pd
import os


def update_csv_with_new_column(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize the new column
    df['New_result4'] = ""
    
    # Iterate through each row
    for index, row in df.iterrows():
        image_path = row['Image Path']
        cells_status = row['cells_status']
        print(cells_status)
        if cells_status == 'cells_exists':
            # Construct the output path
            image_name = os.path.basename(image_path)
            # output_path = os.path.join(output_folder, image_name)
            text_folder_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/out_put/ROOT/results'
            img_base = os.path.splitext(image_name)[0] 
            if os.path.exists(os.path.join(text_folder_path, img_base + '.jpg'+".txt")):
                json_file_path = os.path.join(text_folder_path, img_base + '.jpg'+".txt")
                
            elif os.path.exists(os.path.join(text_folder_path, img_base + '.png'+".txt")):
                json_file_path = os.path.join(text_folder_path, img_base + '.png'+".txt")
            
            
            if os.path.exists(json_file_path):
                # Check if the file exists, if not continue to the next
                # print(f"Output file {output_path} does not exist.")
                # continue
                        # croped_img_ocr_path = 'client_code/table_extraction_api/data/output/test_output1/other/ocr/MicrosoftTeams-image (3).json'
                # json_file_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/out_put/ROOT/results/pdf27-07.png.txt'
                # json_file_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/out_put/ROOT/results/pdf40_page-0014.png.txt'
                # image_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/Images/pdf27-07.png'
                # image_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/Images/pdf40_page-0014.jpg'
                with open(json_file_path, 'r') as json_file:
                    data = json.load(json_file)
                    
                # if 'table_res' in data and 'cells' in data['table_res']:
                #     if len(data['table_res']['cells']) > 0:
                #         cells_status.append('cells_exists')
                print(data)
                table_res = data['table_res']
                row_info = data['row_info']
                column_info = data['column_info']
                print(column_info)
                table_coords = data['table_res']['table_coords']
                image_name = os.path.basename(image_path)
                crop_images_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/crop_images'
                saving_img_path = os.path.join(crop_images_path, image_name)
                crop_image(image_path, table_coords, saving_img_path)
                struct_unstruct_idyobj = IdentifyMiscellaneous(row_info, column_info, saving_img_path, table_res)
                is_unstructured_table = struct_unstruct_idyobj.main()
                print(is_unstructured_table)
                if is_unstructured_table:
                    result = 'unstructured_img'
                else:
                    result = 'structured_img'
                # Update the new column
                df.at[index, 'New_result4'] = result
        
        # Save the updated DataFrame back to the CSV
    df.to_csv(csv_path, index=False)

# Example usage
csv_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/updated_pred.csv'
update_csv_with_new_column(csv_path)
exit('>>>>>>>>>>')


# Check if the file exists, if not continue to the next
# print(f"Output file {output_path} does not exist.")
# continue
# croped_img_ocr_path = 'client_code/table_extraction_api/data/output/test_output1/other/ocr/MicrosoftTeams-image (3).json'
# json_file_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/out_put/ROOT/results/pdf27-07.png.txt'
json_file_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/out_put/ROOT/results/pdf31-15.png.txt'
# image_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/Images/pdf27-07.png'
image_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/Images/pdf31-15.png'
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)
    
# if 'table_res' in data and 'cells' in data['table_res']:
#     if len(data['table_res']['cells']) > 0:
#         cells_status.append('cells_exists')
print(data)
table_res = data['table_res']
row_info = data['row_info']
column_info = data['column_info']
print(column_info)
table_coords = data['table_res']['table_coords']
image_name = os.path.basename(image_path)
crop_images_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/crop_images'
saving_img_path = os.path.join(crop_images_path, image_name)
crop_image(image_path, table_coords, saving_img_path)
struct_unstruct_idyobj = IdentifyMiscellaneous(row_info, column_info, saving_img_path, table_res)
is_unstructured_table = struct_unstruct_idyobj.main()
print(is_unstructured_table)
if is_unstructured_table:
    result = 'unstructured_img'
else:
    result = 'structured_img'

print(result)
exit('??????????????????????')




















exit('>>>>>>>>>>>>>>>>>>>>>.')
# Predefined hints for identifying columns
hints = {
    "goods_description": ["description"],
    "quantity" : ['quantity'],
    "unit_price": ["rate", "price"],
    "amount": ["amount", "total"]
}

def contains_token(word_list, token):
    for word in word_list:
        if word in token:
            return True
    return False


def find_column_indices(headers, hints):
    # Initialize a dictionary to store the column indices for each key
    column_indices = {
        "goods_description": -1,
        "quantity": -1,
        "unit_price": -1,
        "amount": -1
    }
    
    # Iterate over the headers to find the indices based on hints 
    for header in headers:
        header_text = header['content'].lower()
        for key, hint_list in hints.items():
            if contains_token(hint_list, header_text):
                column_indices[key] = header['pos']
                break
    return column_indices

# Function to parse the input JSON and find the required column indices
def parse_json_and_find_columns(json_input):
    headers = json_input.get('headers', [])
    column_indices = find_column_indices(headers, hints)
    return column_indices

# Example JSON input (provided in the question)

def extract_goods_info(json_input, column_indices):
    cells = json_input['cells']
    goods_info = {}
    current_row = -1

    for cell in cells:
        row = cell['row_nums'][0]
        if row != current_row:
            # If this cell starts a new row, create a new goods entry
            current_row = row
            goods_key = f"goods{current_row + 1}"
            goods_info[goods_key] = {
                'goods_description': '',
                'quantity': '',
                'unit_price': '',
                'amount': ''
            }
        
        # Fill in the details based on the column number
        for key, col_num in column_indices.items():
            if col_num in cell['column_nums']:
                goods_info[goods_key][key] = cell['content']

    return goods_info

def aggregate_goods_info(goods_info):
    aggregated_info = {}
    for key, goods in goods_info.items():
        description = goods['goods_description']
        if description not in aggregated_info:
            aggregated_info[description] = {
                'quantity': 0,
                'amount': 0
            }

        # Add quantity if it is a number
        if goods['quantity'].replace('.', '', 1).isdigit():
            aggregated_info[description]['quantity'] += float(goods['quantity'])
        
        # Add amount if it is a number
        if goods['amount'].replace('.', '', 1).isdigit():
            aggregated_info[description]['amount'] += float(goods['amount'])

    # Convert aggregated_info back to goods_info format
    final_goods_info = {}
    counter = 1
    for description, data in aggregated_info.items():
        final_goods_info[f'goods{counter}'] = {
            'goods_description': description,
            'quantity': str(data['quantity']),
            'unit_price': '',
            'amount': str(data['amount'])
        }
        counter += 1

    return final_goods_info



#############################################################
#############################################################
# Example JSON input (provided in the question)
json_input = {
    "table_coords": [104, 898, 1594, 1281],
    "headers": [
        {"pos": 0, "content": "marks & nos", "bbox": [0, 0, 467, 27]},
        {"pos": 1, "content": "no & kind of pkgs description", "bbox": [467, 0, 838, 27]},
        {"pos": 2, "content": "description", "bbox": [838, 0, 1067, 27]},
        {"pos": 3, "content": "quantity", "bbox": [1067, 0, 1211, 27]},
        {"pos": 4, "content": "rate", "bbox": [1211, 0, 1316, 27]},
        {"pos": 5, "content": "amount", "bbox": [1316, 0, 1489, 27]}
    ],
    "cells": [
        {"content": "cold dough premix fdpz3004", "bbox": [0, 86, 467, 116], "column_nums": [0], "row_nums": [0]},
        {"content": "11 ctn cold dough premix", "bbox": [467, 86, 838, 116], "column_nums": [1], "row_nums": [0]},
        {"content": "", "bbox": [838, 86, 1067, 116], "column_nums": [2], "row_nums": [0]},
        {"content": "11.0000", "bbox": [1067, 86, 1211, 116], "column_nums": [3], "row_nums": [0]},
        {"content": "cif 38.324", "bbox": [1211, 86, 1316, 116], "column_nums": [4], "row_nums": [0]},
        {"content": "421.560", "bbox": [1316, 86, 1489, 116], "column_nums": [5], "row_nums": [0]},
        {"content": "cold dough premix fdpz3004 net wt : 38.75 kg batch no sb121288 dom : 22 august 2011 bbf 21 december 2011", "bbox": [0, 86, 467, 383], "column_nums": [0], "row_nums": [1]},
        {"content": "11 ctn cold dough premix packed in 775 g x 50 pkt x 1 sack 11 x 38.75 426.2500", "bbox": [467, 86, 838, 383], "column_nums": [1], "row_nums": [1]},
        {"content": "woven 426.2500 kgs", "bbox": [838, 86, 1067, 383], "column_nums": [2], "row_nums": [1]},
        {"content": "11.0000 ctn", "bbox": [1067, 86, 1211, 383], "column_nums": [3], "row_nums": [1]},
        {"content": "cif 38.324 per ctn", "bbox": [1211, 86, 1316, 383], "column_nums": [4], "row_nums": [1]},
        {"content": "421.560", "bbox": [1316, 86, 1489, 383], "column_nums": [5], "row_nums": [1]}
    ]
}

with open('/datadrive/table_res_trans/miscellaneous_code/src/samples/Covering_Schedule_254_page_5.json', 'r') as exp:
    json_input = json.load(exp)


# print(json_input)
# Find the column indices for "goods description", "unit price", and "amount"
column_indices = parse_json_and_find_columns(json_input)


# Column indices obtained earlier
# column_indices = {'goods_description': 2, 'quantity': 3, 'unit_price': 4, 'amount': 5}

# Extract goods information
print('column_indices', column_indices)
goods_info = extract_goods_info(json_input, column_indices)
print("Extracted Goods Info:")
print(goods_info)

# Aggregate goods information
aggregated_goods_result = aggregate_goods_info(goods_info)
print("Aggregated Goods Info:")
print(aggregated_goods_result)




