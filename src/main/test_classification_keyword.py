from ext_test import image_to_base64
from get_ocr_gv_pytesseract_function import get_ocr_vision_api_charConfi

# I'll rewrite the script to convert the given word coordinates to the desired format.

def informat_classification(original_list):
    # Convert the list to the new format
    converted_list = []
    for item in original_list:
        word_coordinates = [item['x1'], item['y1'], item['x2'], item['y2']]
        new_item = {
            "word": item['word'],
            "word_coordinates": word_coordinates,
            "x1": item['x1'],
            "y1": item['y1'],
            "x2": item['x2'],
            "y2": item['y2'],
            "top": item['top'],
            "left": item['left'],
            "confidence": item['confidence']
        }
        converted_list.append(new_item)
    return converted_list



img_base64 = image_to_base64('/home/ntlpt19/TF_testing_EXT/IMPCertificates/IMPCertificates/COS/Images/Certificate_Of_Origin_347_page_13.png')
wc, all_text = get_ocr_vision_api_charConfi('/home/ntlpt19/TF_testing_EXT/IMPCertificates/IMPCertificates/COS/Images/Certificate_Of_Origin_347_page_13.png')
updated_wc = informat_classification(wc)
# File path to save the JSON file
file_path = 'payload_classification.json'
import json

input_dict = {'image':img_base64, 'ocr_data':updated_wc, "product_name":"Import LC"}

# Save the original dictionary list to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(input_dict, json_file, indent=4)