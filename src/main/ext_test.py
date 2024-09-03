import os
import base64
import requests
import json
import cv2
import matplotlib.pyplot as plt


import cv2
import json

def draw_bounding_boxes_allpage(image_path, extraction_data, save_img_path):
    # Read the image using cv2
    img = cv2.imread(image_path)
    print(image_path)
    
    # Iterate over the keys_extraction dictionary
    for page, elements in extraction_data.items():
        for key, values in elements.items():
            print(values)
            for value in values:
                text, bbox, confidence = value
                print(bbox)
                # Draw bounding box on the image
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                
                # Add label text
                cv2.putText(img, key, (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(img, f'{confidence:.2f}%', (bbox[0], bbox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save the modified image
    cv2.imwrite(save_img_path, img)


# Call the function with sample data

def draw_bounding_boxes(image_path, extraction_data, save_img_path):
    # Read the image using cv2
    img = cv2.imread(image_path)
    print(image_path)
    # Iterate over the keys_extraction dictionary
    for key, data in extraction_data.items():
        print(data)
        if 'coordinate' in data:
            # Extract bounding box coordinates
            for bbox in data['coordinate']:
                print(bbox)
                # Draw bounding box on the image
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

                # Add label text
                cv2.putText(img, key, (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Save the modified image
    cv2.imwrite(save_img_path, img)

def read_text_file(text_file):
    with open(text_file, 'r') as file:
        content = file.read()
        # print(content)
    return content

def read_wc_file(datajson_file):
    with open(datajson_file, 'r') as json_file:
        data = json.load(json_file)
    return data

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # Read the binary data of the image
        image_binary = image_file.read()

        # Encode the binary data into Base64
        base64_encoded = base64.b64encode(image_binary)

        # Decode the bytes to a UTF-8 string
        base64_string = base64_encoded.decode("utf-8")

    return base64_string

def main_ext(root_path, api_end_point):
    for doc_type in os.listdir(root_path):
        all_text_folder = os.path.join(root_path, doc_type, 'all_text')
        ocr_folder = os.path.join(root_path, doc_type, 'OCR')

        for imgs in os.listdir(os.path.join(root_path, doc_type, 'Images')):
            # for imgs in os.listdir(os.path.join(root_path, doc_type, 'Images', imgs_folder)):
            img_base64 = image_to_base64((os.path.join(root_path, doc_type, 'Images', imgs)))
            try:
                all_text_data = read_text_file(os.path.join(all_text_folder, imgs[:-4]+'_all_text.txt'))
                wc_data = read_wc_file(os.path.join(ocr_folder, imgs[:-4]+'_text.txt'))
                print(wc_data)
            except:
                continue
            input_dict = {'image':img_base64, 'docClass':doc_type, 'ocr_path_word':wc_data, 'ocr_path_text': all_text_data}

            response = requests.post(api_end_point, json=input_dict)
            x = response.json()
            print('*************')
            print(type(x))
            #saving the results
            out_put_path = os.path.join(root_path, 'LMV2_OUTPUT')
            if not os.path.exists(out_put_path):
                os.mkdir(out_put_path)
            result_path = os.path.join(out_put_path, doc_type)
            if not os.path.exists(result_path):
                os.mkdir(result_path)
            result_path = os.path.join(out_put_path, doc_type, "RESULTS")
            if not os.path.exists(result_path):
                os.mkdir(result_path)
            result_plot = os.path.join(out_put_path, doc_type, "RESULTS_PLOT")
            if not os.path.exists(result_plot):
                os.mkdir(result_plot)
            with open(os.path.join(result_path,f"{imgs}.txt"), "w") as f:
                json.dump(x, f)
                
            draw_bounding_boxes_allpage((os.path.join(root_path, doc_type, 'Images', imgs)), x, os.path.join(result_plot, imgs))
            print(x)
            print('*************')
        
        
if __name__ == '__main__':
    root_path = '/home/ntlpt19/LLM_training/EVAL'
    api_end_point = 'http://192.168.170.11:8192/transformer/trade-finance/extract/'

    # main_ext(root_path, api_end_point)