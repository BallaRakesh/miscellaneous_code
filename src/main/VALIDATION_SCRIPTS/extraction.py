import os
import base64
import requests
import json
import cv2

def draw_bounding_boxes_allpage(image_path, extraction_data, save_img_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image at {image_path} could not be loaded.")
        return

    for key, elements in extraction_data['keys_extraction'].items():
        text, bbox, confidence = elements['value'] if 'value' in elements.keys() else '',elements['coordinate'] if 'coordinate' in elements.keys() else [],elements['model_confidence'] if 'model_confidence' in elements.keys() else 0
        if text == '':
            continue
        if bbox == []:
            continue
        bbox = bbox[0]
        confidence = confidence[0]
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        cv2.putText(img, key, (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, f'{confidence:.2f}%', (bbox[0], bbox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(save_img_path, img)

def draw_bounding_boxes(image_path, extraction_data, save_img_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image at {image_path} could not be loaded.")
        return

    for key, data in extraction_data.items():
        for data_keys in data.keys():
            lst = data[data_keys]
            for elem in lst:
                bbox = elem[1]
        # if 'coordinate' in data:
        #     for bbox in data['coordinate']:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(img, data_keys, (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(save_img_path, img)

def read_text_file(text_file):
    try:
        with open(text_file, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: File {text_file} not found.")
        return None

def read_wc_file(datajson_file):
    try:
        with open(datajson_file, 'r') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"Error: File {datajson_file} not found.")
        return None

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            image_binary = image_file.read()
            base64_encoded = base64.b64encode(image_binary)
            base64_string = base64_encoded.decode("utf-8")
        return base64_string
    except FileNotFoundError:
        print(f"Error: Image file {image_path} not found.")
        return None

def extract_model_data(json_result):
    return json_result['original_responce']

def extract_normal_data(json_result):
    return json_result['formatted_responce']

def main_ext(root_path, api_end_point, doc_type):
    images_path = os.path.join(root_path, doc_type, 'Images')
    if not os.path.exists(images_path):
        print(f"Error: Path {images_path} does not exist.")
        return

    for img_file in os.listdir(images_path):
        if os.path.exists(os.path.join('/home/administrator/Vasu/EXTRACTION_WORK/EXTRACTION_RESULTS/CI/RESULTS_PLOT/MODEL_PLOTS',img_file)):
            continue
        print("The current image file is:",img_file)
        img_path = os.path.join(images_path, img_file)
        img_base64 = image_to_base64(img_path)
        if img_base64 is None:
            continue
        
        ocr_path = os.path.join('/datadrive/rakesh/OCR', doc_type, 'OCR')        
        all_text_path = os.path.join(ocr_path, 'AllText', f'{img_file[:-4]}.txt')     
        word_coordinates_path = os.path.join(ocr_path, 'WordCoordinates', f'{img_file[:-4]}.json')
        
        input_dict = {
            'image': img_base64,
            'docClass': doc_type,
            # 'ocr_path_word':word_coordinates_path,
            # 'ocr_path_text':all_text_path
        }
        response = requests.post(api_end_point, json=input_dict)
        x = response.json()
        print(x)
        output_path = '/home/administrator/Vasu/EXTRACTION_WORK/EXTRACTION_RESULTS'
        os.makedirs(output_path, exist_ok=True)
        result_path = os.path.join(output_path, doc_type)
        os.makedirs(result_path, exist_ok=True)
        
        model_path = os.path.join(result_path, "MODEL_RESULTS")
        os.makedirs(model_path, exist_ok=True)
        
        post_processing_path = os.path.join(result_path, "POST_PROCESSING_RESULTS")
        os.makedirs(post_processing_path, exist_ok=True)
        
        result_plot = os.path.join(result_path, "RESULTS_PLOT")
        os.makedirs(result_plot, exist_ok=True)
        
        result_model_plot = os.path.join(result_plot, "MODEL_PLOTS")
        os.makedirs(result_model_plot, exist_ok=True)
        
        pp_plots = os.path.join(result_plot, "POST_PROCESSING_PLOTS")
        os.makedirs(pp_plots, exist_ok=True)
        
        with open(os.path.join(model_path, f'{img_file}_model_results.txt'), 'w') as f:
            json.dump(extract_model_data(x), f)
        
        with open(os.path.join(post_processing_path, f'{img_file}_post_processing_results.txt'), 'w') as f:
            json.dump(extract_normal_data(x), f)
                
        draw_bounding_boxes_allpage(img_path, x['formatted_responce'], os.path.join(result_model_plot, img_file))
        draw_bounding_boxes(img_path, x['original_responce'], os.path.join(pp_plots, img_file))

if __name__ == '__main__':
    root_path = '/home/administrator/Vasu/EXTRACTION_WORK/TESTING'
    api_end_point = 'http://10.2.3.14:8177/transformer/trade-finance/extract/'
    main_ext(root_path, api_end_point, 'CI')
