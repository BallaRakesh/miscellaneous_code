import os
import base64
import requests
import json
root_path = '/home/ntlpt19/TF_testing_EXT/ROOT'
out_put_path = '/home/ntlpt19/TF_testing_EXT/output'

import cv2
import matplotlib.pyplot as plt


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

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # Read the binary data of the image
        image_binary = image_file.read()

        # Encode the binary data into Base64
        base64_encoded = base64.b64encode(image_binary)

        # Decode the bytes to a UTF-8 string
        base64_string = base64_encoded.decode("utf-8")

    return base64_string


for doc_type in os.listdir(root_path):
    for imgs_folder in os.listdir(os.path.join(root_path, doc_type)):
        for imgs in os.listdir(os.path.join(root_path, doc_type, imgs_folder)):
            img_base64 = image_to_base64((os.path.join(root_path, doc_type, imgs_folder, imgs)))
            # doc_class = doc_type
            # image=request.image
            # ground_truth=request.docClass
            # doc_type = 'CS'
            input_dict = {'image':img_base64, 'docClass':doc_type}
            # x =  request.json()  # request in str
            # x = json.loads(x) 
            url = 'http://192.168.170.11:8192/transformer/trade-finance/extract/'
            response = requests.post(url, json=input_dict)
            x = response.json()
            print('*************')
            print(type(x))
            #saving the results
            result_path = os.path.join(out_put_path, doc_type)
            if not os.path.exists(result_path):
                os.mkdir(result_path)
            result_path = os.path.join(out_put_path, doc_type, "results")
            if not os.path.exists(result_path):
                os.mkdir(result_path)
            result_plot = os.path.join(out_put_path, doc_type, "results_plot")
            if not os.path.exists(result_plot):
                os.mkdir(result_plot)
            with open(os.path.join(result_path,f"{imgs}.txt"), "w") as f:
                json.dump(x, f)
                
            
            draw_bounding_boxes((os.path.join(root_path, doc_type, imgs_folder, imgs)), x['keys_extraction'], os.path.join(result_plot, imgs))   
                
            print(x)
            print('*************')