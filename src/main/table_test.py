import os
import base64
import requests
import json
import cv2
import matplotlib.pyplot as plt
import pandas as pd


root_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/Master'
complete_image_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/Images'

out_put_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/out_put'

draw_img_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/draw_res'

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

import shutil
import os

def copy_image_to_folder(image_path, image_file_name):
    # Check if the destination folder exists, create it if it doesn't


    # Extract the image file name from the path

    # Create the destination path
    destination_path = os.path.join(draw_img_path, image_file_name)

    # Copy the image to the destination folder
    shutil.copy(image_path, destination_path)
    print(f"Image copied to {destination_path}")

def write_image_paths_to_file(image_paths_):
    with open('failed_imgs.txt', 'w') as file:
        for image_path in image_paths_:
            file.write(image_path + '\n')



image_paths = []
actual_labels = []
predicted_labels = []
failed_images = []
for doc_type in os.listdir(root_path):
    for imgs_folder in os.listdir(os.path.join(root_path, doc_type)):
        for imgs in os.listdir(os.path.join(root_path, doc_type, imgs_folder)):
            try:
                img_base = os.path.splitext(imgs)[0] 
                if os.path.exists(os.path.join(complete_image_path, img_base + '.jpg')):
                    finalized_paths = os.path.join(complete_image_path, img_base + '.jpg')
                    
                elif os.path.exists(os.path.join(complete_image_path, img_base + '.png')):
                    finalized_paths = os.path.join(complete_image_path, img_base + '.png')
                else:
                    failed_images.append(imgs)
                    
                img_base64 = image_to_base64(finalized_paths)
                # doc_class = doc_type
                # image=request.image
                # ground_truth=request.docClass
                # doc_type = 'CS'
                image_paths.append(finalized_paths)
                input_dict = {'image':img_base64, 'layout_result':{}}
                # x =  request.json()  # request in str
                # x = json.loads(x) 
                url = 'http://10.2.3.12:8100/transformer/trade-finance/table-extraction/'
                response = requests.post(url, json=input_dict)
                x = response.json()
                print('*************')
                print(type(x))
                print(x)
                try:
                    copy_image_to_folder(x.get('image_path', ""), imgs)
                except:
                    print('NOOO')
                pred = x.get('is_unstructured_table', "NONE")
                print(type(pred))
                if pred == True:
                    predicted_labels.append('unstructured_img')
                else:
                    predicted_labels.append('structured_img')
                    
                actual_labels.append(imgs_folder)

                    
                #saving the results
                result_path = os.path.join(out_put_path, doc_type)
                if not os.path.exists(result_path):
                    os.mkdir(result_path)
                result_path = os.path.join(out_put_path, doc_type, "results")
                if not os.path.exists(result_path):
                    os.mkdir(result_path)
                # result_plot = os.path.join(out_put_path, doc_type, "results_plot")
                # if not os.path.exists(result_plot):
                #     os.mkdir(result_plot)
                with open(os.path.join(result_path,f"{imgs}.txt"), "w") as f:
                    json.dump(x, f)
                    
                # draw_bounding_boxes((os.path.join(root_path, doc_type, imgs_folder, imgs)), x['keys_extraction'], os.path.join(result_plot, imgs))   
            except:
                failed_images.append(finalized_paths)
                print('bug')
                
data = {
    "Image Path": image_paths,
    "Actual Label": actual_labels,
    "Predicted Label": predicted_labels
}

df = pd.DataFrame(data)

# Save the DataFrame as a CSV file
csv_file_path = "image_predictions.csv"
df.to_csv(csv_file_path, index=False)

write_image_paths_to_file(failed_images)