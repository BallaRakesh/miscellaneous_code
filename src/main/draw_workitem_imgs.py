
import os
import json
import os
import json
import cv2  # Import OpenCV




def is_text_nearby(text_positions, x, y, threshold=50):
    for pos_x, pos_y in text_positions:
        if abs(pos_x - x) < threshold and abs(pos_y - y) < threshold:
            return True
    return False

def draw_bbox(image_path, pdf_data, output_folder):
    # Extract the image number from the filename (e.g., pdf10_page-0003.jpg -> 3)
    base_name = os.path.basename(image_path)
    # # Assuming the format is always 'pdf<doc_number>_page-<page_number>.ext'
    # print(base_name)
    
    # doc_number, page_part = base_name.split('_page-')
    # page_number = int(page_part.split('.')[0])  # Extract the page number

    # # Construct the key used in the JSON data
    # json_key = f"{doc_number}_{page_number}.png"  # e.g., "pdf10_3.png"
    json_key = base_name
    print('base_name >>>>>>>>>>>>>>>', base_name)
    print('json_key>>>>>>>>>>>>>>>>', json_key)
    if json_key in pdf_data:
        img_data = pdf_data[json_key]
        if "keys_extraction" in img_data:
            keys_extraction = img_data["keys_extraction"]
            if keys_extraction:
                # Read the image using OpenCV
                image = cv2.imread(image_path)
                text_positions = set()
                # Iterate over each extracted key and draw bounding boxes
                for key, value in keys_extraction.items():
                    coordinates_lst = value.get("coordinate", [])
                    print('coordinates_lst >>$$$$$$$$$$$$$',coordinates_lst)
                    for coordinates in coordinates_lst:
                        
                        # Extract the top-left coordinates for text placement
                        top_left_x, top_left_y = coordinates[0], coordinates[1]
                        top_left_x, top_left_y, bottom_right_x, bottom_right_y = coordinates
                        # Draw the text on the image at the specified coordinates
                        cv2.rectangle(image, 
                            (top_left_x, top_left_y), 
                            (bottom_right_x, bottom_right_y), 
                            (255, 0, 0),  # Green color for bbox
                            3)  # Thickness of the bbox
                        # Adjust text position to avoid overlapping
                        text_y = top_left_y
                        while (top_left_x, text_y) in text_positions:
                            text_y -= 10  # Move text upward if there's already text here
                            
                        if "date" in key and is_text_nearby(text_positions, top_left_x, text_y):
                            continue
                        # Store the new text position
                        text_positions.add((top_left_x, text_y))

                        # Draw the text on the image
                        cv2.putText(
                            image,
                            key,
                            (top_left_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,  # Font scale (adjust as needed)
                            (0, 0, 255),  # Red color for text
                            2,  # Thickness of the text
                            cv2.LINE_AA  # Anti-aliased line type for better quality
                        )
                output_path = os.path.join(output_folder, os.path.basename(image_path))
                cv2.imwrite(output_path, image)
                print(f"Saved annotated image to {output_path}")
        else:
            print(f"No keys_extraction found for {json_key}")
    else:
        print(f"No data found for {json_key} in the JSON file.")
        
        
        
        
if __name__ == '__main__':
    workitem_data = '/home/ntlpt19/Desktop/TF_release/TradeGPT/ITF_data/pdf16.json'
    pdf_folder = '/home/ntlpt19/TF_testing_EXT/ITF_TESTING/PDFS_imgs_10-16/NEW_PDFS/pdf16/imgs'
    out_put_folder = '/home/ntlpt19/TF_testing_EXT/ITF_TESTING/draw_pdfs'
    folder_name = 'pdf16'
    if not os.path.exists(os.path.join(out_put_folder, folder_name)):
        os.makedirs(os.path.join(out_put_folder, folder_name))
        
    # Open the JSON file and load its contents
    with open(workitem_data, 'r') as file:
        pdf_data = json.load(file)
    ext_data = pdf_data.get('document_extraction_result', {}).get('extraction_result', {})
    
    for pdf_img in os.listdir(pdf_folder):
        if pdf_img.endswith('.png') or pdf_img.endswith('.jpg'):
            image_path = os.path.join(pdf_folder, pdf_img)
            draw_bbox(image_path, ext_data, os.path.join(out_put_folder, folder_name))
            
