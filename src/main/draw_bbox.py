import cv2
import json
import os

# Path to the JSON file
file_path = '/home/ntlpt19/TF_testing_EXT/dummy_responces/Ext_Individual'
# for i in range(8, 25):
for files in os.listdir(file_path):
    print(files)
    json_file_path = os.path.join('/home/ntlpt19/TF_testing_EXT/dummy_responces/Ext_Individual',files)
    # json_file_path = f"/home/ntlpt19/TF_testing_EXT/dummy_responces/Ext_Individual/COO_DocProcessing-0000015577-process_{i}.json"  # Replace with your JSON file path

    # Given file path

    # Extract the file name with extension
    file_name_with_extension = os.path.basename(json_file_path)
    # Remove the extension to get the file name without extension
    file_name_without_extension = os.path.splitext(file_name_with_extension)[0]

    print("File name with extension:", file_name_without_extension)
    print("File name without extension:", len(file_name_without_extension.split('_')[0]))
    # Load JSON data from the file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    # Path to the image
    image_path = f"/home/ntlpt19/TF_testing_EXT/dummy_responces/InputImages/{file_name_without_extension[len(file_name_without_extension.split('_')[0])+1:]}.png"  # Replace with your image path
    print(image_path)
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Iterate over each field in the JSON data
    for key, info in data['keys_extraction'].items():
        values = info.get("value", '')
        if values:
            coordinates_list = info["coordinate"]
            
            # Ensure `values` is a list to handle multiple values
            if not isinstance(values, list):
                values = [values]
            
            # Draw each value at the corresponding coordinates
            for i, coordinates in enumerate(coordinates_list):
                value = values[i] if i < len(values) else values[-1]  # Use the last value if there are more coordinates than values
                # Extract the top-left coordinates for text placement
                top_left_x, top_left_y = coordinates[0], coordinates[1]
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = coordinates
                # Draw the text on the image at the specified coordinates
                cv2.rectangle(image, 
                      (top_left_x, top_left_y), 
                      (bottom_right_x, bottom_right_y), 
                      (255, 0, 0),  # Green color for bbox
                      3)  # Thickness of the bbox
                cv2.putText(image, 
                            key, 
                            (top_left_x, top_left_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1,  # Font scale (adjust as needed)
                            (0, 0, 0),  # Font color (black)
                            2,  # Thickness of the text
                            cv2.LINE_AA)  # Anti-aliased line type for better quality

    # Save the edited image with the same name
    cv2.imwrite(image_path, image)

    print(f"Image saved with annotations at {image_path}")