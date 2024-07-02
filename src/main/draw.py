import cv2


def draw_single(image_path, bbox, output_path):
    image = cv2.imread(image_path)

    # Define the color and thickness of the line
    color = (0, 255, 0)  # Green color in BGR
    thickness = 3  # Thickness of 2 pixels

    # Draw the line on the image
    cv2.line(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

    # Save the modified image
    cv2.imwrite(output_path, image)

def draw_multiple(image_path, bboxes, output_path):
    image = cv2.imread(image_path)

    # Define the color and thickness of the line
    color = (0, 255, 0)  # Green color in BGR
    thickness = 3  # Thickness of 2 pixels

    # Draw lines for each bounding box on the image
    for bbox in bboxes:
        cv2.line(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

    # Save the modified image
    cv2.imwrite(output_path, image)


'''
column_res {'single_line_elements': [2, 3, 4, 5], 'column_seperators': [[[0, 0, 0, 384], ''], ([467, 0, 467, 384], 'solid_lines'), ([1067, 0, 1067, 384], 'solid_lines'), ([1211, 0, 1211, 384], 'solid_lines'), ([1316, 0, 1316, 384], 'solid_lines'),
([467, 0, 467, 384], 'solid_lines'), [[838, 0, 838, 384], ''], ([1067, 0, 1067, 384], 'solid_lines'), ([1211, 0, 1211, 384], 'solid_lines'), 
([1316, 0, 1316, 384], 'solid_lines'), [[1489, 0, 1489, 384], '']], 
'meta': [{'header_cell_content': 'marks & nos', 'header_cell_coordinate': [0, 0, 467, 28], 'full_column_coordinate': [0, 0, 467, 384], 'column_without_header': [0, 29, 467, 384]}, {'header_cell_content': 'no & kind of pkgs description', 'header_cell_coordinate': [467, 0, 838, 28], 'full_column_coordinate': [467, 0, 838, 384], 'column_without_header': [467, 29, 838, 384]}, {'header_cell_content': 'description', 'header_cell_coordinate': [838, 0, 1067, 28], 'full_column_coordinate': [838, 0, 1067, 384], 'column_without_header': [838, 29, 1067, 384]}, {'header_cell_content': 'quantity', 'header_cell_coordinate': [1067, 0, 1211, 28], 'full_column_coordinate': [1067, 0, 1211, 384], 'column_without_header': [1067, 29, 1211, 384]}, {'header_cell_content': 'rate', 'header_cell_coordinate': [1211, 0, 1316, 28], 'full_column_coordinate': [1211, 0, 1316, 384], 'column_without_header': [1211, 29, 1316, 384]}, {'header_cell_content': 'amount', 'header_cell_coordinate': [1316, 0, 1489, 28], 'full_column_coordinate': [1316, 0, 1489, 384], 'column_without_header': [1316, 29, 1489, 384]}]}
'''
row_res = [{'approach': 'cv', 'coords': [0, 160, 1490, 160]}, {'approach': 'cv', 'coords': [0, 243, 1490, 243]}, {'approach': 'cv', 'coords': [0, 87, 1490, 87]}, {'approach': 'cv', 'coords': [0, 117, 1490, 117]}, {'approach': 'cv', 'coords': [0, 202, 1490, 202]}]

col_bboxs = [[467, 0, 467, 384], [838, 0, 838, 384],[1211, 0, 1211, 384], [1316, 0, 1316, 384], [1489, 0, 1489, 384], [2, 3, 4, 5], [0, 0, 0, 384], [467, 0, 467, 384], [1067, 0, 1067, 384], [1211, 0, 1211, 384], [1316, 0, 1316, 384]]

row_bboxs = [row_bbox.get('coords', []) for row_bbox in row_res]

print('row_bboxs')
print(row_bboxs)
print('col_bboxs', col_bboxs)
# Define the bounding box coordinates
bbox = [0, 115, 1490, 115]

output_path = '/datadrive/table_res_trans/miscellaneous_code/src/out_puts/output_with_line.png'  # Path to save the output image
# Load the image
image_path = '/datadrive/clasification_testing_stru_unstru/PDF_101/crop_images/pdf27-07_.png'  # Update the path to your image
image = cv2.imread(image_path)
bbox = [115, 0, 115, image.shape[0]] 
bbox = [[0, 160, 1490, 160],[0, 243, 1490, 243]]
bbox = [2153, 693, 2273, 738]
#21, 62, 1467, 6), (21, 55, 1467, 7), (20, 1, 1468, 6), (561, 0, 927, 3)

# [(1317, 106, 2, 270), (1213, 104, 2, 268), (1067, 104, 2, 268), (469, 104, 2, 269)]
# [(10, 392, 436, 2), (11, 62, 1471, 6), (11, 55, 1471, 7), (10, 1, 1472, 6), (551, 0, 931, 3)]
# print(bbox)
draw_single(image_path, bbox, output_path)

col_bboxs = [[13, 5, 119, 34], [121, 5, 143, 34], [147, 5, 213, 34], [212, 5, 227, 34], [487, 4, 548, 36], [548, 6, 573, 36], [572, 6, 647, 37], [659, 8, 698, 39], [704, 9, 789, 41], [1144, 12, 1332, 45], [1343, 12, 1378, 45], [1386, 12, 1499, 45], [1922, 14, 2064, 49]]
print(len(col_bboxs))
# draw_multiple(image_path, col_bboxs, output_path)