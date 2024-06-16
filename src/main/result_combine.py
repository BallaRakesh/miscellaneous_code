import fitz
import os
from google.cloud import vision
from base64 import b64encode
import glob
import json
 
from master_table_extraction.utils import load_config
cfg = load_config('../main/config.ini')
gv_key = cfg['GV']['KEY']
 

'''def cell_cord(row_info, column_info, header):
    cell2 = dict()
    row_coordinate = {}
    for i in row_info:
        row_coordinate[i] = row_info[i].get('coords')
    print('row_coordinate >>>>>>>>>>>>>>>>>>>>>>', row_coordinate)
    column_coordinate = {}
    try:
        for i in range(len(column_info['column_seperators'])):
            column_coordinate[i]= column_info['column_seperators'][i][0]
        print('column_coordinate >>>>>>>>>>>>>>>>', column_coordinate)
        row_box = []
        for i in range(len((list(row_coordinate.keys())))-1):
            temp1 = row_coordinate.get(i)
            temp2 = row_coordinate.get(i+1)
            rect = [temp1[0], temp1[1], temp2[2], temp2[3]]
            row_box.append(rect)
        print('row_box >>>>>>>>>>>>>>>>', row_box)
        col_box = []
        for i in range(len((list(column_coordinate.keys())))-1):
            temp1 = column_coordinate.get(i)
            temp2 = column_coordinate.get(i+1)
            rect = [temp1[0], temp1[1], temp2[2], temp2[3]]
            col_box.append(rect)
        print('col_box  >>>>>>>>>>>>>>', col_box)
        for i in range(len(col_box)):
            c = []
            col_obj = fitz.Rect(col_box[i])
            for row in row_box:
                row_obj = fitz.Rect(row)
                cell = fitz.Rect.intersect(col_obj, row_obj)
                c.append(list(cell))
            try:
                cell2[header[i]] = c
            except:
                pass
        print('cell2 >>>>>>>>>>>>>>>>>>>>>>', cell2)
    except:
        pass
    return cell2
 
 
def zone_wise_ocr(image_path, zones, header):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gv_key #"/New_Volume/number_theory/GEO_Rakesh_original/spheric-time-383904-f1b421d86eef.json"
 
    with open(image_path, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
 
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content = ctxt)
 
    response = client.text_detection(image=image)
    document = response.full_text_annotation
    
    result = {}
    # print("zone is =========>", zones)
    for i in header:
        col = {}
        # print("zones is ========>", zones[i])
        if i in zones and zones[i] and zones[i][0]:
            for j in range(len(zones[i][0])):
                text_within_bbox = ""
                try:
                    for page in document.pages:
                        for block in page.blocks:
                            for paragraph in block.paragraphs:
                                for word in paragraph.words:
                                    word_text = ''.join([symbol.text for symbol in word.symbols])
                                    vertices = [(vertex.x, vertex.y) for vertex in word.bounding_box.vertices]
                                    if all(zones[i][j][0] <= x <= zones[i][j][2] and
                                            zones[i][j][1] <= y <= zones[i][j][3] for x, y in
                                            vertices):
                                        text_within_bbox += word_text + " "
                except:
                    pass                
                if len(text_within_bbox)!=0:
                    col[j] = text_within_bbox
        result[i] = col
    return result'''
 

def zone_wise_ocr(annotations, zones, header, row_wise_bbox):
        
    result = {}
    master_list =[]
    print("zone is =========>", zones)
    for idx, i in enumerate(header):
        result["COLUMN_" + str(idx+1) + "_HEADER"] = str(i)
        if i in zones:
            for row_idx, row_wise in enumerate(row_wise_bbox):
                print(row_idx, 'row_wise data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', row_wise)
                text_within_bbox = ""
                
                for j in range(len(zones[i])):
                    for word in annotations:
                        word_text = word["word"]
                        master_vertices = [[vertex[0], vertex[1], vertex[2], vertex[3]] for vertex in word["vertices"]]
                        # print(master_vertices[0])
                        # exit('>>??>>')
                        vertices = [(vertex[0], vertex[1]) for vertex in word["vertices"]]
                        
                        if vertices not in master_list and master_vertices[0] in row_wise: 
                            if all(zones[i][j][0] <= x <= zones[i][j][2] and
                                    zones[i][j][1] <= y <= zones[i][j][3] for x, y in
                                    vertices):
                                text_within_bbox += word_text + " "
                                master_list.append(vertices)
                print('text_within_bbox $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$', text_within_bbox)
                if len(text_within_bbox)!=0:
                    result["ROW_" + str(row_idx+1) + "_COLUMN_" + str(idx+1)] = text_within_bbox
    return result  


import ast
ast.literal_eval

def currect_row_dict(row_coordinate):
    modified_dict ={}
    count = 0
    for idx, data in row_coordinate.items():
        modified_dict[str(count)] = data
        count = count+1
    return modified_dict

def cell_cord(row_info, column_info, header):
    row_coordinate = {}
    for i in row_info:
        row_coordinate[str(int(i))] = row_info[i].get('coords')
    column_coordinate = {}
    # for i in range(len(ast.literal_eval(column_info.get("column_seperators")))):
    #     column_coordinate[i]= ast.literal_eval(column_info.get("column_seperators"))[i][0]
        
    for i in range(len(column_info['column_seperators'])):
        column_coordinate[i]= column_info['column_seperators'][i][0]
    print('column_coordinate >>>>>>>>>>>>>>>>', column_coordinate)
    print('row_coordinate >>>>>>>>>>>>>>>>', row_coordinate)
    row_coordinate = currect_row_dict(row_coordinate)
    print("???", row_coordinate)
    row_coordinate[str(len(row_coordinate))] = [row_coordinate[str(len(row_coordinate)-1)][0], column_coordinate[0][3], row_coordinate[str(len(row_coordinate)-1)][2], column_coordinate[0][3]]
    print("row_coordinate :: ",row_coordinate)
    print("column_coordinate :: ",column_coordinate)
    row_box = []
    for i in range(len((list(row_coordinate.keys())))-1):
        temp1 = row_coordinate.get(str(i))
        temp2 = row_coordinate.get(str(i+1))
        rect = [temp1[0], temp1[1], temp2[2], temp2[3]]
        row_box.append(rect)
    print("ROW BOXES    : ", row_box)
    col_box = []
    for i in range(len((list(column_coordinate.keys())))-1):
        temp1 = column_coordinate.get(i)
        temp2 = column_coordinate.get(i+1)
        rect = [temp1[0], temp1[1], temp2[2], temp2[3]]
        col_box.append(rect)
    print("COLUMN BOXES : ", col_box)
    cell2 = dict()
    for i in range(len(col_box)):
        c = []
        for row in row_box:
            col_obj = fitz.Rect(col_box[i])
            row_obj = fitz.Rect(row)
            cell = fitz.Rect.intersect(col_obj, row_obj)
            c.append(list(cell))
        try:
            cell2[header[i]] = c
        except:
            cell2[f'temp_{i}'] = c
    return cell2, row_box, col_box


def are_on_same_line(bbox1, bbox2, min_distance=0, tolerance=10):
    # Check if the vertical distance between the bottom of bbox1 and the top of bbox2 is within the tolerance
    # and if the overall distance is at least min_distance
    return (
        abs(bbox2[1] - bbox1[1]) <= tolerance
        and abs(bbox2[0] - bbox1[2]) >= min_distance
    )

def group_tokens_by_line(bbox_data, line_tolerance=10):
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




#test case1
'''
row_info = {0: {'approach': 'tatr', 'coords': [5, 91, 1497, 91]}, 1: {'approach': 'tatr', 'coords': [4, 138, 1497, 138]}, 2: {'approach': 'tatr', 'coords': [4, 243, 1498, 243]}, 3: {'approach': 'tatr', 'coords': [5, 304, 1497, 304]}, 4: {'approach': 'tatr', 'coords': [5, 49, 1496, 49]}, 5: {'approach': 'tatr', 'coords': [4, 177, 1498, 177]}}
column_info = {'single_line_elements': [], 'column_seperators': [[[0, 0, 0, 313], ''], ([53, 0, 53, 313], 'solid_lines'), ([164, 0, 164, 313], 'solid_lines'), ([723, 0, 723, 313], 'solid_lines'), ([921, 0, 921, 313], 'solid_lines'), ([1167, 0, 1167, 313], 'solid_lines'), ([164, 0, 164, 313], 'solid_lines'), ([723, 0, 723, 313], 'solid_lines'), ([921, 0, 921, 313], 'solid_lines'), ([1167, 0, 1167, 313], 'solid_lines'), [[1504, 0, 1504, 313], '']]}
header = ['S.No.', 'Description', 'of', 'Goods', 'Quantity', 'Unit', 'Price', 'US', '$', 'Amount', 'USS']
'''
#test case 2

row_info =  {0: {'approach': 'tatr', 'coords': [5, 101, 1437, 101]}, 1: {'approach': 'tatr', 'coords': [4, 164, 1436, 164]}}
column_info = {'single_line_elements': [], 'column_seperators': [[[0, 0, 0, 224], ''], ((263, 0, 263, 224), 'padded_lines'), ((550, 0, 550, 224), 'padded_lines'), ((775, 0, 775, 224), 'padded_lines'), ((943, 0, 943, 224), 'padded_lines'), ((1043, 0, 1043, 224), 'padded_lines'), ((1157, 0, 1157, 224), 'padded_lines'), ((1245, 0, 1245, 224), 'padded_lines'), [[1445, 0, 1445, 224], '']]}
header = ['Description', 'Weight', 'Rate', 'Duty', 'Cess', 'HCess', 'Cvd', 'Amount']

image_path = '/New_Volume/number_theory/table_extraction_api/src/data/crop_images/CI_1_0.png'
import json
wc_path = '/New_Volume/number_theory/table_extraction_api/src/data/ocr_all/CI_1_0.json'
with open(wc_path, 'r') as file:
    wc_cords = json.load(file)
print(wc_cords)
final_word_coordinates = []
modified_ocr_bbox = []

for idx, wc_ in wc_cords.items():
    final_word_coordinates.append({"word":wc_['text'], 'vertices':[wc_['bbox']]})
    modified_ocr_bbox.append(wc_['bbox'])
print(final_word_coordinates)
zones, row_box, col_box  = cell_cord(row_info, column_info, header)
print(zones)


print(modified_ocr_bbox)
row_wise_bbox = group_tokens_by_line(modified_ocr_bbox)
    
print('??????????????????????????????')
print(len(row_wise_bbox))
# exit('>>>>>>>>>>>>>>>>>>')
fina_res = zone_wise_ocr(final_word_coordinates, zones, header, row_wise_bbox)
print('????????????????')
print(fina_res)



# header = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# header = ['larks', '&', 'Nos', '/', 'No.', '&', 'Kind', 'of', 'Pkgs', 'Description', 'of', 'Goods', 'Quantity', 'Rate', 'Amount']
# column_info = {'single_line_elements': [], 'column_seperators': [[[0, 0, 0, 811], ''], ([897, 0, 897, 811], 'solid_lines'), ([1051, 0, 1051, 811], 'solid_lines'), ([1241, 0, 1241, 811], 'solid_lines'), ((170, 0, 170, 810), 'padded_lines'), ((381, 0, 381, 810), 'padded_lines'), ([1241, 0, 1241, 811], 'solid_lines'), [[1461, 0, 1461, 811], '']]}
# row_info = {0: {'approach': 'tatr', 'coords': [5, 160, 1454, 160]}, 1: {'approach': 'tatr', 'coords': [4, 276, 1453, 276]}, 2: {'approach': 'tatr', 'coords': [5, 67, 1454, 67]}}

# zones = cell_cord(row_info, column_info, header)
# print(zone)
# print(">>>>>>>>>>>>>>>>>>>>")
# print(zone_wise_ocr(image_path, zones))
# res = zone_wise_ocr(image_path, zones)
# print('resssssssssssssssssssssssssss', res)