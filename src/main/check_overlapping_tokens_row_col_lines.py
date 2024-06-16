import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from get_ocr_gv_pytesseract_function import get_ocr_vision_api_charConfi

def words_on_line(words: List[Dict], row_line: List[int], col_line: List[int], line_height_tolerance: int = 5) -> List[Dict]:
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

# Example word coordinates and row line


row_lines = [[0, 115, 1490, 115]]
col_lines = [[115, 0, 115, 394]]
image_path = '/New_Volume/Rakesh/miscellaneous_code/src/samples/Covering_Schedule_254_page_5.png'

wc_, all_extracted_text = get_ocr_vision_api_charConfi(image_path)
# Extract tokens and their bounding boxes which lie on the specified line
row_lines,col_lines = words_on_line(wc_, row_lines, col_lines)

# Print the result
print('row_lines',row_lines)
print('col_lines',col_lines)