import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple



def words_on_line(words: List[Dict], row_line: List[int], line_height_tolerance: int = 5) -> List[Dict]:
    tokens_on_line = []
    for word in words:
        # Check if the word's bounding box intersects with the row line (considering some tolerance for height)
        if row_line[1] - line_height_tolerance <= word['y2'] <= row_line[1] + line_height_tolerance:
            tokens_on_line.append(word)
    return tokens_on_line

# Example word coordinates and row line
words = [
    {'word': 'MARKS', 'left': 37, 'top': 26, 'width': 78, 'height': 16, 'confidence': 0.9861640334129333, 'x1': 37, 'y1': 26, 'x2': 115, 'y2': 42},
    {'word': '&', 'left': 121, 'top': 26, 'width': 16, 'height': 16, 'confidence': 0.9540494680404663, 'x1': 121, 'y1': 26, 'x2': 137, 'y2': 42},
    {'word': 'NOS', 'left': 146, 'top': 26, 'width': 47, 'height': 16, 'confidence': 0.9874510765075684, 'x1': 146, 'y1': 26, 'x2': 193, 'y2': 42},
    # Additional tokens from the image
    {'word': 'COLD', 'left': 37, 'top': 107, 'width': 45, 'height': 16, 'confidence': 0.9800000190734863, 'x1': 37, 'y1': 107, 'x2': 82, 'y2': 123},
    {'word': 'DOUGH', 'left': 87, 'top': 107, 'width': 55, 'height': 16, 'confidence': 0.9800000190734863, 'x1': 87, 'y1': 107, 'x2': 142, 'y2': 123},
    {'word': 'PREMIX', 'left': 147, 'top': 107, 'width': 65, 'height': 16, 'confidence': 0.9800000190734863, 'x1': 147, 'y1': 107, 'x2': 212, 'y2': 123},
    {'word': 'FDPZ3004', 'left': 217, 'top': 107, 'width': 80, 'height': 16, 'confidence': 0.9800000190734863, 'x1': 217, 'y1': 107, 'x2': 297, 'y2': 123},
]

row_lines = [[0, 115, 1490, 115]]

# Extract tokens and their bounding boxes which lie on the specified line
tokens_on_row_line = words_on_line(words, row_lines[0])

# Print the result
print(tokens_on_row_line)
