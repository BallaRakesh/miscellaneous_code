import cv2
import numpy as np


abc = '* Paella Valenciana,\n* Pulpo a la Gallega,\n* Croquetas de Jamón,\n* Tortilla Española,\n* Patatas Bravas,\n* Ensalada de Tomate con Jamón,\n* Gazpacho Andaluz,\n* Sangría,\n* Cerveza,\n* Vino Blanco'
# Splitting the provided text by spaces to count the tokens
text = "+ discrepancy fee of usd 100 / eur 100 / jpy 10000 / gbp 100 per set of discre\n+ third party documents except draft and commercial invoice are acceptabli\n"
tokens = text.split()
token_length = len(tokens)
print(token_length)
exit('???????????')

menu_items = abc.strip().split(',')
print(menu_items)
for itm in menu_items:
    print(itm)
    
exit('???????????')
def tokenize(content):
    return content.split()
content = 'nylon\n adv'
tokens = tokenize(content)

print(tokens)
exit('>>>>>>>')



class TableLinesRemover:

    def __init__(self, image):
        self.image = image
        self.vertical_bboxes = []
        self.horizontal_bboxes = []

    def execute(self):
        self.grayscale_image()
        self.store_process_image("0_grayscaled.jpg", self.grey)
        self.threshold_image()
        self.store_process_image("1_thresholded.jpg", self.thresholded_image)
        self.invert_image()
        self.store_process_image("2_inverted.jpg", self.inverted_image)
        self.erode_vertical_lines()
        self.store_process_image("3_erode_vertical_lines.jpg", self.vertical_lines_eroded_image)
        self.erode_horizontal_lines()
        self.store_process_image("4_erode_horizontal_lines.jpg", self.horizontal_lines_eroded_image)
        self.combine_eroded_images()
        self.store_process_image("5_combined_eroded_images.jpg", self.combined_image)
        self.dilate_combined_image_to_make_lines_thicker()
        self.store_process_image("6_dilated_combined_image.jpg", self.combined_image_dilated)
        self.subtract_combined_and_dilated_image_from_original_image()
        self.store_process_image("7_image_without_lines.jpg", self.image_without_lines)
        self.remove_noise_with_erode_and_dilate()
        self.store_process_image("8_image_without_lines_noise_removed.jpg", self.image_without_lines_noise_removed)
        self.draw_bboxes()
        return self.image_without_lines_noise_removed

    def grayscale_image(self):
        self.grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def threshold_image(self):
        self.thresholded_image = cv2.threshold(self.grey, 127, 255, cv2.THRESH_BINARY)[1]

    def invert_image(self):
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)

    def erode_vertical_lines(self):
        ver_kernel = np.ones((25, 1), np.uint8)
        self.vertical_lines_eroded_image = cv2.erode(self.inverted_image, ver_kernel, iterations=2)
        self.vertical_lines_eroded_image = cv2.dilate(self.vertical_lines_eroded_image, ver_kernel, iterations=2)
        self.vertical_bboxes = self.get_bounding_boxes(self.vertical_lines_eroded_image)

    def erode_horizontal_lines(self):
        hor_kernel = np.ones((1, 25), np.uint8)
        self.horizontal_lines_eroded_image = cv2.erode(self.inverted_image, hor_kernel, iterations=2)
        self.horizontal_lines_eroded_image = cv2.dilate(self.horizontal_lines_eroded_image, hor_kernel, iterations=2)
        self.horizontal_bboxes = self.get_bounding_boxes(self.horizontal_lines_eroded_image)

    def get_bounding_boxes(self, eroded_image):
        contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(contour) for contour in contours]
        print(bboxes)
        return bboxes

    def combine_eroded_images(self):
        self.combined_image = cv2.add(self.vertical_lines_eroded_image, self.horizontal_lines_eroded_image)

    def dilate_combined_image_to_make_lines_thicker(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.combined_image_dilated = cv2.dilate(self.combined_image, kernel, iterations=5)

    def subtract_combined_and_dilated_image_from_original_image(self):
        self.image_without_lines = cv2.subtract(self.inverted_image, self.combined_image_dilated)

    def remove_noise_with_erode_and_dilate(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.image_without_lines_noise_removed = cv2.erode(self.image_without_lines, kernel, iterations=1)
        self.image_without_lines_noise_removed = cv2.dilate(self.image_without_lines_noise_removed, kernel, iterations=1)

    def draw_bboxes(self):
        # Convert back to BGR for drawing colored rectangles
        image_with_bboxes = cv2.cvtColor(self.grey, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in self.vertical_bboxes:
            cv2.rectangle(image_with_bboxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (x, y, w, h) in self.horizontal_bboxes:
            cv2.rectangle(image_with_bboxes, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite("/New_Volume/Rakesh/miscellaneous_code/src/sample_outputs/9_image_with_bboxes.jpg", image_with_bboxes)

    def store_process_image(self, file_name, image):
        path = "/New_Volume/Rakesh/miscellaneous_code/src/sample_outputs" + file_name
        cv2.imwrite(path, image)

# Example usage:
image_path = "/New_Volume/Rakesh/miscellaneous_code/src/samples/Covering_Schedule_254_page_5.png"
image = cv2.imread(image_path)
remover = TableLinesRemover(image)
processed_image = remover.execute()
print("Vertical lines detected:", len(remover.vertical_bboxes))
print("Horizontal lines detected:", len(remover.horizontal_bboxes))
