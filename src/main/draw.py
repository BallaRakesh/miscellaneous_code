import cv2


def draw(image_path, bbox):
    image = cv2.imread(image_path)

    # Define the color and thickness of the line
    color = (0, 255, 0)  # Green color in BGR
    thickness = 2  # Thickness of 2 pixels

    # Draw the line on the image
    cv2.line(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

    # Save the modified image
    output_path = '/New_Volume/Rakesh/miscellaneous_code/src/sample_outputs/output_with_line.png'  # Path to save the output image
    cv2.imwrite(output_path, image)


# Define the bounding box coordinates
bbox = [0, 115, 1490, 115]


# Load the image
image_path = '/New_Volume/Rakesh/miscellaneous_code/src/samples/Covering_Schedule_254_page_5.png'  # Update the path to your image
image = cv2.imread(image_path)
bbox = [115, 0, 115, image.shape[0]] 
print(bbox)
draw(image_path, bbox)