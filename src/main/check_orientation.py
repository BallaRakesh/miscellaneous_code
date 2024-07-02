"""
* -----------------------------------------------------------------------------------------
*                              NEWGEN SOFTWARE TECHNOLOGIES LIMITED
*
* Group: Number Theory
* Product/Project: Intelligent Trade Finance
* Module: Image Processing
* File Name:
* Author: Tarun Sharma
* Date(DD/MM/YYYY): 12/12/2023
* Description: 
*
* -----------------------------------------------------------------------------------------
*                              CHANGE HISTORY
* -----------------------------------------------------------------------------------------
* Date(DD/MM/YYYY)               Change By              Change Description(Bug No.(If Any))
* -----------------------------------------------------------------------------------------
"""

import cv2
import numpy as np
import cv2
import paddleclas
import os
from copy import deepcopy



THRESHOLD = 0.70

label_dictionary = {
        0:0,
        1:90,
        2:180,
        3:270
    }

def get_model_predictions(model,file_path):
    result = model.predict(input_data=file_path)
    return next(result)

model = paddleclas.PaddleClas(model_name="text_image_orientation")

def image_orientation_fixed_image_generation_old(deskewed_path_dir):
    for file in os.listdir(deskewed_path_dir):
        file_path = os.path.join(deskewed_path_dir, file)
        print(file_path)
        image = cv2.imread(file_path)
        print(image.shape)
        model_prediction_dict = get_model_predictions(model, file_path)[0]
        if model_prediction_dict['scores'][0] < THRESHOLD:
            output_path = os.path.join(deskewed_path_dir,file)
            print('fail img', output_path)
            cv2.imwrite(output_path,image)
        else:
            class_prediction = model_prediction_dict['class_ids'][0]
            if class_prediction != 0:
                rotation_angle = 360 - label_dictionary[class_prediction]
                if image is not None:
                    if rotation_angle == 90:
                        result_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    elif rotation_angle == 180:
                        result_image = cv2.rotate(image, cv2.ROTATE_180)
                    else:
                        result_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)   
                    if result_image is not None:
                        output_path = os.path.join(deskewed_path_dir, file)
                        print('currected', output_path)
                        cv2.imwrite(output_path, result_image)
            else:
                output_path = os.path.join(deskewed_path_dir, file)
                print('currected', output_path)
                cv2.imwrite(output_path, image)   
                       

def image_orientation_fixed_image_generation(file_path):
    print('STARTING image_orientation_fixed_image_generation')
    print('file_path :', file_path)
    image = cv2.imread(file_path)
    print('image size ====> image_orientation', image.shape)
    model_prediction_dict = get_model_predictions(model, file_path)[0]
    print('file_path: ', file_path, 'scores:', model_prediction_dict)
    if model_prediction_dict['scores'][0] < THRESHOLD:
        return image

    else:
        class_prediction = model_prediction_dict['class_ids'][0]
        if class_prediction != 0:
            rotation_angle = 360 - label_dictionary[class_prediction]
            if image is not None:
                if rotation_angle == 90:
                    result_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif rotation_angle == 180:
                    result_image = cv2.rotate(image, cv2.ROTATE_180)
                else:
                    result_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)   
                if result_image is not None:
                    return result_image
        else:
            return image

def deskew(im, orig_image, max_skew=10):
    height, width = im.shape

    # Create a grayscale image and denoise it
    # im_gs = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gs = cv2.fastNlMeansDenoising(im, h=3)

    # Create an inverted B&W copy using Otsu (automatic) thresholding
    im_bw = cv2.threshold(im_gs, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Detect lines in this image. Parameters here mostly arrived at by trial and error.
    lines = cv2.HoughLinesP(
        im_bw, 1, np.pi / 180, 200, minLineLength=width / 12, maxLineGap=width / 150
    )
    print(lines)
    if lines is None:
        print("case1: no lines jun 28")
        return orig_image, None
    # Collect the angles of these lines (in radians)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angles.append(np.arctan2(y2 - y1, x2 - x1))

    print( np.sum([abs(angle) > np.pi / 4 for angle in angles]))
    print(len(angles))
    # If the majority of our lines are vertical, this is probably a landscape image
    landscape = np.sum([abs(angle) > np.pi / 4 for angle in angles]) > len(angles) / 1.5
    print(landscape)
    # print(f"angles: {angles}")

    # Filter the angles to remove outliers based on max_skew
    if landscape:
        print('yesssssssssssssssssssssss')
        angles = [
            angle
            for angle in angles
            if np.deg2rad(90 - max_skew) < abs(angle) < np.deg2rad(90 + max_skew)
        ]
    else:
        angles = [angle for angle in angles if abs(angle) < np.deg2rad(max_skew)]

    if len(angles) < 5:
        # Insufficient data to deskew
        return orig_image, None

    # Average the angles to a degree offset
    angle_deg = np.rad2deg(np.median(angles))
    new_width, new_height = width, height
    # If this is landscape image, rotate the entire canvas appropriately
    if landscape:
        if angle_deg < 0:
            orig_image = cv2.rotate(orig_image, cv2.ROTATE_90_CLOCKWISE)
            angle_deg += 90
        elif angle_deg > 0:
            orig_image = cv2.rotate(orig_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            angle_deg -= 90
        new_width, new_height = height, width

    # Rotate the image by the residual offset
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1)
    orig_image = cv2.warpAffine(orig_image, M, (new_width, new_height), borderMode=cv2.BORDER_REPLICATE)
    return orig_image, angle_deg


def main(deskewed_path_dir):
    # process an image 
    for img_name in os.listdir(deskewed_path_dir):
        img = os.path.join(deskewed_path_dir, img_name)
        image = cv2.imread(img)
        print('before', image.shape)
        orig_image = deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image, angle = deskew(image, orig_image)
        print(img, '>>>>>>>>>', angle)
        print('after', image.shape)
        cv2.imwrite(os.path.join(deskewed_path_dir, img_name), image)
        res_image = image_orientation_fixed_image_generation(os.path.join(deskewed_path_dir, img_name))
        cv2.imwrite(os.path.join(deskewed_path_dir, img_name), res_image)
    
        
deskewed_path_dir = '/home/ntlpt19/TF_testing_EXT/code/miscellaneous_code/src/main/pdf_exp1'
'''
if not os.path.exists(deskewed_path_dir):
    os.makedirs(deskewed_path_dir)

image_orientation_fixed_dir = '/home/ntlpt19/TF_testing_EXT/code/miscellaneous_code/src/main/image_orientation_fixed_dir'
if not os.path.exists(image_orientation_fixed_dir):
    os.makedirs(image_orientation_fixed_dir)

fail_image_directory = '/home/ntlpt19/TF_testing_EXT/code/miscellaneous_code/src/main/fail_image_directory'
if not os.path.exists(fail_image_directory):
    os.makedirs(fail_image_directory)

org_img_dir = '/home/ntlpt19/TF_testing_EXT/code/miscellaneous_code/src/main/NewImages'
'''


main(deskewed_path_dir)
# image_orientation_fixed_image_generation(deskewed_path_dir)
