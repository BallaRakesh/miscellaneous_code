import fitz
import torchvision.transforms as transforms
import torch
from google.cloud import vision
from base64 import b64encode
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import random
import warnings
from PIL import ImageSequence
import pytesseract
import gc
import logging
from datetime import datetime
from scipy import stats
from configparser import ConfigParser
# from training.lmv2code.src.main.extraction.config.prod_mapping import product_code_map, document_code_map

configur = ConfigParser()
configur.read('./traini_valid_utility.ini')
gv_key = configur['OCR']['gv_key']

# # product config
# product_config = ConfigParser()
# product_config.read("training/lmv2code/src/main/extraction/config/config.ini")

# prod_code = product_code_map[product_config["Product"]["code"]]
# doc_code = document_code_map[product_config["Product"]["document_code"]]

# # data folder path
# product_wise_folder = ConfigParser()
# product_wise_folder.read("training/lmv2code/src/main/extraction/config/prod.ini")
# folder_path = product_wise_folder[prod_code][doc_code]

# print("==================Trade Finance Solutions===================")
# print("Product Code: {product_code}")
# print("Documenry Code: {doc_code}")
# print(f"folder_path: {folder_path}")

# ocr_path = os.path.join(folder_path, "OCR")
# path = os.path.join(folder_path, "Images")
# label_path = os.path.join(folder_path, "classes.txt")

os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.cluster import DBSCAN
from functools import cmp_to_key
import json
import traceback
from nltk import ngrams
import re
from scipy.special import softmax
from fuzzywuzzy import fuzz
import psutil

transform2 = transforms.ToPILImage()
transform = transforms.ToTensor()

zoom = 300 / 72
mat = fitz.Matrix(zoom, zoom)

vertical_merge_labels = [
		'shipper_country',
		"shipper_address",
		'carriage_condition',
		'consignee_country',
		'carrier_address', 'agent_address',
		'goods_description', "description_of_goods",
		'amount_in_words',
		"drawee_bank_address",
		"drawer_bank_address",
		"drawee_address", "nostro_bank_address", "drawer_address",
		"consignee_address", 'consignee_addres', 'shipper_address',
		'notify_party_address', "consignor_address", "coo_issuer_address",
		'nostro_bank_address', 'consignor_address', 'address_of_assured',
		'drawee_address', 'insurance_issuer_address', 'remitter_address',
		'beneficiary_address', 'coo_issuer_address', 'notify_party_address',
		'drawer_bank_address', 'consignee_address', 'drawer_address',
		'drawee_bank_address', 'insurance_issuer_address_bottom',
		'drawer_bank_bottom_address', 'shipper_address','drawee_bank_name',
		'claim_payable_by_address', "carrier_country", 'agent_country',"declaration","dimension",
		"carrier_name","agent_name","shipper_name","consignee_name","notify_party_name","carrier_name",\
		"agent_name","notify_party_name","consignee_name","shipper_name","drawer_bank_name","drawee_bank_name",\
		"freight_collect_or_prepaid","signed_by_carrier", "drawer_name",\
		"tenore_details", "signed_By_agent", "signed_by_agent", "subject_matter_insured",\
		"diclaration_by", "payment_terms_terms_of_delivery_payment"
			]

# will not use DBScan with these tokens
single_text_labels = [
	"bill_exchange_no",
	"bill_exchange_date",
	"boe_currency",
	"boe_amount",
	"country_of_origin",
	"invoice_no",
	"invoice_date",
	"invoice_currency",
	"invoice_amount",
	"tenor_type",
	"usance_tenor",
	"tenor_indicator",
	"indicator_type",
	"indicator_date",
	"invoice_due_date",
	"original_or_copy",
	"original_number",
	"lc_ref_no",
	"lc_date",
	"issue_place",
	"transaction_date",
	"awb_bill_no",
	"master_awb_bill_no",
	"house_awb_bill_no",
	"awb_bill_issue_date",
	"flight_no",
	"flight_date",
	"shipper_country",
	"consignee_country",
	"notify_party_country",
	"place_of_receipt",
	"airport_of_departure",
	"airport_of_destination",
	"final_destination",
	"declared_value_of_carriage",
	"amount_insurance",
	"gross_quantity",
	"gross_weight",
	"net_weight",
	"good_marks",
	"invoice_number",
	"invoice_date",
	"lc_no",
	"lc_date",
	"freight_collected_at",
	"awb_original_number",
	"awb_original_or_copy",
	"flight_details",
	"declared_value_of_custom",
	"transaction_date",
	"awb_bill_no",
	"master_awb_bill_no",
	"house_awb_bill_no",
	"awb_bill_issue_date",
	"flight_no",
	"flight_date",
	"notify_party_country",
	"place_of_receipt",
	"airport_of_departure",
	"airport_of_destination",
	"final_destination",
	"declared_value_of_carriage",
	"amount_insurance",
	"gross_quantity",
	"gross_weight",
	"net_weight",
	"good_marks",
	"invoice_number",
	"invoice_date",
	"lc_no",
	"lc_date",
	"freight_collect_or_prepaid",
	"freight_collected_at",
	"awb_original_number",
	"awb_original_or_copy",
	"flight_details",
	"declared_value_of_custom",
	"at_place",	
	"port_of_loading",
	"container_number",
	"country_of_final_destination"
]

master_keys = ['goods_description']


# "****************************************************************************************************************************************"
def set_basic_config_for_logging(filename: str = None, folder_path=None):
	"""
    Set the basic config for logging python program.   
    :return: None   
    """
	# Create and configure logger
	log_file_path = os.path.join(folder_path, f"{filename}.log")
	logging.basicConfig(filename=log_file_path, format='%(asctime)s %(message)s',
	                    filemode='w')


def get_logger_object_and_setting_the_loglevel():
	"""    get the logger object and set the loglevel for the logger object
    :return: Logger Object    
    """
	# Creating an object
	logger_object = logging.getLogger()
	# Setting the threshold of logger to DEBUG
	logger_object.setLevel(logging.DEBUG)
	return logger_object



# process_memory = psutil.Process()


def area(coordinates):
	l = coordinates[2] - coordinates[0]
	h = coordinates[3] - coordinates[1]
	return l * h


def get_iou_new(bb1, bb2):
	"""
	Calculate the Intersection over Union (IoU) of two bounding boxes.

	Parameters
	----------
	bb1 : dict
		Keys: {0, '2', 1, '3'}
		The (x1, 1) position is at the top left corner,
		the (2, 3) position is at the bottom right corner
	bb2 : dict
		Keys: {0, '2', 1, '3'}
		The (x, y) position is at the top left corner,
		the (2, 3) position is at the bottom right corner

	Returns
	-------
	float
		in [0, 1]
	"""
	assert bb1[0] < bb1[2]
	assert bb1[1] < bb1[3]
	assert bb2[0] < bb2[2]
	assert bb2[1] < bb2[3]

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1[0], bb2[0])
	y_top = max(bb1[1], bb2[1])
	x_right = min(bb1[2], bb2[2])
	y_bottom = min(bb1[3], bb2[3])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)

	# compute the area of both AABBs
	bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
	bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	assert iou >= 0.0
	assert iou <= 1.0
	return iou


def minimum_distance(bb1, bb2):
	# bb1 points
	min_distance = 9999999999
	p_11 = np.array((bb1[0], bb1[1]))
	#(x2, (y1 + y2) / 2)
	# p_12 = np.array((bb1[0], bb1[3]))
	r_m_point = np.array((bb1[1], (bb1[3]+bb1[1])/2)) #???? r_m_point = np.array((bb1[2], (bb1[1] + bb1[3]) / 2))

	p_13 = np.array((bb1[2], bb1[3]))
	p_14 = np.array((bb1[2], bb1[1]))
	# all_points_bb1 = [p_11, p_12, p_13, p_14]
	all_points_bb1 = [p_11, p_13, p_14]#, r_m_point]
	# bb2 points
	p_21 = np.array((bb2[0], bb2[1]))
	p_22 = np.array((bb2[0], bb2[3]))
	p_23 = np.array((bb2[2], bb2[3]))
	# p_24 = np.array((bb2[2], bb2[1]))
	# all_points_bb2 = [p_21, p_22, p_23, p_24]
	all_points_bb2 = [p_21, p_22, p_23]
	for point1 in all_points_bb1:
		for point2 in all_points_bb2:
			dist = abs(np.linalg.norm(point1 - point2))
			if dist < min_distance:
				min_distance = dist
	return min_distance


def check_vertical_distribution(bb1, bb2):
	y1 = bb1[1]
	y2 = bb2[1]
	return abs(y1 - y2)


def model_output_sum(key, box, model_output):
	all_values = model_output[key]
	all_values = sorted(all_values, key=cmp_to_key(contour_sort))
	all_text = ""
	for value in all_values:
		try:
			iou = get_iou_new(value[1], box)
		except:
			continue
		if iou > 0:
			if all_text == "":
				all_text = value[0]
			else:
				all_text = all_text + " " + value[0]
	return all_text


# def merge_surrounding(data, model_output):
# 	new = data.copy()
# 	for key in list(data.keys()):
# 		print(key)
# 		bboxes = [x[1] for x in data[key]]
# 		eps = 100
# 		all_values = data[key]
# 		print(all_values)
# 		length = len(all_values)
# 		if length > 1:
# 			i = 0
# 			while i in range(length - 1):
# 				print(i)
# 				bb1 = all_values[i][1]
# 				bb2 = all_values[i + 1][1]
# 				confs = [all_values[i][2], all_values[i + 1][2]]
# 				# ocr_confs = [all_values[i][3],all_values[i+1][3]]
# 				min_dist = minimum_distance(bb1, bb2)
# 				try:
# 					IOU = get_iou_new(bb1, bb2)
# 				except:
# 					i = i + 1
# 					continue
# 				if min_dist <= eps or IOU > 0:
# 					print("merging: " + all_values[i][0] + " and " + all_values[i + 1][0])
# 					x_left = min(bb1[0], bb2[0])
# 					y_top = min(bb1[1], bb2[1])
# 					x_right = max(bb1[2], bb2[2])
# 					y_bottom = max(bb1[3], bb2[3])
# 					box = [x_left, y_top, x_right, y_bottom]
# 					text = model_output_sum(key, box, model_output)
# 					print("merged text is ", text)
# 					avg_confs = (confs[0] * area(bb1) + confs[1] * area(bb2)) / (area(bb1) + area(bb2))
# 					new_value = [text, box, avg_confs]
# 					print(new_value)
# 					all_values.remove(all_values[i])
# 					all_values.remove(all_values[i])
# 					all_values.insert(i, new_value)
# 					print(all_values)
# 					length = len(all_values)
# 					if length == 1:
# 						print("will break")
# 						break
# 				else:
# 					print("distance is very high")
# 					i = i + 1
# 		else:
# 			print("will continue")
# 			continue
def get_iou_horizontal(bb1, bb2):
	# Calculate the Intersection over Union (IoU) of two bounding boxes (horizontal intersection)
	x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
	x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

	x_left = max(x1_bb1, x1_bb2)
	y_top = max(y1_bb1, y1_bb2)
	x_right = min(x2_bb1, x2_bb2)
	y_bottom = min(y2_bb1, y2_bb2)

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
	bb1_area = (x2_bb1 - x1_bb1) * (y2_bb1 - y1_bb1)
	bb2_area = (x2_bb2 - x1_bb2) * (y2_bb2 - y1_bb2)
	union_area = bb1_area + bb2_area - intersection_area

	return intersection_area / union_area


def get_iou_vertical(bb1, bb2):
	# Calculate the Intersection over Union (IoU) of two bounding boxes (vertical intersection)
	x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
	x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

	x_left = max(x1_bb1, x1_bb2)
	y_top = max(y1_bb1, y1_bb2)
	x_right = min(x2_bb1, x2_bb2)
	y_bottom = min(y2_bb1, y2_bb2)

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
	bb1_area = (x2_bb1 - x1_bb1) * (y2_bb1 - y1_bb1)
	bb2_area = (x2_bb2 - x1_bb2) * (y2_bb2 - y1_bb2)
	union_area = bb1_area + bb2_area - intersection_area

	return intersection_area / union_area


def get_intersection_percentage(bb1, bb2):
	# Calculate the percentage of vertical intersection between two bounding boxes
	x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
	x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

	x_left = max(x1_bb1, x1_bb2)
	y_top = max(y1_bb1, y1_bb2)
	x_right = min(x2_bb1, x2_bb2)
	y_bottom = min(y2_bb1, y2_bb2)

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	intersection_area = max(0, y_bottom - y_top)
	bb1_area = y2_bb1 - y1_bb1
	bb2_area = y2_bb2 - y1_bb2

	return intersection_area / min(bb1_area, bb2_area)


def minimum_distance_vertical(bb1, bb2):
	# Calculate the minimum vertical distance between two bounding boxes
	x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
	x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2
	'''
	top to middle
	X-coordinate: x2
	Y-coordinate: ((y1_bb1 + y2_bb1) / 2 + y1_bb1) / 2 
	bottom to middle
	X-coordinate: x2
	Y-coordinate: ((y1_bb1 + y2_bb1) / 2 + y2_bb1) / 2
	'''
	min_distance_y = min(abs(y1_bb2 - y2_bb1), abs(y1_bb1 - y2_bb2), abs(y1_bb1 - y1_bb2), abs(y2_bb1 - y2_bb2), abs(y1_bb2 - (((y1_bb1 + y2_bb1) / 2 + y1_bb1) / 2)), abs(y2_bb2 - (((y1_bb1 + y2_bb1) / 2 + y2_bb1) / 2)))

	return min_distance_y

def check_vertical_indetween(bb1, bb2):
	in_between_flag = False
	# Calculate the minimum vertical distance between two bounding boxes
	x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
	x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2
	
	'''
	X-coordinate: x2
	Y-coordinate: ((y1 + y2) / 2 + y1) / 2
	'''
 
	if (y1_bb2 < y1_bb1 < y2_bb2) or (y1_bb2 < y2_bb1 > y2_bb2) or (y1_bb1 < y1_bb2 < y2_bb1) or (y1_bb1 < y2_bb2 < y2_bb1):
		in_between_flag = True
	return in_between_flag


def vertical_horizontal_values(word_bbox_list):
	# Initialize two lists for separate groups
	group1 = []  # Words with height > width
	group2 = []  # Words with height <= width
	# Iterate through the word_bbox_list and separate words based on their bbox dimensions
	for word, bbox in word_bbox_list:
		x1, y1, x2, y2 = bbox
		width = x2 - x1
		height = y2 - y1
		if height > width:
			group1.append([word, bbox])
		else:
			group2.append([word, bbox])

	return [group1, group2]


def special_chr_check(bb_token, flag):
	if ',' in bb_token:
		flag = False
	if '-' in bb_token:
		flag = False
	return flag

def calculate_orientation(word_bbox):
	vertical_alignment_bbox = False
	# Calculate centroids of characters
	centroids = [(bbox[0] + bbox[2]) / 2 for bbox in word_bbox]

	# Calculate angle between horizontal axis and line connecting first and last centroids
	delta_y = centroids[-1] - centroids[0]
	delta_x = word_bbox[-1][2] - word_bbox[0][0]
	angle = np.arctan2(delta_y, delta_x)

	# Convert angle from radians to degrees
	angle_degrees = np.degrees(angle)
	print(angle_degrees)
	if angle_degrees < -50:
		vertical_alignment_bbox = True
	return vertical_alignment_bbox

	
def after_skipping(all_values, inx_ele_skip): 
    # Add addition_val to the elements specified by inx_ele_skip
    # for index in range(len(inx_ele_skip)):
    #     inx_ele_skip[index] += addition_val 
    # Delete elements specified by inx_ele_skip 
    final_all_values = {}
    for idx, value in all_values.items():
        if idx not in inx_ele_skip:
            final_all_values[idx] = value
    # all_values = [value for idx, value in enumerate(all_values) if idx not in inx_ele_skip]
    return final_all_values

import copy
def merge_by_skipping_running(model_output, w, h, key, all_values):
	print("entered final validation", all_values)
	'''
	[["500 , telangana , india", [161, 172, 308, 180], 67.32535079575597], =>1
	["perak .", [290, 222, 329, 232], 84.68], =>2
	["dusun kabupaten i pauh utara deli , kecamatan sumatera serdang 20374 , hamparan indonesia -", [65, 223, 285, 270], 91.24132404181185]],
	                                          => 3 (need to merge 2 and 3)
	'''
	all_values = sorted(all_values, key=lambda bbox: bbox[1][0])#, reverse=True)
	all_values = {idx: item for idx, item in enumerate(all_values)}
	print("before starting", all_values)
	no_of_ele_skip = 1
	inx_ele_skip = [0]
	current_itteration = 0
	initial_check = True
	inx_ele_skip_flag = True
	len_idx_ele_skip = 1
	while len(all_values)>2:# and no_of_ele_skip-1 < len(all_values)-2:
		print('Before', all_values)
		if inx_ele_skip_flag:
			copied_inx_ele_skip = copy.deepcopy(inx_ele_skip)
			inx_ele_skip_flag = False
		if initial_check:
			actual_length = len(all_values)
			initial_check = False
		deep_copied_all_values = copy.deepcopy(all_values)
		all_values = after_skipping(all_values, inx_ele_skip)
		print('after_skipping these indexes', inx_ele_skip)
		print(all_values)
		length = len(all_values)
		print('>>>>>>>>>>>#################>>>>>>>>>>>>>>>>>>>>>>>>')
		print('>>>>>>>>>>>#################>>>>>>>>>>>>>>>>>>>>>>>>')
		# all_values = data[key]
		# for sort_idx in range(0,4):
		bboxes = [x[1] for x in all_values.values()]
		vertical_alignment_bbox = calculate_orientation(bboxes)
		vertical_alignment_bbox = False
		print(key, ">>>>>>> vertical_alignment_bbox >>>>>>>>>", vertical_alignment_bbox)
		# eps_horizontal = 100  # Threshold for horizontal merging
		# eps_vertical = 50  # Threshold for vertical merging
		# eps_vertical2 = 100  # Threshold for vertical merging
		eps_horizontal = round(w*5.5/100)#100
		eps_vertical = round(h*2.5/100) #100
		eps_vertical2 = 100  # Threshold for vertical merging
		if length > 1:
			i = 0
			while i in range(length - 1):
				print(i)
				bb1 = list(all_values.values())[i][1]
				bb2 = list(all_values.values())[i + 1][1]
				bb1_token = list(all_values.values())[i][0]
				bb2_token = list(all_values.values())[i + 1][0]
				confs = [list(all_values.values())[i][2], list(all_values.values())[i + 1][2]]
				min_dist_horizontal = minimum_distance(bb1, bb2)
				min_dist_vertical = minimum_distance_vertical(bb1, bb2)
				vertical_flag = check_vertical_indetween(bb1, bb2)
				try:
					IOU_horizontal = get_iou_horizontal(bb1, bb2)
					IOU_vertical = get_iou_vertical(bb1, bb2)
					inter_percentage = get_intersection_percentage(bb1, bb2)
				except:
					i = i + 1
					continue
				bb1_x1, bb1_y1, bb1_x2, bb1_y2 = bb1
				bb2_x1, bb2_y1, bb2_x2, bb2_y2 = bb2
				bb1_width = bb1_x2 - bb1_x1
				bb1_height = bb1_y2 - bb1_y1
				bb2_width = bb2_x2 - bb2_x1
				bb2_height = bb2_y2 - bb2_y1
				# if len(bb1_token)<3:
				print('beore', bb2_width, bb2_height)
				print('bb2_token length', len(bb2_token))
				flag1 = True
				flag2 = True
				flag1 = special_chr_check(bb1_token, flag1)
				flag2 = special_chr_check(bb2_token, flag2)

				if len(bb1_token) == 1 or (len(bb1_token) < 3 and flag1 == False):
					temp = bb1_width
					bb1_width = bb1_height
					bb1_height = temp
				if len(bb2_token) == 1 or (len(bb2_token) < 3 and flag2 == False):
					temp = bb2_width
					bb2_width = bb2_height
					bb2_height = temp
				print('flag2', flag2)
				print('bb2_token', bb2_token)
				print('bb1_width', bb1_width, 'bb1_height', bb1_height)
				print('bb2_width', bb2_width, 'bb2_height', bb2_height)

				# if (bb1_height >= bb1_width and bb2_height >= bb2_width) or (
				# 		bb1_height <= bb1_width and bb2_height <= bb2_width):

				print('entered into first if')
				print(">>>>>>>>>>RRRRRRRRAAAAAAAAAKKKKKKKEEE", min_dist_horizontal)
				if key in master_keys or vertical_alignment_bbox:
					merge_flag = (min_dist_horizontal <= eps_horizontal or IOU_horizontal > 0 or inter_percentage > 0) or (
							min_dist_vertical <= eps_vertical2 or IOU_vertical > 0 or inter_percentage)
				else:
					merge_flag = (min_dist_horizontal <= eps_horizontal or IOU_horizontal > 0 or inter_percentage > 0) and (
							min_dist_vertical <= eps_vertical or IOU_vertical > 0 or inter_percentage)# or vertical_flag) 
				print('min_dist_vertical', min_dist_vertical)
				print("IOU_horizontal > 0 or inter_percentage", IOU_horizontal, inter_percentage)
				if merge_flag:
					print('entered into second if')
					print("merging: " + list(all_values.values())[i][0] + " and " + list(all_values.values())[i + 1][0])
					x_left = min(bb1[0], bb2[0])
					y_top = min(bb1[1], bb2[1])
					x_right = max(bb1[2], bb2[2])
					y_bottom = max(bb1[3], bb2[3])
					box = [x_left, y_top, x_right, y_bottom]
					text = model_output_sum(key, box, model_output)
					print("merged text is ", text)
					avg_confs = (confs[0] * area(bb1) + confs[1] * area(bb2)) / (area(bb1) + area(bb2))
					new_value = [text, box, avg_confs]
					print(new_value)
					print(all_values)
					print(i)
					deletionkey = next((k for k, v in all_values.items() if v == list(all_values.values())[i + 1]), None)
					updation_key = next((k for k, v in all_values.items() if v == list(all_values.values())[i]), None)
					print('deletion_key >>>>>>>>>>',deletionkey)
					print('updation_key >>>>>>>',updation_key)
					del all_values[deletionkey]        # we are deleting the right value/ after value and updating the left/previous value
					all_values[updation_key] = new_value
     
					# all_values.remove(all_values[i])
					# # all_values.remove(all_values[i]) ############??????
					# all_values.insert(i, new_value)
     
					print(all_values)
					length = len(all_values)
					if length == 1:
						print("will break")
						break
					# else:
					# 	print("distance is very high")
					# 	i = i + 1
				else:
					i = i + 1
     
			print('after', inx_ele_skip, '$$$$$$$', all_values)
			######################################################
			######################################################
			#adding the skipped elements
			for itter_idx in inx_ele_skip:
				all_values[itter_idx] = deep_copied_all_values[itter_idx]
			print('After', all_values)
   
			#### sorting the values
			all_values = dict(sorted(all_values.items()))
   
			# updating the index with 1
			# for idx_, itter_idx_up in enumerate(inx_ele_skip):
			if max(all_values.keys()) not in inx_ele_skip:
				idx_ = 0
				increment_value = 1
				while idx_ < len(inx_ele_skip):
					req_val = inx_ele_skip[idx_] + increment_value
					if req_val in all_values:
						inx_ele_skip[idx_] = req_val
						idx_ = idx_ + 1
						increment_value = 1
					else:
						increment_value += 1
				######### break
    
    
			if max(all_values.keys()) in inx_ele_skip:
				initial_check = True
				if actual_length > len(all_values):
					inx_ele_skip = copied_inx_ele_skip
					inx_ele_skip_flag = True
				else:
					len_idx_ele_skip +=1
					inx_ele_skip = sorted(all_values.keys())[:len_idx_ele_skip]
     
			if len_idx_ele_skip+1 == len(all_values):
				print('i need to break here')
				print(inx_ele_skip)
				print(len_idx_ele_skip+2)
				print(len(all_values))
				break
		else:
			break
		

		print(all_values)
	return [item for item in all_values.values()]

def merge_surrounding(data, model_output, w, h):
	# print('start>>>>>>>>>>>>')
	# print(data)
	# print(model_output)
	# print(w, h)

	new = data.copy()
	print('entered into merging_surroundings ++++++++++++++++++++++++++++++++++++++')
	for key in list(data.keys()):
		print(key)
		if key in vertical_merge_labels or key in master_keys:
			# for validate in range(0,2):
			all_values = data[key]
   
			for sort_idx in range(0,4): #uncomment this later ####################################################
				########################################################################################################
				########################################################################################################
				# sort_idx = 0
				print('STARTED FOR ####### ITTERATION',sort_idx)
				bboxes = [x[1] for x in all_values]
				vertical_alignment_bbox = calculate_orientation(bboxes)
				print('vertical_alignment_bbox -->', vertical_alignment_bbox)
				vertical_alignment_bbox = False
				all_values = sorted(all_values, key=lambda bbox: bbox[1][sort_idx])
				print(f"sorting the values on the base of {sort_idx} ##### >>>>> {all_values}")
				'''
				[["house - 400 021 ,", [1291, 501, 1366, 532], 71.82], ["245 , martamo house cama road , mumbai", [1068, 516, 1300, 532], 81.86662341004609]]
				'''
				if len(all_values) > 1:
					# all_values = vertical_horizontal_values(new_all_values)
					# if key=='drawee_address':
					#     print(all_values)
					#     exit("PPPPPPPPPPPP")
					bboxes = [x[1] for x in data[key]]
					# eps_horizontal = 90  # Threshold for horizontal merging
					# eps_vertical = 50  # Threshold for vertical merging
					eps_horizontal = round(w*5.5/100)#100
					eps_vertical = round(h*2.5/100) #100
					eps_vertical2 = 100  # Threshold for vertical merging
					######################
					# if w>h:
					#     eps_horizontal = round(h*17/100)#100
					#     eps_vertical = round(w*12/100) #100
					# else:
					#     eps_horizontal = round(h*12/100)#100
					#     eps_vertical = round(w*17/100) #100
					############################
					# all_values = data[key]
					print(all_values)
					# for all_values in new_all_values:
					length = len(all_values)
					if length > 1:
						i = 0
						# if (bb1_height > bb1_width and bb2_height > bb2_width) or (bb1_height < bb1_width and bb2_height < bb2_width):
						while i in range(length - 1):
							print(i)
							bb1 = all_values[i][1]
							bb2 = all_values[i + 1][1]
							bb1_token = all_values[i][0]
							bb2_token = all_values[i + 1][0]
							print(f"TOKENS GOING CHECKING FOR MERGING >> {bb1_token} >>and>> {bb2_token}")
		
							confs = [all_values[i][2], all_values[i + 1][2]]
							min_dist_horizontal = minimum_distance(bb1, bb2)
							min_dist_vertical = minimum_distance_vertical(bb1, bb2)
							vertical_flag = check_vertical_indetween(bb1, bb2)
							try:
								IOU_horizontal = get_iou_horizontal(bb1, bb2)
								IOU_vertical = get_iou_vertical(bb1, bb2)
								inter_percentage = get_intersection_percentage(bb1, bb2)
							except:
								i = i + 1
								continue
							bb1_x1, bb1_y1, bb1_x2, bb1_y2 = bb1
							bb2_x1, bb2_y1, bb2_x2, bb2_y2 = bb2
							bb1_width = bb1_x2 - bb1_x1
							bb1_height = bb1_y2 - bb1_y1
							bb2_width = bb2_x2 - bb2_x1
							bb2_height = bb2_y2 - bb2_y1
							# if len(bb1_token)<3:
							print('beore', bb2_width, bb2_height)
							print('bb2_token length', len(bb2_token))
							flag1 = True
							flag2 = True
							flag1 = special_chr_check(bb1_token, flag1)
							flag2 = special_chr_check(bb2_token, flag2)

							if len(bb1_token) == 1 or (len(bb1_token) < 3 and flag1 == False):
								temp = bb1_width
								bb1_width = bb1_height
								bb1_height = temp
							if len(bb2_token) == 1 or (len(bb2_token) < 3 and flag2 == False):
								temp = bb2_width
								bb2_width = bb2_height
								bb2_height = temp
							print('flag2', flag2)
							print('bb2_token', bb2_token)
							print('bb1_width', bb1_width, 'bb1_height', bb1_height)
							print('bb2_width', bb2_width, 'bb2_height', bb2_height)
			
							# if (bb1_height >= bb1_width and bb2_height >= bb2_width) or (
							# 		bb1_height <= bb1_width and bb2_height <= bb2_width):
			
							print('entered into first if')
							print(">>>>>>>>>>RRRRRRRRAAAAAAAAAKKKKKKKEEE", min_dist_horizontal)
							if key in master_keys or vertical_alignment_bbox:
								merge_flag = (min_dist_horizontal <= eps_horizontal or IOU_horizontal > 0 or inter_percentage > 0) or (
										min_dist_vertical <= eps_vertical2 or IOU_vertical > 0 or inter_percentage)
							else:
								merge_flag = (min_dist_horizontal <= eps_horizontal or IOU_horizontal > 0 or inter_percentage > 0) and (
										min_dist_vertical <= eps_vertical or IOU_vertical > 0 or inter_percentage)# or vertical_flag) 
								print("Yes entered in the else block", merge_flag)
							print('min_dist_vertical', min_dist_vertical)
							print(f"eps_horizontal = {eps_horizontal}, eps_vertical = {eps_vertical}")
							print("IOU_horizontal > 0 or inter_percentage", IOU_horizontal, inter_percentage)
							print("IOU_vertical > 0 or inter_percentage or vertical_flag", IOU_vertical, inter_percentage, vertical_flag)
							if merge_flag:
								print('entered into second if')
								print("merging: " + all_values[i][0] + " and " + all_values[i + 1][0])
								x_left = min(bb1[0], bb2[0])
								y_top = min(bb1[1], bb2[1])
								x_right = max(bb1[2], bb2[2])
								y_bottom = max(bb1[3], bb2[3])
								box = [x_left, y_top, x_right, y_bottom]
								text = model_output_sum(key, box, model_output)
								print("merged text is ", text)
								avg_confs = (confs[0] * area(bb1) + confs[1] * area(bb2)) / (area(bb1) + area(bb2))
								new_value = [text, box, avg_confs]
								print(new_value)
								all_values.remove(all_values[i])
								all_values.remove(all_values[i])
								all_values.insert(i, new_value)
								print(all_values)
								length = len(all_values)
								if length == 1:
									print("will break")
									break
								# else:
								# 	print("distance is very high")
								# 	i = i + 1
							else:
								i = i + 1
					else:
						print("will continue")
						continue
			########################################################################################################
			########################################################################################################
			print('before merge_by_skipping_running', all_values)
			data[key] = merge_by_skipping_running(model_output, w, h, key, all_values)\
       
			print("################# all_values #####################", all_values)
			# data[key] = all_values
		else:
			print(f'Vertical merging not happening ++++++++++ {key} ++++++++++++++')
			print(key)
			bboxes = [x[1] for x in data[key]]
			if w > h:
				v_eps = 10#round(h * 1.5 / 100)  # 10round(number)
				h_eps = 50 #36 #round(w * 5 / 100)  # 36
			else:
				v_eps = 10#round(h * 1.1 / 100)  # 10round(number)
				h_eps = 50#36 #round(w * 5.8 / 100)  # 36
			all_values = data[key]
			print('all_values >>>>>>>>>>>', all_values)
			length = len(all_values)
			if length > 1:
				i = 0
				while i in range(length - 1):
					print(i)
					bb1 = all_values[i][1]
					bb2 = all_values[i + 1][1]
					confs = [all_values[i][2], all_values[i + 1][2]]
					# ocr_confs = [all_values[i][3],all_values[i+1][3]]
					# min_dist = minimum_distance(bb1, bb2)
					vertical_distance = check_vertical_distribution(bb1, bb2)
					# dist_bwt_words = (abs(bb1[2]-bb2[0])/w)*100
					hori_distance = abs(bb1[2] - bb2[0])
					print('min_dist========>', vertical_distance, 'bb1', bb1, 'bb2', bb2)
					print('hori_distance ==========>', hori_distance)
					try:
						IOU = get_iou_new(bb1, bb2)
						print('IOU>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', IOU)
					except:
						i = i + 1
						continue
					if (vertical_distance <= v_eps and hori_distance < h_eps) or IOU > 0.1:
						print("merging: " + all_values[i][0] + " and " + all_values[i + 1][0])
						x_left = min(bb1[0], bb2[0])
						y_top = min(bb1[1], bb2[1])
						x_right = max(bb1[2], bb2[2])
						y_bottom = max(bb1[3], bb2[3])
						box = [x_left, y_top, x_right, y_bottom]
						text = model_output_sum(key, box, model_output)
						print("merged text is ", text)
						avg_confs = (confs[0] * area(bb1) + confs[1] * area(bb2)) / (area(bb1) + area(bb2))
						"""if "NA" in ocr_confs:
                            avg_ocr_confs = "NA"
                        else:
                            avg_ocr_confs = ( ocr_confs[0]* area(bb1) + ocr_confs[1]*area(bb2) )/(area(bb1) + area(bb2))"""
						new_value = [text, box, avg_confs]
						print(new_value)
						all_values.remove(all_values[i])
						all_values.remove(all_values[i])
						all_values.insert(i, new_value)
						print(all_values)
						length = len(all_values)
						if length == 1:
							print("will break")
							break
					else:
						print("distance is very high")
						i = i + 1
			else:
				print("will continue")
				continue
		
		if key=="consignee_address":
			print('YES')
			print(data[key])
			print('True')
			# exit('+++++++++++++++++=')
	

# find most common label
def most_common(lst):
	return max(set(lst), key=lst.count)


# converts a tuple to string
def tuple_to_string(sen):
	str_test = sen
	word = ""
	for i, w in enumerate(str_test):
		if i == 0:
			word = word + w
		else:
			word = word + ' ' + w
	return word


# runs the lookup script for comparison with main model

# lookup function to find contries in 'bill to', 'ship to', 'remit to' fields
def lookup(
		text,
		n_words,
		match_threshold,
		file_path,
		result_set,
		key
):
	try:
		out_dict = {}
		out_list = []
		all_box = []
		message = ""

		file = open(file_path, "r")
		lines = file.readlines()

		txt_words = []
		for l in lines:
			line = l.split("\n")
			txt_words.append(line[0])

		if n_words > 5:
			status = "N words larger than 5, provide N words less than 5"
			out_dict["status"] = status
			return out_dict

		if match_threshold < 80:
			status = "Matching threshold value less than 80, provide Matching threshold greater than 80"
			out_dict["status"] = status
			return out_dict
		else:
			res = re.sub(r"[^\w\s]", "", text)
			gram_list = []
			for j in range(n_words):
				gram_count = ngrams(res.split(), j + 1)

				for gram in gram_count:
					sen = tuple_to_string(gram)
					gram_list.append(sen)

			for word in gram_list:
				for txt_char in txt_words:

					# print(word[0].lower())

					if fuzz.ratio(word.lower(), txt_char) > match_threshold:
						# if word.lower() in txt_words:
						# print(word, "----", txt_char, fuzz.ratio(word.lower(), txt_char))

						info_dict = {}
						info_dict["searched_string"] = word  # searched string is our data.
						info_dict["found_string"] = txt_char  # found string is present in countries.txt (lookup file)
						info_dict["string_match_value"] = fuzz.ratio(
							word.lower(), txt_char
						)
						out_list.append(info_dict)
			if len(out_list) != 0:
				for res in out_list:
					look_up = res['found_string']
					# print(found)
					original = res['searched_string']
					original_words = original.split()
					for val in result_set[key]:
						for word in original_words:
							if fuzz.ratio(word.lower(), val[0]) > match_threshold:
								all_box.append(val[1])
					x1 = min([x[0] for x in all_box])
					x2 = max([x[2] for x in all_box])
					y1 = min([x[1] for x in all_box])
					y2 = max([x[3] for x in all_box])
					box_result = [x1, y1, x2, y2]
					res['bbox'] = box_result

			print(out_list)

			return out_list
		file.close()
	except Exception as e:
		print(traceback.format_exc())


# Normalizes all points after zooming
def normalize(points: list, width: int, height: int) -> list:
	x0, y0, x2, y2 = [int(p) for p in points]

	x0 = int(1000 * (x0 / width))
	x2 = int(1000 * (x2 / width))
	y0 = int(1000 * (y0 / height))
	y2 = int(1000 * (y2 / height))

	if x0 > 1000:
		x0 = 1000
	# print(">")
	if x0 < 0:
		x0 = 0
	# print("<")
	if x2 > 1000:
		x2 = 1000
	# print(">")
	if x2 < 0:
		x2 = 0
	# print("<")
	if y0 > 1000:
		y0 = 1000
	# print(">")
	if y0 < 0:
		y0 = 0
	# print("<")
	if y2 > 1000:
		y2 = 1000
	# print(">")
	if y2 < 0:
		y2 = 0
	# print("<")
	return [x0, y0, x2, y2]


# Unnormalize the file
def unnormalize_box(bbox, width, height):
	return [
		int(width * (bbox[0] / 1000)),
		int(height * (bbox[1] / 1000)),
		int(width * (bbox[2] / 1000)),
		int(height * (bbox[3] / 1000)),
	]


# used to order the field values inside a label. Refer Videos.
def contour_sort(a, b):
	if abs(a[1][1] - b[1][1]) <= 15:
		return a[1][0] - b[1][0]
	return a[1][1] - b[1][1]

def are_on_same_line(bbox1, bbox2, min_distance=0, tolerance=10):
    # Check if the vertical distance between the bottom of bbox1 and the top of bbox2 is within the tolerance
    # and if the overall distance is at least min_distance
    return (
        abs(bbox2[1] - bbox1[1]) <= tolerance
        and abs(bbox2[0] - bbox1[2]) >= min_distance
    )

def group_tokens_by_line(bbox_data, line_tolerance=5):
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

def validate_contour_sort(word_bbox):
	all_bboxes = []
	for i in word_bbox:
		all_bboxes.append(i[-1])
	print(all_bboxes)
	final_word_bbox = []
	ordered_bbox = group_tokens_by_line(all_bboxes)
	for values in ordered_bbox:
		for val in values:
			for j in word_bbox:
				if j[-1] == val:
					final_word_bbox.extend([j])
	return final_word_bbox

# OCR Vision function
def get_ocr_vision_api_(file):
	t1 = datetime.now()
	image = file
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "spheric-time-383904-f1b421d86eef.json"
	ctxt = b64encode(image.read()).decode()
	client = vision.ImageAnnotatorClient()
	image = vision.Image(content=ctxt)

	response = client.text_detection(image=image)
	for res in response.text_annotations:
		print(res.confidence)

	word_coordinates = []
	all_text = ""
	logger.info("is word_coordinates is instance of list? %s", isinstance(word_coordinates, list))
	logger.info("is all_text is instance of str? %s", isinstance(all_text, str))
	for i, text in enumerate(response.text_annotations):
		if i != 0:
			vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
			x1 = min([v.x for v in text.bounding_poly.vertices])
			x2 = max([v.x for v in text.bounding_poly.vertices])
			y1 = min([v.y for v in text.bounding_poly.vertices])
			y2 = max([v.y for v in text.bounding_poly.vertices])
			if x2 - x1 == 0:
				x2 += 1
			if y2 - y1 == 0:
				y2 += 1
			""""left": x1,
				"top": y1,
				"width": x2 - x1,
				"height": y2 - y1,"""
			word_coordinates.append({
				"word": text.description,
				"left": x1,
				"top": y1,
				"width": x2 - x1,
				"height": y2 - y1,
				"x1": x1,
				"y1": y1,
				"x2": x2,
				"y2": y2
			})
		else:
			all_text = text.description
	t2 = datetime.now()
	im_name = list(file.split('/'))[-1]
	logger.info(f"OCR process time for {im_name}:" + str(t2 - t1))
	return word_coordinates, all_text


def get_ocr_vision_api(image_path):
	t1 = datetime.now()
	# Open the image file
	with open(image_path, 'rb') as image_file:
		ctxt = b64encode(image_file.read()).decode()

	# Set the Google Cloud credentials file
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/datadrive/TradeFinance_rakesh/google_vision_key.json"

	# Create a Vision API client
	client = vision.ImageAnnotatorClient()
	image = vision.Image(content=ctxt)

	# Perform text detection
	response = client.text_detection(image=image)

	word_coordinates = []
	all_text = ""

	for i, text in enumerate(response.text_annotations):
		if i != 0:
			vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
			x1 = min([v.x for v in text.bounding_poly.vertices])
			x2 = max([v.x for v in text.bounding_poly.vertices])
			y1 = min([v.y for v in text.bounding_poly.vertices])
			y2 = max([v.y for v in text.bounding_poly.vertices])
			if x2 - x1 == 0:
				x2 += 1
			if y2 - y1 == 0:
				y2 += 1

			word_coordinates.append({
				"word": text.description,
				"left": x1,
				"top": y1,
				"width": x2 - x1,
				"height": y2 - y1,
				"x1": x1,
				"y1": y1,
				"x2": x2,
				"y2": y2
			})
		else:
			all_text = text.description

	t2 = datetime.now()
	im_name = os.path.basename(image_path)
	return word_coordinates, all_text


# OCR tesseract function
def get_ocr_tesserract(img, file):
	t1 = datetime.now()
	print("called Image OCR...")
	d = pytesseract.image_to_data(img)
	all_text = pytesseract.image_to_string(img)
	word_coordinates = []
	logger.info("is word_coordinates is instance of list? %s", isinstance(word_coordinates, list))
	for i, b in enumerate(d.splitlines()):
		if i != 0:
			b = b.split()
			if len(b) == 12:
				word = b[11]
				x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
				word_coordinates.append({
					"word": word,
					"left": x,
					"top": y,
					"width": w,
					"height": h,
					"x1": x,
					"y1": y,
					"x2": x + w,
					"y2": y + h
				})
	t2 = datetime.now()
	logger.info(f"OCR process time for {file}:" + str(t2 - t1))
	return word_coordinates, all_text


# GETOCR FITZ
def get_text_fitz(page):
	print("called Text OCR...", end="")
	word_coordinates = []
	logger.info("is word_coordinates is instance of list? %s", isinstance(word_coordinates, list))
	all_text = ""
	text = page.getText("words")
	for t in text:
		word_coordinates.append({
			"word": t[4],
			"left": t[0] * zoom,
			"top": t[1] * zoom,
			"right": t[2] * zoom,
			"bottom": t[3] * zoom,
			"x1": t[0] * zoom,
			"y1": t[1] * zoom,
			"x2": t[2] * zoom,
			"y2": t[3] * zoom,
		})
	return word_coordinates, all_text


def cal_word_dist(wc):
	dis_values = []
	for i in range(len(wc) - 1):
		dis_values.append(abs(wc[i]['x2'] - wc[i + 1]['x1']))
	mode_val = stats.mode(dis_values)
	return mode_val[0].item()


# "****************************************************************************************************************************************"
# function called when image file is passed(png, jpeg, tif)
def image_result(file, path01, model, processor, device, folder_path):
	result_path = f"{folder_path}/Results_{path01}/"
	label_path = os.path.join(folder_path, "classes.txt")
	label_path = '/datadrive/Trained_Models/Extraction/BOL/classes.txt'
	ocr_path = os.path.join(folder_path, "OCR")
	if not os.path.exists(result_path):
		os.makedirs(result_path)
	set_basic_config_for_logging(filename="inference", folder_path = folder_path)
	logger = get_logger_object_and_setting_the_loglevel()
	#############################################################
	logger.info(f"file name is:{file}")
	logger.info(f"folder path :{path01}")
	t_total_start = datetime.now()
	logger.info("Calling Image result")
	all_page_result: dict = {}
	logger.info("is all_page_result is instance of dict? %s", isinstance(all_page_result, dict))
	#############################################################

	# initialising basic variables
	count = 0

	if len(path01) == 0:
		im = Image.open(os.path.join(file))
		file = list(file.split('/'))[-1]
	else:
		im = Image.open(os.path.join(folder_path, path01, file))

		############################################################
	for i, image in enumerate(ImageSequence.Iterator(im)):
		count += 1
		print(f"********** Index : {i} ***************")
		print("******* Page " + str(count) + "********")

		t_page_start = datetime.now()
		w, h = image.size
		temp = image.convert("L")
		image_data = np.asarray(temp)
		image = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
		arr = transform(image)
		#only for abbyy OCR
		'''word_coordinates = json.load(open(os.path.join(ocr_path, 
																file[:-4] + ".json"), "r"))'''
		try:
			with open(os.path.join(ocr_path, file[:-4] + "_text.txt"), "r") as f:
				word_coordinates = json.load(f)['word_coordinates']
				# Debug statements
				# print("word_coordinates are:")
				# print(word_coordinates)
			f.close()

		except Exception as e:
			print(e)
			# exit("Image ocr not found")
			try:
				# print(os.path.join(folder_path, path01, file))
				word_coordinates, all_text = get_ocr_vision_api(os.path.join(folder_path,
				                                                             path01, file))
				# word_coordinates, all_text = get_ocr_tesserract(image,file)
			except:
				word_coordinates, all_text = get_ocr_vision_api(os.path.join(folder_path,
				                                                             path01, file))

		# check if infence is already present
		if os.path.exists(os.path.join(result_path, file[:-4] + str(count) + "model_output.txt")) and \
				os.path.exists(os.path.join(result_path, file[:-4] + str(count) + "_lookup.txt")) and \
				os.path.exists(os.path.join(result_path, file[:-4] + str(count) + ".txt")) and \
				os.path.exists(os.path.join(result_path, file[:-4] + str(count) + ".png")) and \
				os.path.exists(os.path.join(result_path, file[:-4] + "all_page_result.txt")):
			print(f"Inference for this image {file} already exists")
			continue

		if len(word_coordinates) == 0:
			print("Not enough text")

		words: list = []
		bboxes: list = []
		bounding_boxes: list = []
		logger.info("is words is instance of list? %s", isinstance(words, list))
		logger.info("is bboxes is instance of list? %s", isinstance(bboxes, list))
		logger.info("is bounding_boxes is instance of list? %s", isinstance(bounding_boxes, list))

		for t in word_coordinates:
			if 'right' in list(t.keys()):
				t['x1'] = t['left']
				t['y1'] = t['top']
				t['x2'] = t['right']
				t['y2'] = t['bottom']
			words.append(t['word'])
			bounding_boxes.append([t['x1'], t['y1'], t['x2'], t['y2']])
			bboxes.append(normalize([t['x1'], t['y1'], t['x2'], t['y2']], w, h))

		# Debug statements
		# print(words)
		# print(bboxes)
		# print(bounding_boxes)
		# exit()

		encoded_inputs = processor(arr, words, boxes=bboxes, return_tensors="pt")
		print(
			'#################**********************************************###########################################################')
		# exit()
		# print(encoded_inputs)

		######################################################################
		input_id_chunks = list(encoded_inputs['input_ids'][0].split(510))
		# print(input_id_chunks)
		token_type_id_chunks = list(encoded_inputs['token_type_ids'][0].split(510))
		# print(token_type_id_chunks)
		mask_chunks = list(encoded_inputs['attention_mask'][0].split(510))
		# print(mask_chunks)
		bbox_chunks = list(encoded_inputs['bbox'][0].split(510))
		# print(bbox_chunks)
		image_chunk = encoded_inputs['image'][0]
		# print(image_chunk)
		image_chunks = list()
		logger.info("is input_id_chunks is instance of list? %s", isinstance(input_id_chunks, list))
		logger.info("is mask_chunks is instance of list? %s", isinstance(mask_chunks, list))
		logger.info("is bbox_chunks is instance of list? %s", isinstance(bbox_chunks, list))
		logger.info("is image_chunks is instance of list? %s", isinstance(image_chunks, list))
		# loop through each chunk}
		# exit()
		#######################################################################

		for i in range(len(input_id_chunks)):
			image_chunks.append(image_chunk)
			# add CLS (start-of-sequence) and SEP (separator) tokens to input IDs
			input_id_chunks[i] = torch.cat([
				torch.tensor([101]), input_id_chunks[i], torch.tensor([102])
			])
			token_type_id_chunks[i] = torch.cat([
				torch.tensor([0]), token_type_id_chunks[i], torch.tensor([0])
			])
			# add attention tokens to attention mask
			mask_chunks[i] = torch.cat([
				torch.tensor([1]), mask_chunks[i], torch.tensor([1])
			])
			bbox_chunks[i] = torch.cat([
				torch.tensor([[0, 0, 0, 0]]), bbox_chunks[i], torch.tensor([[0, 0, 0, 0]])
			])

			# get required padding length
			pad_len = 512 - input_id_chunks[i].shape[0]
			# check if tensor length satisfies required chunk size
			if pad_len > 0:
				# if padding length is more than 0, we must add padding
				input_id_chunks[i] = torch.cat([
					input_id_chunks[i], torch.Tensor([0] * pad_len)
				])
				token_type_id_chunks[i] = torch.cat([
					token_type_id_chunks[i], torch.Tensor([0] * pad_len)
				])
				mask_chunks[i] = torch.cat([
					mask_chunks[i], torch.Tensor([0] * pad_len)
				])

				bbox_chunks[i] = torch.cat([
					bbox_chunks[i], torch.Tensor([[0, 0, 0, 0]] * pad_len)
				])

		###################################################
		input_ids = torch.stack(input_id_chunks)
		attention_mask = torch.stack(mask_chunks)
		token_type_ids = torch.stack(token_type_id_chunks)
		bbox = torch.stack(bbox_chunks)
		images = torch.stack(image_chunks)
		####################################################

		####################################################
		input_dict = {
			'input_ids': input_ids.long().to(device),
			'attention_mask': attention_mask.float().to(device),
			'token_type_ids': token_type_ids.long().to(device),
			'bbox': bbox.long().to(device),
			'image': images.float().to(device)
		}
		#####################################################

		logger.info("is input_dict is instance of dict? %s", isinstance(input_dict, dict))
		outputs = model(**input_dict)
		logger.info("Model Called")
		logger.info(f'RAM memory % a used for {file}:', psutil.virtual_memory()[2])
		# print(outputs)

		#######################################################
		all_predictions: list = []
		all_boxes: list = []
		all_confidences: list = []
		all_text: list = []
		logger.info("is all_predictions is instance of list? %s", isinstance(all_predictions, list))
		logger.info("is all_boxes is instance of list? %s", isinstance(all_boxes, list))
		logger.info("is all_confidences is instance of list? %s", isinstance(all_confidences, list))
		logger.info("is all_text is instance of list? %s", isinstance(all_text, list))
		########################################################

		# this is the list of classes that will be given to us to be extracted.
		with open(label_path, "r") as f:
			labels = eval(f.read())
		f.close()

		# Creating two dictionaries labels2id and id2labels
		labels = [x.replace("S-", "") for x in labels]
		label2id = {label: idx for idx, label in enumerate(labels)}
		id2label = {idx: label for idx, label in enumerate(labels)}
		logger.info("is label2id is instance of dict? %s", isinstance(label2id, dict))
		logger.info("is id2label is instance of dict? %s", isinstance(id2label, dict))

		# fixed 80 unique hexcodes has been created if more than 80 classes will be there
		# this needs to change
		##############################################################################
		number_of_colors: int = 200
		# creating random hexcodes
		color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
		         for i in range(number_of_colors)]
		##############################################################################

		# color for each label
		label2color = {}
		logger.info("is label2color is instance of dict? %s", isinstance(label2color, dict))
		for i, l in enumerate(labels):
			label2color[l] = color[i]

		logger.info(f"label2color : {label2color}")
		########################################################################3

		for i, output in enumerate(outputs.logits):
			# print(i, output)
			# converting back into PIL image
			new_img = transform2(arr)
			# loading the image font
			font = ImageFont.truetype(font="./arial.ttf", size=20)

			predictions = output.argmax(-1).squeeze().tolist()
			# print('the predictions', predictions)

			confidences = softmax(output.cpu().detach().numpy(), axis=1)
			# print(confidences)

			max_confidences = np.max(confidences, axis=1).reshape(confidences.shape[0], -1)
			# print(max_confidences)

			all_confidences += [x[0] for x in max_confidences]
			# print(all_confidences)

			token_boxes = bbox_chunks[i].squeeze().tolist()
			width, height = new_img.size
			true_predictions = [id2label[prediction] for prediction in predictions]
			all_predictions += true_predictions
			# print('all_predictions',all_predictions)

			true_boxes = [unnormalize_box(box, width, height) for box in token_boxes]
			all_boxes += true_boxes
			# print(all_boxes)

			for id in input_dict['input_ids'][i]:
				all_text.append(processor.tokenizer.decode(id))

		# print(all_text)
		# exit()

		del outputs

		new_img = transform2(arr)
		draw = ImageDraw.Draw(new_img)
		# print("%20s - %30s - %12s - %30s" % ("Text", "Prediction", "Confidence", "Bounding Box"))

		curr_box: list = []
		results_pred: list = []
		results_conf: list = []
		results_bbox: list = []
		results_text: list = []
		temp_preds: list = []
		temp_confs: list = []
		temp_text: list = []
		sep_index: list = all_text.index('[SEP]')

		if len(all_text) > 512:
			if '[PAD]' in all_text:
				sep_index = all_text.index('[PAD]') - 2
			else:
				sep_index = len(all_text) - 3

		# print(sep_index)
		for i in range(len(all_text)):
			if all_text[i] not in ['[CLS]', '[SEP]', '[PAD]']:  # and all_predictions[i] != 'O':
				# print(i)
				if (curr_box != all_boxes[i] and \
				    len(temp_text) > 0) or \
						(i == sep_index - 1 and len(temp_text) > 0):

					# print("1: ", all_text[i])
					if i == sep_index - 1:
						temp_text.append(all_text[i])
						temp_confs.append(all_confidences[i])
						temp_preds.append(all_predictions[i])
					text = ""
					pred = "O"
					preds = [x for x in temp_preds if x != 'O']
					conf = 0
					if len(preds) > 0:
						pred = most_common(preds)

					for j in range(len(temp_text)):
						text += temp_text[j].replace("##", "")
						if temp_preds[j] == pred:
							conf += temp_confs[j]

					conf = float(np.round(conf * 100 / len(temp_text), 2))

					results_text.append(text)
					results_conf.append(conf)
					results_pred.append(pred)
					results_bbox.append(curr_box)

					temp_text = []
					temp_confs = []
					temp_preds = []
					if i != sep_index - 1:
						temp_text.append(all_text[i])
						temp_confs.append(all_confidences[i])
						temp_preds.append(all_predictions[i])
						curr_box = all_boxes[i]
				elif curr_box == all_boxes[i]:
					temp_text.append(all_text[i])
					temp_confs.append(all_confidences[i])
					temp_preds.append(all_predictions[i])
				elif len(temp_text) == 0:
					temp_text.append(all_text[i])
					temp_confs.append(all_confidences[i])
					temp_preds.append(all_predictions[i])
					curr_box = all_boxes[i]
		# print('#',results_text)
		# print('##',results_conf)
		# print(results_pred)
		# print(results_bbox)
		# exit()
		result_set = {}
		logger.info("is result_set is instance of dict? %s", isinstance(result_set, dict))

		for i in range(len(results_pred)):
			if results_pred[i] != 'O':
				if results_pred[i] not in list(result_set.keys()):
					result_set[results_pred[i]] = []
				result_set[results_pred[i]].append([results_text[i],
				                                    results_bbox[i], results_conf[i]])
		# print(result_set)
		model_output = result_set.copy()
		print(f'model output++++++++++++++++++++++++++++=={model_output}')
		# exit()
		print("+++++++++++++++++++reached here+++++++++++++++++")
		# exit("+++++++++")
		with open(os.path.join(result_path, file[:-4] + str(count) + "model_output.txt"), "w") as f:
			json.dump(result_set, f)
		final_result_set = {}
		logger.info("is final_result_set is instance of dict? %s", isinstance(final_result_set, dict))
		f.close()
		#######################
		for k in list(result_set.keys()):
			if k not in single_text_labels:
				try:
					alpha = float(configur[k]['ALPHA'])
				except:
					alpha = float(configur['Default']['ALPHA'])
				if len(result_set[k]) > 1:
					print("++++++++++++++entry in this block+++++++++++")
					texts = [x[0] for x in result_set[k]]
					bboxes = [x[1] for x in result_set[k]]
					confs = [x[2] for x in result_set[k]]
					avg_w = np.mean([abs(x[0] - x[2]) for x in bboxes])
					avg_h = np.mean([abs(x[1] - x[3]) for x in bboxes])
					eps = np.sqrt(avg_w ** 2 + avg_h ** 2) * alpha
					# if eps<=0.0:
					# 	eps=0.1
					clustering = DBSCAN(eps=eps, min_samples=1).fit(bboxes)
					label_set = set(clustering.labels_)
					for l in label_set:
						selected = list(np.where(clustering.labels_ == l)[0])
						selected_texts = [x for i, x in enumerate(texts) if i in selected]
						selected_boxes = [x for i, x in enumerate(bboxes) if i in selected]
						selected_confs = [x for i, x in enumerate(confs) if i in selected]
						text_boxes = [[x, y] for x, y in zip(selected_texts, selected_boxes)]
						text_boxes = sorted(text_boxes, key=cmp_to_key(contour_sort))
						text_boxes = validate_contour_sort(text_boxes)
						text_result = ""
						print(k)
						print(text_boxes)
						for tb in text_boxes:
							if text_result == "":
								text_result += tb[0]
							else:
								text_result += " " + tb[0]
						print(text_result)
						x1 = min([x[0] for x in selected_boxes])
						x2 = max([x[2] for x in selected_boxes])
						y1 = min([x[1] for x in selected_boxes])
						y2 = max([x[3] for x in selected_boxes])
						box_result = [x1, y1, x2, y2]
						conf_result = float(np.round(np.mean(selected_confs), 2))
						# print(box_result)
						if k not in list(final_result_set.keys()):
							final_result_set[k] = []
						final_result_set[k].append([text_result, box_result, conf_result])

				else:
					if k not in list(final_result_set.keys()):
						final_result_set[k] = []
					final_result_set[k].append([result_set[k][0][0], result_set[k][0][1], result_set[k][0][2]])
			else:
				if len(result_set[k]) > 1:
					print("++++++++++++++entry in this block+++++++++++")
					texts = [x[0] for x in result_set[k]]
					bboxes = [x[1] for x in result_set[k]]
					confs = [x[2] for x in result_set[k]]
					for i, value in enumerate(zip(texts, bboxes, confs)):
						print(list(value))
						if k not in list(final_result_set.keys()):
							final_result_set[k] = []
						final_result_set[k].append(list(value))
				else:
					if k not in list(final_result_set.keys()):
						final_result_set[k] = []
					final_result_set[k].append([result_set[k][0][0], result_set[k][0][1], result_set[k][0][2]])
     
		# print(final_result_set)
		# print(f"testing: {final_result_set['consignee_address']}")
		merge_surrounding(final_result_set, model_output, w, h)
		# print(f"testing: {final_result_set['consignee_address']}")
		# exit('.................')
		print(final_result_set)
		# exit('final_result_set>>>>>>>>')
		print("+++++++++++reached here after merge surrounding++++++++++")
		for k in list(final_result_set.keys()):
			all_values = final_result_set[k]
			print(all_values)
			# exit('________________')
			# for value in all_values:
			#     draw.rectangle(value[1], outline=label2color[k], width=2)
			#     draw.text((value[1][0] + 5, value[1][1] - 20),
			#                 text=k , fill=label2color[k], font=font)
			for value in all_values:
				draw.rectangle(value[1], outline=label2color[k], width=2)
				draw.text((value[1][0] + 5, value[1][1] - 20),
				          text=k + " - " + str(value[2]), fill=label2color[k], font=font)
		print(final_result_set)
		# exit('++++++++++++======')
		lookup_result = {}
		t_page_end = datetime.now()
		logger.info("Time taken for page " + str(count) + ' of ' + str(file) + ":" + str(t_page_end - t_page_start))
		all_page_result["Page Number " + str(count)] = final_result_set
		for k in list(final_result_set.keys()):
			if k in ["applicant_country", "beneficiary_country"]:
				for val in final_result_set[k]:
					result_country = lookup(val[0], 4, 90, "countries.txt", result_set, k)
					result_company = lookup(val[0], 4, 90, "organization.txt", result_set, k)
					for res in result_company:
						found = res['found_string']
						searched = res['searched_string']
						new_val = val[0].replace(searched, found)
						val[0] = new_val
					# replacing ocr result with correct result
					for res in result_country:
						found = res['found_string']
						searched = res['searched_string']
						new_val = val[0].replace(searched, found)
						val[0] = new_val
					for res in result_country:
						if (str(k) + "-country") not in lookup_result:
							lookup_result[('LUT_' + str(k) + "-country")] = []
						lookup_result[('LUT_' + str(k) + "-country")].append(
							(res['found_string'], res['string_match_value'], res["bbox"]))
					for res in result_company:
						if (str(k) + "-organization") not in lookup_result:
							lookup_result[('LUT_' + str(k) + "-organization")] = []
						lookup_result[('LUT_' + str(k) + "-organization")].append(
							(res['found_string'], res['string_match_value'], res["bbox"]))
		# print(lookup_result)
		with open(os.path.join(result_path, file[:-4] + str(count) + "_lookup.txt"), "w") as f:
			json.dump(lookup_result, f)
		f.close()
		with open(os.path.join(result_path, file[:-4] + str(count) + ".txt"), "w") as f:
			json.dump(final_result_set, f)

		# del image
		# del new_img
		# del encoded_inputs

		f.close()
		new_img.save(os.path.join(result_path, file[:-4] + str(count) + ".png"))
	with open(os.path.join(result_path, file[:-4] + "all_page_result.txt"), "w") as f:
		json.dump(all_page_result, f)
	f.close()
	im.close()
	logger.info(f"*****Generated All Pages****** for {file}")
	logger.info(f'RAM memory % a used for {file} Generating All Pages:', psutil.virtual_memory()[2])
	t_total_end = datetime.now()
	logger.info(f"Processing time taken for all pages of {file}" + str(count) + ":" + str(t_total_end - t_total_start))

	gc.collect()
	return all_page_result
