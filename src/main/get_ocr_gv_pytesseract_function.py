import os
from google.cloud import vision
import pytesseract

gv_key = '/New_Volume/Rakesh/miscellaneous_code/src/main/gv_key.json'

def get_ocr_vision_api_charConfi(image_path):
	"""
    Performs OCR (Optical Character Recognition) using Google Cloud Vision API and returns character-level confidence scores.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: A tuple containing formatted results (list of dictionaries) and all the extracted text (str).
    """
	# Initialize the Vision API client
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gv_key 
	client = vision.ImageAnnotatorClient()

	# Load the image
	try:
		image_file = open(image_path, 'rb')
		image_data = image_file.read()
	except Exception as e:
		raise HTTPException(status_code=400, detail={"message": "{}".format(e), "error_code": "501"})
	finally:
		if hasattr(image_file,"close"):
			image_file.close()

	# Perform text detection
	image = vision.Image(content=image_data)
	response = client.document_text_detection(image=image)

	# Initialize a list to store the formatted results
	formatted_results = []

	# Initialize a string to store all the extracted text
	all_extracted_text = ""

	# Extract and format the text and bounding box information
	for page in response.full_text_annotation.pages:
		for block in page.blocks:
			for paragraph in block.paragraphs:
				for word in paragraph.words:
					word_text = "".join([symbol.text for symbol in word.symbols])
					confidence = word.confidence
					_ = [symbol.confidence for symbol in word.symbols]

					vertices = [(vertex.x, vertex.y) for vertex in word.bounding_box.vertices]
					x1 = min([v[0] for v in vertices])
					x2 = max([v[0] for v in vertices])
					y1 = min([v[1] for v in vertices])
					y2 = max([v[1] for v in vertices])
					formatted_word = {
						"word": word_text,
						"left": x1,
						"top": y1,
						"width": x2 - x1,
						"height": y2 - y1,
						"confidence": confidence,
						# 'char_confi': char_confidences,  # List of character-level confidences
						"x1": x1,
						"y1": y1,
						"x2": x2,
						"y2": y2,
					}
					formatted_results.append(formatted_word)
					all_extracted_text += word_text + ' '

	return formatted_results, all_extracted_text

def get_ocr_tesseract(image_path):
	"""
    Performs OCR (Optical Character Recognition) using Tesseract OCR engine.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: A tuple containing word coordinates (list of dictionaries) and all the extracted text (str).

    """
	img=None
	word_coordinates, all_text = [],""
	t1 = datetime.now()
	print("called Image OCR...", end="")
	try:
		img = Image.open(image_path)
		d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
		all_text = pytesseract.image_to_string(img)
		for i in range(len(d['text'])):
			word = d['text'][i]
			conf = float(d['conf'][i])
			if conf > 0:
				x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
				word_coordinates.append({
					"word": word,
					"confidence": conf,
					"left": x,
					"top": y,
					"width": w,
					"height": h,
					"x1": x,
					"y1": y,
					"x2": x + w,
					"y2": y + h
				})
	except Exception as e:
		print(f"exception: {e}")	
	finally:
		if hasattr(img,"close"):
			img.close()
	t2 = datetime.now()
	return word_coordinates, all_text

# image_path = '/New_Volume/Rakesh/miscellaneous_code/src/samples/Covering_Schedule_254_page_5.png'
# formatted_results, all_extracted_text = get_ocr_vision_api_charConfi(image_path)
# print(formatted_results)