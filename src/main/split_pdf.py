
import fitz
import os
import multiprocessing
from PyPDF2 import PdfReader, PdfWriter
import io
from pdf2image import convert_from_bytes, convert_from_path

A4_SIZE: tuple = (8.27,11.69)
MODE_MULITPLIER:int = 100
PROCESSING_MODES: list = ["low","balance","super"]
pp_cls_flag = True

def split_pdf(src: str = None, destination: str = None, product_name: str = None,
			  processing_mode:str = None):
	if not processing_mode:
		processing_mode = "balance mode"
	W,H = A4_SIZE 
	filtered_processing_mode = processing_mode.replace("mode","").lower().strip()
	multiplier: int = MODE_MULITPLIER * (PROCESSING_MODES.index(filtered_processing_mode) + 1)
	size: tuple = (multiplier*W, multiplier*H)
	f = src.split("/")[-1]
	print(f"filename : {f}")
	if f.split(".")[-1].lower() in ["tiff", "tif", "pdf"]:
		if product_name in []:
			with fitz.open(src) as doc:
				for i, page in enumerate(doc):
					print(f"i: {i}, page: {page}")
					pix = page.get_pixmap()
					f_n = "_".join(f.split(".")[0].split(" "))
					output = f"{f_n}_%d.png" % (i + 1)
					pix.save(os.path.join(destination, output))
		elif product_name in ["Import LC", "Export LC", "Import Bills", "Export Bills",
		                      "LCIssuance", "LCAmendment", "LCTransfer", "LCAdvising",
		                      "AdvancedImportBills", "AdvancedExportBills"]:
			if pp_cls_flag:
				# Use PdfReader and PdfWriter to read and extract pages
				pdf = PdfReader(src)
				for page_num, page in enumerate(pdf.pages, 1):
					# Extract page as bytes
					page_bytes = io.BytesIO()
					pdf_writer = PdfWriter()
					pdf_writer.add_page(page)
					pdf_writer.write(page_bytes)
					page_bytes.seek(0)
					# Convert page bytes to image with high DPI
					# dpi=300 == "super" Mode , with out dip == "balance" Mode
					# suggestible to use  dpi=300 for getting better performance from PPcls
					images = convert_from_bytes(page_bytes.getvalue(), dpi=300)
					# Save the image in high quality
					f_n = "_".join(f.split(".")[0].split(" "))
					for img in images:
						output = f"{f_n}_%d.png" % (page_num)
						img.save(os.path.join(destination, output), 'PNG', compress_level=0)
						print(f'Saved high-quality page {page_num} as {output}')
						break  # We only need one image per page
			else:
				# Store Pdf with convert_from_path function
				thread_count = int(multiprocessing.cpu_count()/2)
				num_threads = 4 if  thread_count > 4 else thread_count
				images = convert_from_path(src, size=size,thread_count=num_threads)
				f_n = "_".join(f.split(".")[0].split(" "))
				for i in range(len(images)):
					output = f"{f_n}_%d.png" % (i + 1)
					# Save pages as images in the pdf
					images[i].save(f"{destination}/{output}", 'PNG', compress_level=0)
    

src = '/home/ntlpt19/TF_testing_EXT/code/miscellaneous_code/src/main/sample_testing_rotations.pdf'
destination = '/home/ntlpt19/TF_testing_EXT/code/miscellaneous_code/src/main/pdf_exp1'
product_name = "Export Bills"
processing_mode = "super"

split_pdf(src, destination, product_name,processing_mode)
exit('........')
import os
from pdf2image import convert_from_path

def pdf_to_images(pdf_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert PDF to images
    images = convert_from_path(pdf_path)

    # Save each image
    for i, image in enumerate(images):
        image.save(os.path.join(output_folder, f'page_{i+1}.png'), 'PNG')

# Example usage
# pdf_path = '/home/ntlpt19/TF_testing_EXT/code/miscellaneous_code/src/main/orientation_check.pdf'
# output_folder = '/home/ntlpt19/TF_testing_EXT/code/miscellaneous_code/src/main/pdf_exp1'
# # pdf_to_images(pdf_path, output_folder)


import os
from PyPDF2 import PdfReader, PdfWriter
import io
from pdf2image import convert_from_bytes

def pdf_to_images_high_quality(pdf_path, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the PDF
    pdf = PdfReader(pdf_path)
    
    # Iterate through pages
    for page_num, page in enumerate(pdf.pages, 1):
        # Extract page as bytes
        page_bytes = io.BytesIO()
        pdf_writer = PdfWriter()
        pdf_writer.add_page(page)
        pdf_writer.write(page_bytes)
        page_bytes.seek(0)
        # Convert page bytes to image with high DPI
        images = convert_from_bytes(page_bytes.getvalue(), dpi=300)
        # Save the image in high quality
        for img in images:
            image_path = os.path.join(output_folder, f'page_{page_num}.png')
            img.save(image_path, 'PNG', optimize=False, compress_level=0)
            print(f'Saved high-quality page {page_num} as {image_path}')
            break  # We only need one image per page


# pdf_path = '/home/ntlpt19/TF_testing_EXT/code/miscellaneous_code/src/main/sample_testing_rotations.pdf'
# output_folder = '/home/ntlpt19/TF_testing_EXT/code/miscellaneous_code/src/main/pdf_exp1'
# pdf_to_images_high_quality(pdf_path, output_folder)



import os
import multiprocessing
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader, PdfWriter
import io
import fitz

A4_SIZE: tuple = (8.27, 11.69)
MODE_MULTIPLIER: int = 100
PROCESSING_MODES: list = ["low", "balance", "super"]

def split_pdf_testing(src: str = None, destination: str = None, product_name: str = None,
              processing_mode: str = None):
    if not processing_mode:
        processing_mode = "balance mode"
    W, H = A4_SIZE
    filtered_processing_mode = processing_mode.replace("mode", "").lower().strip()
    multiplier: int = MODE_MULTIPLIER * (PROCESSING_MODES.index(filtered_processing_mode) + 1)
    size: tuple = (multiplier * W, multiplier * H)
    f = src.split("/")[-1]
    print(f"filename: {f}")

    if f.split(".")[-1].lower() in ["tiff", "tif", "pdf"]:
        if product_name in []:
            with fitz.open(src) as doc:
                for i, page in enumerate(doc):
                    print(f"i: {i}, page: {page}")
                    pix = page.get_pixmap()
                    f_n = "_".join(f.split(".")[0].split(" "))
                    output = f"{f_n}_%d.png" % (i + 1)
                    pix.save(os.path.join(destination, output))
        elif product_name in ["Import LC", "Export LC", "Import Bills", "Export Bills",
                              "LCIssuance", "LCAmendment", "LCTransfer", "LCAdvising",
                              "AdvancedImportBills", "AdvancedExportBills"]:
            # Use PdfReader and PdfWriter to read and extract pages
            pdf = PdfReader(src)
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract page as bytes
                page_bytes = io.BytesIO()
                pdf_writer = PdfWriter()
                pdf_writer.add_page(page)
                pdf_writer.write(page_bytes)
                page_bytes.seek(0)
                # Convert page bytes to image with high DPI
                images = convert_from_bytes(page_bytes.getvalue(), size=size)
                # Save the image in high quality
                f_n = "_".join(f.split(".")[0].split(" "))
                for img in images:
                    output = f"{f_n}_%d.png" % (page_num)
                    img.save(os.path.join(destination, output), 'PNG', compress_level=0)
                    print(f'Saved high-quality page {page_num} as {output}')
                    break  # We only need one image per page
