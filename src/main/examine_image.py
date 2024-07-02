from PIL import Image
import numpy as np

def compare_images(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path).convert('RGB'))
    img2 = np.array(Image.open(img2_path).convert('RGB'))
    print(img1.shape)
    print(img2.shape)
    if img1.shape != img2.shape:
        print("Images have different dimensions")
        return
    difference = np.sum(np.abs(img1 - img2))
    print(f"Total pixel difference: {difference}")
    if difference == 0:
        print("Images are identical")
    else:
        print("Images differ")

compare_images('./pdf_exp1/page_15.png','pdf_exp_3/page_1.png')


from PIL import Image
from PIL.ExifTags import TAGS

def get_image_metadata(image_path):
    image = Image.open(image_path)
    metadata = {}
    
    # Try to get EXIF data
    exif_data = image._getexif()
    if exif_data:
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            metadata[tag] = value
    
    # Get general image info
    metadata['format'] = image.format
    metadata['mode'] = image.mode
    metadata['size'] = image.size
    
    # Try to get other metadata
    for key, value in image.info.items():
        if isinstance(value, (int, float, str, bytes)):
            metadata[key] = value
    
    return metadata

# Usage
image_path = './pdf_exp1/page_15.png'
metadata = get_image_metadata(image_path)
print(metadata)

# metadata_25 = get_image_metadata('./pdf_exp1/page_15.png')
# metadata_3 = get_image_metadata('pdf_exp_3/page_1.png')
# print("Metadata from 25-page PDF:", metadata_25)
# print("Metadata from 3-page PDF:", metadata_3)