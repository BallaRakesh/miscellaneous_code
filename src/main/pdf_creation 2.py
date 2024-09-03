import img2pdf
import glob
import os

input = "/home/tarun/Desktop/wrong_images"
output = "/home/tarun/Desktop/wrong_images_output"

for folder in sorted(glob.glob(f"{input}/*"), reverse=True):                                                                                                              
    if os.path.isdir(folder):
        folder_name = folder.split("/")[-1]                                                                                                                                     
        with open(f"{output}/{folder_name}.pdf", "wb") as f:
            f.write(img2pdf.convert([f"{folder}/{i}" for i in sorted(os.listdir(folder), reverse=False) if i.endswith(".png")]))
