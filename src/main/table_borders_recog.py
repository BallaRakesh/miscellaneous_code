import pandas as pd
import cv2
import numpy as np


# from google.colab.patches import cv2.imshow
import easyocr
reader = easyocr.Reader(['th','en'])

def table_detection(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img_bin) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = cv2.bitwise_not(img_bin)

    kernel_length_v = (np.array(img_gray).shape[1])//120
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v))
    im_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(im_temp1, vertical_kernel, iterations=3)
    cv2.imwrite('/New_Volume/Rakesh/miscellaneous_code/src/sample_outputs/aaaaaaaa.png', vertical_lines_img)
    
    kernel_length_h = (np.array(img_gray).shape[1])//40
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_h, 1))
    im_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(im_temp2, horizontal_kernel, iterations=3)
    cv2.imwrite('/New_Volume/Rakesh/miscellaneous_code/src/sample_outputs/bbbbbbbbbbb.png', horizontal_lines_img)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    table_segment = cv2.addWeighted(vertical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0)
    table_segment = cv2.erode(cv2.bitwise_not(table_segment), kernel, iterations=2)
    thresh, table_segment = cv2.threshold(table_segment, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite('/New_Volume/Rakesh/miscellaneous_code/src/sample_outputs/cccccccccccc.png', table_segment)
    contours, hierarchy = cv2.findContours(table_segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    full_list=[]
    row=[]
    data=[]
    first_iter=0
    firsty=-1

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if  h > 9 and h<100:
            if first_iter==0:
              first_iter=1
              firsty=y
            if firsty!=y:
              row.reverse()
              full_list.append(row)
              row=[]
              data=[]
            print(x,y,w,h)
            cropped = img[y:y + h, x:x + w]
            cv2.imwrite(f'/New_Volume/Rakesh/miscellaneous_code/src/sample_outputs/itter_in_if_{h}.png', cropped)
            
            # cv2.imshow(cropped)
            # cv2.imshow('Image cropped', cropped)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            bounds = reader.readtext(cropped)


            try:
              data.append(bounds[0][1])
              data.append(w)
              row.append(data)
              data=[]
            except:
              data.append("--")
              data.append(w)
              row.append(data)
              data=[]
            firsty=y
        cv2.rectangle(img,(x, y),(x + w, y + h),(0, 255, 0), 2)
        cv2.imwrite(f'/New_Volume/Rakesh/miscellaneous_code/src/sample_outputs/itter_out_if_{x}.png', img)
        
        # cv2.imshow(img)
        # cv2.imshow('Image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    full_list.reverse()
    print(full_list)

    new_data=[]
    new_row=[]
    for i in full_list:
      for j in i:
        new_row.append(j[0])
      new_data.append(new_row)
      new_row=[]
    print(new_data)

    # Convert list of lists into a DataFrame
    df = pd.DataFrame(new_data)
    df = df.applymap(lambda x: '' if pd.isna(x) else x)
    from tabulate import tabulate
    table = tabulate(df, headers='firstrow', tablefmt='grid')

    # Print DataFrame
    print(table)
#sample calling

table_detection("/New_Volume/Rakesh/miscellaneous_code/src/samples/Covering_Schedule_254_page_5.png")