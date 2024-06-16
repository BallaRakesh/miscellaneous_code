import OcrToTableTool as ottt
import TableExtractor as te
import TableLinesRemover as tlr
import cv2

path_to_image = "/New_Volume/Rakesh/miscellaneous_code/src/samples/Covering_Schedule_254_page_5.png"
image_new = cv2.imread(path_to_image)
table_extractor = te.TableExtractor(path_to_image)

# perspective_corrected_image = table_extractor.execute()
# cv2.imshow("perspective_corrected_image", perspective_corrected_image)


lines_remover = tlr.TableLinesRemover(image_new)
image_without_lines = lines_remover.execute()
# cv2.imshow("image_without_lines", image_without_lines)

# ocr_tool = ottt.OcrToTableTool(image_without_lines, perspective_corrected_image)
# ocr_tool.execute()

# cv2.waitKey(0)
# cv2.destroyAllWindows()