from pdf2image import convert_from_path
def split_pdf(src: str = None, destination: str = None):
     f = src.split("/")[-1]
     print(f"filename : {f}")
     if f.split(".")[-1].lower() in ["tiff", "tif", "pdf"]:
         # Store Pdf with convert_from_path function
         images = convert_from_path(src)
         f_n = "_".join(f.split(".")[0].split(" "))
         for i in range(len(images)):
             output = f"{f_n}_%d.png" % (i + 1)
             # Save pages as images in the pdf
             images[i].save(f"{destination}/{output}", 'PNG')


src = "/home/tarun/NumberTheory/TradeFinance/Repos/final_delivery_sample_document/FinalEvaluationEvalData/FinalPdfs/DemoPdf/Bills/Product-wise/export-bills/pdf5/export_bill_5.pdf"
destination = "/home/tarun/NumberTheory/TradeFinance/Repos/final_delivery_sample_document/FinalEvaluationEvalData/FinalPdfs/DemoPdf/Bills/Product-wise/export-bills/pdf5"
split_pdf(src, destination)