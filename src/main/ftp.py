# Import Module
import ftplib
 
# Ref Link: https://docs.python.org/3/library/ftplib.html
 
# Fill Required Information
HOSTNAME = "4.224.51.244"
USERNAME = "azureuser"
PASSWORD = "NumberTheory@54321"
 
# Connect FTP Server
ftp_server = ftplib.FTP(HOSTNAME, USERNAME, PASSWORD)
 
# force UTF-8 encoding
ftp_server.encoding = "utf-8"
 
# ftp_server.mkd("TradeFinance/orientation_check1/")#1
# ftp_server.mkd("TradeFinance/orientation_check1/Images")#2
 
# """
# with open('/home/tarun/NumberTheory/TradeFinance/Repos/final_delivery_sample_document/BundleApi/TradeFinance/WI_12345678/WI_123456$
#     ftp_server.retrbinary(' README', fp.write)orientation_check.pdf
# """
 
with open("/home/ntlpt19/TF_testing_EXT/code/miscellaneous_code/src/main/orientation_check.pdf", "rb") as local_file:
    ftp_server.storbinary(f'STOR /TradeFinance/orientation_check1/orientation_check1.pdf', local_file)#3
 
 
 
# with open('/home/tarun/NumberTheory/TradeFinance/Repos/final_delivery_sample_document/BundleApi/TradeFinance/WI_12345678/WI_12345678.pdf', 'wb') as fp:
#     ftp_server.retrbinary('RETR Images/Export-Bill_0.png', fp.write)