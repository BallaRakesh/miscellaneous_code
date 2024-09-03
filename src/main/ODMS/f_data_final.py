import pymongo
from ast import literal_eval
import ast
import re
from copy import deepcopy
from json import JSONDecoder
from typing import List, Union, Any, Mapping
import orjson
import pymongo
import configparser
import datetime as dt
from bson import SON
from fastapi import HTTPException
from mongosanitizer.sanitizer import sanitize
from loguru import logger
# from datetime import datetime
from re import compile,IGNORECASE
import pandas as pd
# from datetime import datetime
from pymongo import MongoClient
# from datetime import datetime
from copy import deepcopy
import  pymongo
import uvicorn
from fastapi import FastAPI,Request
import requests
from fastapi.responses import JSONResponse
import json
import re
from mongoengine import *
from mongoengine import connect, StringField, DateTimeField, Document, DictField, IntField, BooleanField
from datetime import datetime
import configparser
from bson import ObjectId, json_util


config = configparser.ConfigParser()
config.read('config.ini')
try:
    DB_NAME = config['WRAPPER']['DB_NAME']
    DB_IP = config['WRAPPER']['DB_IP']
    DB_PORT = config['WRAPPER']['DB_PORT']
    acceptance_amd = config['INFO']['acceptance_amd']
except:
    pass


app= FastAPI()

# db_connect = pymongo.MongoClient('127.0.0.1',27017)
# db  = db_connect.get_database('trade_finance_demo')
# db.MT700.drop()
# db.MASTER_LC.drop()
# db.MT707_AMD.drop()
# db.MT710_ADV.drop()
# db.MT720_TRANS.drop()
# db.master_backup.drop()

# update_list = {'update_MT707_AMD':[['32B','33B'],{'45B':'45A','46B':'46A','47B':'47A', '49M':'49G', '49N':'49H'}], 'update_MT720_TRANS':[['59'], {}], 'update_MT710_AVD':[[],{}]}

# table_name = ['MT707_AMD']
# main_table1 = 'MASTER_LC'
# main_table2 = 'MT700'
# backup_table = 'master_backup'

acceptance_amd = True


###########################################
###########################################
# connect(DB_NAME, host=DB_IP, port=DB_PORT)
connect('trade_finance_demo', host='127.0.0.1', port=27017)

class MT700(Document):
    meta = {"allow_inheritance": True}
    lc_number = StringField(db_field="lc_number")
    work_item_no = StringField(db_field="work_item_no")
    timestamp = DateTimeField(db_field="timestamp")
    lc_info = DictField(db_field="lc_info")

class MASTER_LC(Document):
    lc_number = StringField(db_field="lc_number")
    work_item_no = StringField(db_field="work_item_no")
    timestamp = DateTimeField(db_field="timestamp")
    latest_amd = IntField(db_field="latest_amd", default=0)
    total_valid_amd = IntField(db_field="total_valid_amd", default=0)
    transfer_flag = BooleanField(db_field="transfer_flag", default=False)
    advise_flag = BooleanField(db_field="advise_flag", default=False)
    
# MASTER_LC.objects().update(enabled=True)


class MT707_AMD(Document):
    lc_number = StringField(db_field="lc_number")
    work_item_no = StringField(db_field="work_item_no")
    timestamp = DateTimeField(db_field="timestamp")
    lc_info = DictField(db_field="lc_info")
    Amd_no = IntField(db_field="Amd_no")
    
class MT720_TRANS(Document):
    lc_number = StringField(db_field="lc_number")
    work_item_no = StringField(db_field="work_item_no")
    timestamp = DateTimeField(db_field="timestamp")
    lc_info = DictField(db_field="lc_info")
    
class MT710_ADV(Document):
    lc_number = StringField(db_field="lc_number")
    work_item_no = StringField(db_field="work_item_no")
    timestamp = DateTimeField(db_field="timestamp")
    lc_info = DictField(db_field="lc_info")

class BackupDocument(Document):
    lc_number = StringField(db_field="lc_number")
    work_item_no = StringField(db_field="work_item_no")
    timestamp = DateTimeField(db_field="timestamp")
    latest_amd = IntField(db_field="latest_amd", default=0)
    total_valid_amd = IntField(db_field="total_valid_amd", default=0)
    transfer_flag = BooleanField(db_field="transfer_flag", default=False)
    advise_flag = BooleanField(db_field="advise_flag", default=False)


# MT700.drop_collection()
# MASTER_LC.drop_collection()
# MT707_AMD.drop_collection()
# MT720_TRANS.drop_collection()
# MT710_ADV.drop_collection()



def backup_documents(data_list):
    backup_collection = BackupDocument
    # Iterate over the provided list of dictionaries and save each document to the backup collection
    print('*******************')
    print(data_list)
    # exit('/////')
    for document_data in data_list:
        # Save the document to the backup collection
        document_data.pop('_id', None)
        print(document_data)
        backup_document = backup_collection(**document_data)
        backup_document.save()


def update_MT700(ans, amd_ans, lc_number):

    if '32B' in amd_ans:
        # print(value)
        ans['32B']["amount"] = float(ans['32B']["amount"]) + float(amd_ans['32B']['amount'])
    elif '33B' in amd_ans:
        ans['32B']["amount"] = float(ans['32B']["amount"]) - float(amd_ans['33B']['amount'])
            
    timestamp_ = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    MT700(lc_number=lc_number, work_item_no=lc_number, timestamp=timestamp_,lc_info=ans).save()
                
    
def find_rows_by_lc_number(lc_number):
    final_values = []
    # Query the collection based on lc_number and convert the result to a list
    results = list(MASTER_LC.objects(lc_number=lc_number)) #.as_pymongo()
    for document in results:
        final_values.append(document.to_mongo().to_dict())

    return final_values

        
def update_master(lc_number, key, value, db_name_):
    print(find_rows_by_lc_number(lc_number))
    backup_documents(find_rows_by_lc_number(lc_number))
    # print(find_rows_by_lc_number(lc_number))
    if db_name_ == 'MT707_AMD':
        final_values = find_rows_by_lc_number(lc_number)[0]
        total_valid_amd = final_values.get("total_valid_amd", 0)
        if acceptance_amd == True:
            total_valid_amd+=1
        MASTER_LC.objects(lc_number=lc_number).update_one(set__total_valid_amd = total_valid_amd)
    
    update_field = f"set__{key}"
    MASTER_LC.objects(lc_number=lc_number).update_one(**{update_field: value})



def MT_query(lc_number, mt_value):
    master_data = find_rows_by_lc_number(lc_number)[0]
    if mt_value == 'amendment':
        latest_amd = master_data.get("latest_amd", 0)
        print('>>>>>>>>>>>')
        #########################################################
        ######## here we also can have one condition, checking the is_amd == True when a column added in the master or MT707_AMD as is_amd #########
        ####### here we can also write the quires for extract data when AMD = True | all the cases  #############
        #########################################################
        master_lc_document = list(MT707_AMD.objects(Amd_no=latest_amd).as_pymongo())
        master_lc_document = sorted(master_lc_document, key=lambda x: x["timestamp"], reverse=True)
        master_lc_document = master_lc_document[0]
        print(master_lc_document)
        return master_lc_document
    elif mt_value == "transfer":
        transfer_flag = master_data.get("transfer_flag", False)
        if transfer_flag:
            MT720_db_info = list(MT720_TRANS.objects().as_pymongo())
            MT720_db_info = sorted(MT720_db_info, key=lambda x: x["timestamp"], reverse=True)
            MT720_db_info = MT720_db_info[0]
            return MT720_db_info
    elif mt_value == "advice":
        advise_flag = master_data.get("advise_flag", False)
        if advise_flag:
            MT710_db_info = list(MT720_TRANS.objects().as_pymongo())
            MT710_db_info = sorted(MT710_db_info, key=lambda x: x["timestamp"], reverse=True)
            MT710_db_info = MT710_db_info[0]
            return MT710_db_info
    else:
        pass
    
    
    

    
# Sort the list using the custom sort function
@app.post("/f_data_preparation")
async def f_data_db(req:Request):
    input_json = await req.json()
    print(input_json)
    lc_number = input_json.get("lc_no")
    lc_information = input_json.get("lc_info")
    url = 'http://0.0.0.0:8175/f_data_preparation/classification_fdata'
    response = requests.post(url, json=input_json)
    print(response)
    classify_respone = response.json()
    db_name_ = classify_respone['result']
    print(db_name_)
    timestamp_ = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(db_name_)
    if db_name_ == 'MT707_AMD':  #verfiy that, if idex = 0 , means the most resent one or not ??
        MT700_DB_lc_info = list(MT700.objects().as_pymongo())[-1]['lc_info']
        print(MT700_DB_lc_info)
        update_MT700(MT700_DB_lc_info, lc_information, lc_number)
        update_master(lc_number, 'latest_amd', lc_information['26E'], db_name_)
        #add a flag to accept or reject the amedment ##########################################################
        MT707_AMD(lc_number=lc_number, work_item_no=lc_number, timestamp=timestamp_,lc_info=lc_information, Amd_no = lc_information['26E']).save()
    else:
        if db_name_ == 'MT700':
            MT700(lc_number=lc_number, work_item_no=lc_number, timestamp=timestamp_,lc_info=lc_information).save()
            #create the master
            MASTER_LC(lc_number=lc_number, work_item_no=lc_number, timestamp=timestamp_).save()
        elif db_name_ == 'MT710_ADV':
            MT710_ADV(lc_number=lc_number, work_item_no=lc_number, timestamp=timestamp_,lc_info=lc_information).save()
            update_master(lc_number,'advise_flag', True, db_name_)
        elif db_name_ == 'MT720_TRANS':
            MT720_TRANS(lc_number=lc_number, work_item_no=lc_number, timestamp=timestamp_,lc_info=lc_information).save()
            update_master(lc_number,'transfer_flag', True, db_name_)

        
    return JSONResponse(content={"result":lc_information},status_code=200)



def serialize_custom(obj):
    if isinstance(obj, (datetime, ObjectId)):
        return str(obj)
    elif hasattr(obj, "__dict__"):
        # For objects with __dict__ attribute (custom classes), convert to dictionary
        return obj.__dict__
    else:
        try:
            # Try to use bson.json_util default
            return json_util.default(obj)
        except TypeError:
            raise TypeError(f"Type {type(obj)} not serializable")



@app.post("/f_data_preparation/query")
async def f_data_db(req:Request):
    input_json = await req.json()
    print(input_json)
    lc_number = input_json.get("lc_no")
    MT_value = input_json.get("MT_message")
    mt_result = MT_query(lc_number, MT_value)
    print(mt_result)
    print(type(mt_result))
    mt_result.pop('_id', None)
    print(mt_result)
    master_lc_document_serializable = json.loads(json.dumps(mt_result, default=serialize_custom))
    return JSONResponse(content=master_lc_document_serializable,status_code=200)
        


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8185)
