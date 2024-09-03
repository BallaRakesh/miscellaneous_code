"""
* *********************************************************************************
* Number Theory S/W Pvt. Ltd CONFIDENTIAL                                       *
* *
* [2016] - [2023] Number Theory S/W Pvt. Ltd Incorporated                       *
* All Rights Reserved.                                                          *
* *
* NOTICE:  All information contained herein is, and remains                     *
* the property of Number Theory S/W Pvt. Ltd Incorporated and its suppliers,    *
* if any.  The intellectual and technical concepts contained                    *
* herein are proprietary to Number Theory S/W Pvt. Ltd Incorporated             *
* and its suppliers and may be covered by India. and Foreign Patents,           *
* patents in process, and are protected by trade secret or copyright law.       *
* Dissemination of this information or reproduction of this material            *
* is strictly forbidden unless prior written permission is obtained             *
* from Number Theory S/W Pvt. Ltd Incorporated.                                 *
* *
* *********************************************************************************
"""

import re

import openai
from src.main.processed_lc.config import string_character_validation, character_length_map, \
    character_set_map, possible_values_map
from typing import Tuple
import re
import json
import spacy
import time

# model loading
# time taking job
start_time =  time.time()  
nlp = spacy.load("en_core_web_sm")
time_taken = time.time() - start_time  
print(f"time taken in loading space ner model: {time_taken:.6f} seconds")

# MT 700 Code
# first
character_set_json_path: str  = "src/data/character_sets.json"
with open(f"{character_set_json_path}", "r") as file:
    character_set_json = json.load(file) 
print(character_set_json)


def pre_operation_decorator(func):
    def wrapper(*args, **kwargs):
        print("kwargs:", kwargs)
        # Perform pre-operation
        kwargs["value"] = kwargs["value"].strip()
        
        # Call the original function
        result = func(*args, **kwargs)
        
        # Optionally, you can perform post-operation here
        
        return result
    
    return wrapper

def is_special_character(char):
    special_characters = "!@#$%^&*()-_=+[]{}|;:'\",.<>?/"
    
    if char in special_characters:
        return True
    else:
        return False

@pre_operation_decorator
def find_company_and_address(value):
    # Process the input text with spaCy NLP pipeline
    doc = nlp(value)
    
    # Extract company names and addresses using NER
    companies = []
    addresses = []
    
    for ent in doc.ents:
        if ent.label_ == 'ORG':
            companies.append(ent.text)
        elif ent.label_ == 'LOC' or ent.label_ == 'GPE':
            addresses.append(ent.text)
    
    return companies, addresses


@pre_operation_decorator
def test_pre(value):
    return value


def get_country_from_place(value):
    """
    # Not implemented yet
    """
    return value


@pre_operation_decorator
def handling_sequence_total(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None) -> Tuple:
    """
    sequence of total
    1!n/1!n
    """
    
    # regex based filtering
    
    
    # initial variable declarations
    content_validation_seq_number = False
    length_validation_seq_number = False    
    content_validation_sequence_total = False
    length_validation_sequence_total = False

    
    sequence_number, total_pages =  value.split('/')    
    if len(total_pages) <= character_length_map[mt_value][field_name][inner_fields[0]]:
        length_validation_sequence_total = True
    
    if len(sequence_number) <= character_length_map[mt_value][field_name][inner_fields[1]]:
        length_validation_seq_number = True
    
    
    if re.match(character_set_json["n"]["name"], sequence_number) and \
        (1 <= int(sequence_number) <= 8):
        content_validation_seq_number = True
    
    
    if re.match(character_set_json["n"]["name"], total_pages) and \
        (1 <= int(sequence_number) <= 8):
        length_validation_sequence_total = True
    
    
    assert isinstance(int(sequence_number), int) == True
    assert isinstance(int(total_pages), int) == True   

    return sequence_number, total_pages,\
        content_validation_seq_number, length_validation_seq_number,\
        content_validation_sequence_total, length_validation_sequence_total

@pre_operation_decorator
def handle_formofdocumentary_credit(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None) -> Tuple:
    """
    Form of Documentary Credit
    """
    content_validation = False
    length_validation = False
    
    # strip the beginning and trailing spaces 
    # replace the extra spaces with single spaces
    value = re.sub(r'\s+', ' ', value.strip())

    if re.match(character_set_json["x"]["regex"], value) and \
        value.upper() in ["IRREVOCABLE", "IRREVOCABLE TRANSFERABLE", "IRREVOCABLETRANSFERABLE"]:
        content_validation = True
    
        if value and len(value) <= character_length_map[mt_value][field_name]:
            length_validation = True
    
    return value, content_validation, length_validation

@pre_operation_decorator
def handle_documentary_credit_number(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    
    value = value.strip()
    content_validation = False
    length_validation = False
    
    if re.match(character_set_json["x"]["regex"], value) and not value.startswith("/") and not \
        value.endswith("/") and value.__contains__("//"):
        content_validation = True
    
    if value and len(value) <= character_length_map[mt_value][field_name]:
        length_validation = True
    
    return value, content_validation, length_validation

def handle_reference_to_preadvise(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    you will get the temporary documentary credit number 
    previously given to beneficiary before an actual documentary credit number is issued to an
    applicant.
    """      
    value = value.strip()
    content_validation = False
    length_validation = False
    
    if re.match(character_set_json["x"]["regex"], value) and \
        value.upper().__contains__("PREADV/"):
        content_validation = True
    
    if value and len(value) <= character_length_map[mt_value][field_name]:
        length_validation = True
    
    return value, content_validation, length_validation


def handle_date_of_issue(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    Date of Issue
    """
    content_validation = False
    length_validation = False
    
    # same for date of amendment as well
    if len(value) <= character_length_map[mt_value][field_name]:
        length_validation = True 
    
    # if month_validation(value) == True:
    #     content_validation = True
    
    return value, content_validation, length_validation

@pre_operation_decorator
def handle_applicable_rules(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    
    content_validation_applicable_rules = False
    content_validation_narrative = False
    
    
    # replace the extra spaces with single spaces
    value = re.sub(r'\s+', ' ', value.strip())
    
    for version in ["EUCP LATEST VERSION", "EUCPURR LATEST VERSION",\
        "UCP LATEST VERSION", "UCPURR LATEST VERSION"]:
        starting_index = value.upper().find(version)
        end_index = starting_index + len(version)      
        content_validation_applicable_rules = True
        content_validation_narrative = True
        applicable_rules = value[: end_index]
        narrative = value[end_index:]
        
        
        if re.match(character_set_json["x"]["regex"], applicable_rules):
            content_validation_applicable_rules = True
        
        if re.match(character_set_json["x"]["regex"], narrative):
            content_validation_narrative = True        
        
        if len(applicable_rules) <= character_length_map[mt_value][field_name][inner_fields[0]]:
            length_validation_applicable_rules = True
        
        if len(narrative) <= character_length_map[mt_value][field_name][inner_fields[1]]:
            length_validation_narrative = True
        
        return applicable_rules, narrative, \
            content_validation_applicable_rules, length_validation_applicable_rules, \
                content_validation_narrative, length_validation_narrative
            

@pre_operation_decorator
def handle_date_and_place_of_expiry(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    1) first 6 characters will always be date and remaining will be place
    2) extract country from the place
    3) check if the value is place or organisation name using the Ner model
    Returns
    """
    content_validation_date  = False
    length_validation_date = False
    content_validation_place  = False
    length_validation_place = False
    content_validation_country  = True
    length_validation_country = True
    
    # 1) first 6 characters will always be date and remaining will be place
    date = value[:6]
    place = value[6:]
    country = get_country_from_place(place)
    
    if re.match(character_set_json["n"]["regex"], date) and\
        re.match(character_set_json["x"]["regex"], place) and \
            month_validation(date):
                content_validation_date = True
                content_validation_place = True

    if len(date) <= character_length_map[mt_value][field_name][inner_fields[0]] and \
        len(place) <= character_length_map[mt_value][field_name][inner_fields[1]]:
        length_validation_place = True
        length_validation_date = True

    return date, place, country, content_validation_date, length_validation_date,\
                content_validation_place,length_validation_place, \
                    content_validation_country, length_validation_country


def handle_applicant_bank(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    applicant bank: 51a
    """
    content_validation = False
    length_validation = False
    party_identifier = ""
    identifier_code = ""
        
    if field_name == "51A":
        print(f"value: {value}")
        # assumption is that party identifier and identifier code is separated by newline
        party_identifier, identifier_code = value.split("\n")[0],\
            " ".join(value.split("\n")[1:])
        
        if len(party_identifier) <= character_length_map[mt_value][field_name][inner_fields[0]]:
            length_validation =  True
        
        if len(identifier_code) <= character_length_map[mt_value][field_name][inner_fields[1]]:
            length_validation = True

        return party_identifier, identifier_code, \
            content_validation, length_validation
    
    elif field_name == "51D":
        party_identifier, name_and_address = value.split("\n")[0], " ".join(value.split("\n")[1:])
        
        if len(party_identifier) <= character_length_map[mt_value][field_name][inner_fields[0]]:
            length_validation = True        
        
        if len(name_and_address) <= character_length_map[mt_value][field_name][inner_fields[1]]:
            length_validation = True
            
        name, address = find_company_and_address(value = name_and_address)
        
        return party_identifier, "~".join(name), "~".join(address), \
            content_validation, length_validation

@pre_operation_decorator
def handle_applicant(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    Field 50: Applicant
    """
    content_validation = False
    length_validation = False
    
    if len(value) <= character_length_map[mt_value][field_name]:
        length_validation = True

    if re.match(character_set_json["x"]["regex"], value):
        content_validation = True

    name, address = find_company_and_address(value=value)
    
    country = get_country_from_place(address)
    
    return "~".join(name), "~".join(address), "~".join(country),\
        content_validation, length_validation

@pre_operation_decorator
def handle_beneficiary(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    content_validation = True
    length_validation = True
    account_no = ""
    name_and_address = ""
    
    splitted_values = value.split("\n")
    
        
    if len(splitted_values) > 1:
        account_no, name_and_address = splitted_values[0], splitted_values[1]
    elif splitted_values:
        account_no, name_and_address = "", splitted_values[0]
    
    if re.match(character_set_json["x"]["regex"], account_no) and \
        re.match(character_set_json["x"]["regex"], name_and_address):
        content_validation = True
    
    if len(account_no) <= character_length_map[mt_value][field_name][inner_fields[0]] and\
        len(name_and_address) <= character_length_map[mt_value][field_name][inner_fields[1]]:
            length_validation = True
    
    name, address = find_company_and_address(value=value)
    country = get_country_from_place(address)
    
    return account_no, "~".join(name),\
        "~".join(address),"~".join(country), content_validation, length_validation
        
@pre_operation_decorator
def handle_currency_code_and_amount(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    1) extract currency code and amount separately
    2) replace , comma with .
    3) there is also the configuration which we need to put the permissible charcater after decimal
    depending on the currency code
    """
    content_validation_currency = False
    length_validation_currency = False
    content_validation_amount = False    
    length_validation_amount = False

    currency, amount = value[:3], value[3:]
    
    if re.match(character_set_json["a"]["regex"], currency):
        content_validation_currency = True
    
    if re.match(character_set_json["d"]["regex"], amount):
        content_validation_amount = True   
    
    if len(currency) <= character_length_map[mt_value][field_name][inner_fields[0]]:
        length_validation_currency = True
    
    
    if len(amount) <= character_length_map[mt_value][field_name][inner_fields[1]]:
        length_validation_amount = True
        
    return currency, amount, content_validation_currency, length_validation_currency,\
        content_validation_amount, length_validation_amount

@pre_operation_decorator
def handle_percentage_credit_amount_tolerance(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    
    content_validation_positive_percentage =  False
    length_validation_positive_percentage = False
    content_validation_negative_percentage =  False
    length_validation_negative_percentage = False
    
    print(f"value: {value}")
    # exit("+++++++++++++")
    positive_percentage_change, negative_percentage_change = value.split('/')
    
    
    if len(positive_percentage_change) <= character_length_map[mt_value][field_name]:
        length_validation_negative_percentage = True

    if len(negative_percentage_change) <= character_length_map[mt_value][field_name]:
        length_validation_positive_percentage = True
    
    positive_percentage_change, negative_percentage_change = int(positive_percentage_change), \
        int(negative_percentage_change)

    if  0 <= positive_percentage_change <= 99:
        content_validation_positive_percentage = True
    if  0 <= negative_percentage_change <= 99:
        content_validation_negative_percentage = True
    
    return positive_percentage_change, negative_percentage_change,\
                content_validation_negative_percentage, length_validation_negative_percentage,\
                    content_validation_positive_percentage, length_validation_positive_percentage

def handle_additional_amounts_covered(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    Field 39C: Additional Amounts Covered
    Narratibe fields: example should be covered
    """
    content_validation = False
    length_validation = False
    
    if re.match(character_set_json["x"]["regex"], value):
        content_validation = True

    if len(value) <= character_length_map[mt_value][field_name]:
        length_validation = True

    return value, content_validation, length_validation


@pre_operation_decorator
def handle_drafts_at(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    Field 42C: Drafts at ...
    This field specifies the tenor of drafts to be drawn under the documentary credit.
    Narrative
    60 DAYS/Months FROM/After (BL DATE)
    """
    content_validation = False
    length_validation = False
    
    # List of delimiters to split by
    delimiters_days = ['days', 'dys', "months"]
    
    # take the first value only
    tenor_in_days = split_by_list(value, delimiters_days)[0] if split_by_list(value, delimiters_days) else ""
    print(f"tenor in days : {tenor_in_days}")
    
    tenor_by_possible_values = ["from", "after"]
    pattern_tenor_by = '|'.join(map(re.escape, tenor_by_possible_values))
    
    # take the first value only  
    tenor_by = re.findall(pattern_tenor_by,value)[0] if re.findall(pattern_tenor_by,value) else ""
    
    print(f"tenor by : {tenor_by}")
    
    # reference date
    possible_reference_date = ["BL", "B/L", "shipment", "onboard", "dispatch", "invoice",\
        "inv", "sight"]
    pattern_reference_date = '|'.join(map(re.escape, possible_reference_date))
    reference_date = re.findall(pattern_reference_date, value)[0] if re.findall(pattern_reference_date, value) else ""
    
    if re.match(character_set_json["x"]["regex"], value):
        content_validation =  True
    
    if len(value) <= character_length_map[mt_value][field_name]:
        length_validation =  True

    return tenor_in_days, tenor_by, reference_date,\
        content_validation, length_validation


@pre_operation_decorator
def handle_available_with_by(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    # lowering the value of the string
    value = value.lower()
    
    content_validation_available_with = False
    content_validation_available_by = False
    length_validation_available_with = False
    length_validation_available_by = False    
    
    if field_name == "41A":
        available_with = value.split(" ")[0]
        
        if len(available_with) <= character_length_map[mt_value][field_name][inner_fields[0]]:
            content_validation_available_with = True
        
        first_four = available_with[:4]
        next_two = available_with[4:6]
        remaining = available_with[6:]
        if re.match(character_set_json["a"]["regex"], first_four) and \
            re.match(character_set_json["a"]["regex"], next_two) and \
                re.match(character_set_json["c"]["regex"],remaining):
            content_validation_available_with =  True
        
        available_by = " ".join(value.split(" ")[1:])
        
        if len(available_by) <= character_length_map[mt_value][field_name][inner_fields[1]]:
            length_validation_available_by = True
        
        if re.match(character_set_json["x"]["regex"], available_by):
            content_validation_available_by = True
    
    if field_name == "41D":
        available_with = "".join(value.split("by")[0])
        
        if len(available_with) <= character_length_map[mt_value][field_name][inner_fields[0]]:
            length_validation_available_with = True
        
        if re.match(character_set_json["x"]["regex"], value):
            content_validation_available_with =  True
        
        available_by = " ".join(value.split("by")[1])
        
        if len(available_by) <= character_length_map[mt_value][field_name][inner_fields[1]]:
            content_validation_available_by = True
        
        if re.match(character_set_json["x"]["regex"], available_by):
            content_validation_available_by = True
    
    return available_with, available_by,\
        content_validation_available_with, length_validation_available_with,\
            content_validation_available_by, length_validation_available_by
                    
    
#######################################################################################
def split_by_list(string, delimiters):
    pattern = '|'.join(map(re.escape, delimiters))
    parts = re.split(pattern, string)
    return [part for part in parts if part]


    
########################################################################################

def handle_drawee(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    Field 42a: Drawee
    """
    content_validation = False
    length_validation = False
        
    if field_name == "42A":
        
        # assumption is that party identifier and identifier code is separated by newline
        party_identifier, identifier_code = "",\
            value
        
        if len(party_identifier) <= character_length_map[mt_value][field_name][inner_fields[0]]:
            length_validation =  True
        
        if len(identifier_code) <= character_length_map[mt_value][field_name][inner_fields[1]]:
            length_validation = True

        return party_identifier, identifier_code, content_validation, length_validation
    
    
    elif field_name == "42D":
        party_identifier, name_and_address = "", value
        
        if len(party_identifier) <= character_length_map[mt_value][field_name][inner_fields[0]]:
            length_validation = True        
        
        if len(name_and_address) <= character_length_map[mt_value][field_name][inner_fields[1]]:
            length_validation = True
        
        return party_identifier, name_and_address, content_validation, length_validation
################################################################################################
def get_extra_info_mixed_payment(condition, gen_ai: bool = False):
    if gen_ai:
        """
        Not implemented yet
        Returns
        """
        pass
        
    percent_pattern: str = r'\d+' 
    """
    example tested
    text1 = 'abc123pct'
    >>> re.findall(pattern, text1)
        ['123pct']

    text2 = 'xyz45pct'
    >>> re.findall(pattern, text2)
    ['45pct']

    text3 = '12pct'
    >>> re.findall(pattern, text3)
    ['12pct']        
    """
    # taking the first element
    payment_percent = condition.lower().split("pct")[0]
    payment_percent = re.findall(percent_pattern, payment_percent)
    print(f"payment percent: {payment_percent}")    
    
    
    tenor_pattern = r'from|FROM|After'  
    """
    >>> text4 = "60 days from B/L date" 
    >>> re.findall(pattern, text4)
    ['from']
    >>> text5 =  "50 days after inv date"
    >>> pattern = r'from|FROM|After|after|AFTER'  
    >>> re.findall(pattern, text5)
    ['after']    
    """
    tenor_type = re.findall(tenor_pattern, condition.lower())
    tenor_in_days = 0
    if "sight" not in condition:
        # List of delimiters to split by
        delimiters_days = ['days', 'DYS']
        # take the first value only
        tenor_in_days = split_by_list(condition.strip().lower(), delimiters_days)[0]
        print(f"tenor in days: {tenor_in_days}")
        tenor_in_days = tenor_in_days[-4:]
        tenor_in_days = re.findall(percent_pattern, tenor_in_days)

    print(f"tenor in days : {tenor_in_days}")
    return payment_percent, tenor_type, tenor_in_days

def handle_mixed_payment_details(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    
    
    content_validation = False
    length_validation = False
    
    if len(value) <= character_length_map[mt_value][field_name]:
        length_validation = True
    
    if re.match(character_set_json["x"]["regex"], value):
            content_validation = True
    
    # first check if index is there or not
    # you can check with \d+)\\s or \d+\s regex 
    # if index is there like that handle indexes else 
    # check first character is any special character
    index_flag = False
    pattern = [re.compile(r'\d+\)\s'), re.compile(r'\d+\s')]
    
    for p in pattern:
        matches =  p.findall(value)

        if matches:
            index_flag = True
        break

    condition_separator = matches
    
    print("condition separator")
    print(condition_separator)
    
    print(f"index flag: {index_flag}")

    if not index_flag:
        value = value.strip()
        # first character after stripping all white space
        condition_separator = value.strip()[0]
        print(f"condition separator : {condition_separator}")
        
        if is_special_character(condition_separator):
            payment_conditions = value.split(condition_separator)
            
            # clean payment conditions
            payment_conditions = [value for value in payment_conditions if len(value) != 0]
            
            print("payment conditions")
            print(payment_conditions)
            payment_percent_list = []
            tenor_type_list = []
            days_list = []
            for condition in payment_conditions:
                print(f"condition will be :{condition}")
                payment_percent, tenor_type, day = get_extra_info_mixed_payment(condition, gen_ai=False)        
                payment_percent_list.append(payment_percent)
                tenor_type_list.append(tenor_type)
                days_list.append(day)
        else:
            payment_percent_list = []
            tenor_type_list = []
            days_list = []
    else:
        payment_percent_list = []
        tenor_type_list = []
        days_list = []
        for separator in condition_separator:
            condition = value.split(separator)[0]
            payment_percent, tenor_type, day = get_extra_info_mixed_payment(condition, gen_ai=False)        
            payment_percent_list.append(payment_percent)
            tenor_type_list.append(tenor_type)
            days_list.append(day)
    
    print("Final Result")
    print({"payment_percent": payment_percent_list,
             "tenor_type": tenor_type_list,
             "days": days_list})
    
    return  {"payment_percent": payment_percent_list,
             "tenor_type": tenor_type_list,
             "days": days_list}, content_validation, length_validation


def handle_negotiation_and_deferred_payment_details():
    """
    Field 42P: Negotiation/Deferred Payment Details
    30 DAYS FROM THE DATE OF SHIPMENT
    Currently handling just like drafts at
    In future may implement new function
    """
    pass

@pre_operation_decorator
def handle_partial_shipments(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    Field 43P: Partial Shipments
    """
    content_validation = False
    length_validation = False
    
    if re.match(character_set_json["x"]["regex"], value) and value.upper() in \
        ["ALLOWED", "CONDITIONAL", "NOT ALLOWED"]:
        content_validation = True

    if len(value) <= character_length_map[mt_value][field_name]:
        length_validation = True
    
    return value, content_validation, length_validation

@pre_operation_decorator
def handle_transhipments(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    Field 43T: Transhipment
    """
    content_validation = False
    length_validation = False
    
    if re.match(character_set_json["x"]["regex"], value) and value.upper() in \
        ["ALLOWED", "CONDITIONAL", "NOT ALLOWED"]:
        content_validation = True

    if len(value) <= character_length_map[mt_value][field_name]:
        length_validation = True
    
    return value, content_validation, length_validation

#######################################################################################3
def extract_country_and_place(text):
    # Load the spaCy English model with NER capabilities
    nlp = spacy.load('en_core_web_sm')

    # Process the input text using spaCy
    doc = nlp(text)

    # Extract country and place entities
    countries = []
    places = []

    for ent in doc.ents:
        if ent.label_ == 'GPE':  # 'GPE' represents geopolitical entities (countries, cities, etc.)
            if ent.text not in countries:
                countries.append(ent.text)
        elif ent.label_ == 'LOC':
            if ent.text not in places:
                places.append(ent.text)

    return countries, places

def handle_place_of_taking_in_charge_dispatch_port(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    Field 44A: Place of Taking in Charge/Dispatch from .../Place of Receipt
    1) either you will get anywhere in country_name
    2) direct name country name
    """
    content_validation = False
    length_validation = False
    
    if re.match(character_set_json["z"]["regex"], value):
        content_validation = True
    
    if len(value) <= character_length_map[mt_value][field_name]:
        length_validation = True
     
    country, place =  extract_country_and_place(value)
    return country, place, content_validation, length_validation
    
#########################################################################################    

def handle_port_of_loading_airport_of_departure(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    content_validation = False
    length_validation = False
    
    if re.match(character_set_json["z"]["regex"], value):
        content_validation = True
    
    if len(value) <= character_length_map[mt_value][field_name]:
        length_validation = True
    
    country_name, port_sea_name = extract_country_and_place(value) 
    
    return country_name, port_sea_name, content_validation, length_validation    
    

def handle_port_of_discharge_airport_of_destination(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    content_validation = False
    length_validation = False
    
    if re.match(character_set_json["z"]["regex"], value):
        content_validation = True
    
    if len(value) <= character_length_map[mt_value][field_name]:
        length_validation = True
    
    country_name, port_airport_name = extract_country_and_place(value) 
    
    return country_name, port_airport_name, content_validation, length_validation    

def handle_place_of_final_destination_place_of_delibery_tranportation_to(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    content_validation = False
    length_validation = False
    
    if re.match(character_set_json["z"]["regex"], value):
        content_validation = True
    
    if len(value) <= character_length_map[mt_value][field_name]:
        length_validation = True
    
    country_name, place_of_final_destination = extract_country_and_place(value) 
    
    return country_name, place_of_final_destination, content_validation, length_validation


############################################################################################
def is_valid_date_format(date_string):
    pattern = r'\b\d{2}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])\b'
    return bool(re.match(pattern, date_string))

def handle_latest_date_of_shipment(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    content_validation = False
    length_validation = False
    
    if re.match(character_set_json["n"]["regex"],value) and is_valid_date_format(value):
        content_validation = True
    
    if len(value) <= character_length_map[mt_value][field_name]:
        length_validation = True    
    
    latest_date_of_shipment =  value
    return latest_date_of_shipment, content_validation, length_validation
############################################################################################

@pre_operation_decorator
def handle_shipment_period(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    Field 44D: Shipment Period
    """
    content_validation = False
    length_validation = False
    
    if re.match(character_set_json["x"]["regex"], value) and isinstance(value,int):
        content_validation = True
    
    if len(value) <= character_length_map[mt_value][field_name]:
        length_validation = True
    
    shipment_period = value
    return shipment_period, content_validation, length_validation


def check_if_goods_are_bulky_or_not_and_add_for_each_good(goods_description):
    """
        {
    "result": {
        "products": [
        {
            "name": "Coated Float Glass",
            "quantity": 11,
            "units": "KGs",
            "purchaseOrderNumber": "NOK 169",
            "date": "06-10-2023",
            "incoterm": "EX WORKS"
        },
        {
            "name": "Black Coated Float Glass",
            "quantity": 14,
            "units": "KGs",
            "purchaseOrderNumber": "NOK 169",
            "date": "06-10-2023",
            "incoterm": "EX WORKS"
        }
        ]
    }
    }
    """
    RESULT_KEY = "result"
    
    goods_data = goods_description[RESULT_KEY]   
    
    for good, item in goods_data.items():
        units = good["units"]
        
        if units.lower() in ["kgs","kilo grams", "kilo litres", "litres","ltrs", "metrics tonnes","MT", "MTs", "quintel", "metres", "MTR", "MTRs"]:
            good["is_bulky"] = True

    return goods_data

def handle_goods_descriptions(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    Field 45A: Description of Goods and/or Services
    Not implemented
    Usage Rules
    ===========
    1) Terms such as FOB, CIF, etc. should be specified in this field.
    2) The specification of each new item should begin on a new line, preceded by the sign \
        '+' or numbered using +1), +2), etc.

    Units to handle:
    ["CV", ""]
    
    {
        "shipment_1":{"G1":{"quantity_of_each_unit": "", 
        "is_approx_given"},
        "G2": {},
        "incoterms" :"",
        "incoterms_place_at": "",
        "shipment_date": "",
        "HScode": ""
        }
    }
    
    delimiter between goods: [hscode, ]
    """
    # openai.api_key = "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"
    # prompt = f'''given the goods description "{value}". Give the product name, quantity, \
    #     units, purchase order number it belongs to, date , invoice number , \
    #         date and incoterm involved? Generate a result in json form?'''            
    # print("Promt is : ",prompt)
    # response = openai.Completion.create(engine="text-davinci-003", prompt=prompt,max_tokens=100)
    # print(response)
    # # exit("+++++++++++++")
    # # Extract the generated response
    # generated_response = response['choices'][0]['text']
    # return generated_response.strip("\n").strip()
    return [value]
    
def handle_documents_required(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    Field 45A: Description of Goods and/or Services
    Not implemented
    1) When the ultimate date of issue of a transport document is specified, it is to be \
        specified with the relative document in this field.
    2) For credits subject to eUCP, the format in which electronic records are to be \
        presented must be specified in this field.
    3) The specification of each new item should begin on a new line, preceded by the sign \
        '+' or numbered using +1), +2), etc.
    """    
    # pass
    return [value]

def handle_additional_conditions(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    Usage rules
    ============
    1) If presentation of both electronic records and paper documents is allowed,\
       the place for presentation of the electronic records (that is, the electronic \
       address to which presentation must be made) as well as the place for presentation \
       of the paper documents must be specified in this field.
    2) If presentation of only electronic records is allowed, the place for presentation of \
       the electronic records (that is, the electronic address to which presentation must be made) \
       must be specified in this field.
    3) If not already part of the original documentary credit, the advising bank,\
       that is, the receiver of the message, must provide the beneficiary or another \
       advising bank with the electronic address of the issuing bank. Furthermore, \
       the advising bank must provide the beneficiary or another advising bank with \
       the electronic address where they wish the electronic records to be presented.
    """
    # pass
    return [value]


def handle_special_payments_conditions_for_beneficiary(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    # document why this method is empty
    """
    Not implemented yet
    """
    return [value]


def handle_special_payment_conditions_for_bank_only(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    # document why this method is empty
    """
    Not implemented yet
    """
    print(value)
    # exit("+++++++++++")
    return [value]

###################################################################################
def extract_numeric_digits(s):
    match = re.search(r'\b\d{1,13}\b', s)
    if match:
        return match.group()
    else:
        return None


def get_code_currency_amount_from_charges(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    print(f"value is : {value}")
    code_list : list = ["AGENT", "COMM", "CORCOM", \
        "DISC", "INSUR", "POST", "STAMP", "TELECHAR", "WAREHOUS"] 
    
    code, currency, amount, narrative = "", "" , "", ""    
    for code in code_list:
        index = value.find(code)
        if index != -1:
            code = value[:index + len(code)]
        currency = value[index + len(code): index + len(code) + 3]
        amount =  extract_numeric_digits(value[index + len(code) + 3:])

        narrative = value[value.find(amount) + len(amount):] 
    
    return code, currency, amount, narrative

def handle_charges_normal(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    return [value]


def handle_charges(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    1) Code, currency, amount, narrative
    2) In the absence of this field, all charges, except negotiation and \
        transfer charges, are to be borne by the applicant.
    3) Any code used in this field must be between slashes and must  \
        appear at the beginning of a line.
    4) Narrative text must not begin with a slash and, if used, must begin \
        on a new line and be the last information in the field.
    * This field may be used only to specify charges to be borne by the beneficiary.
    """
    content_validation_code = False
    length_validation_code = False
    
    content_validation_currency = False    
    length_validatio_currency = False
    
    content_validation_amount = False
    length_validation_amount = False
    
    content_validation_narrative = False
    length_validation_narrative = False
    
    code, currency, amount, narrative = get_code_currency_amount_from_charges(value)
    
    
    if re.match(character_set_json["a"]["regex"], code):
        content_validation_code = True
    
    if re.match(character_set_json["a"]["regex"],currency):
        content_validation_currency = True 
    
    if re.match(character_set_json["d"]["regex"], amount):
        content_validation_amount = True   
    
    if re.match(character_set_json["x"]["regex"],narrative):
        content_validation_narrative = True
    
    
    if len(code) <= character_length_map[mt_value][field_name]["code"]:
        length_validation_code = True
    
    if len(code) <= character_length_map[mt_value][field_name]["code"]:
        length_validatio_currency = True
    
    if len(code) <= character_length_map[mt_value][field_name]["code"]:
        length_validation_amount = True    

    if len(code) <= character_length_map[mt_value][field_name]["code"]:
        length_validation_narrative = True        
    
    return code, currency, amount,narrative, content_validation_code, length_validation_code,\
        content_validation_currency, length_validatio_currency\
            , content_validation_amount, length_validation_amount, \
                content_validation_narrative, length_validation_narrative
    
####################################################################################
def extract_numeric_digits(s):
    match = re.search(r'\b\d{1,3}\b', s)
    if match:
        return match.group()
    else:
        return None

def handle_period_of_presentation_in_days(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    Field 48: Period for Presentation in Days
    3n[/35x]
    (Days)(Narrative)
    This field specifies the number of calendar days after the date of shipment within \
    which the documents must be presented for payment, acceptance, or negotiation. Narrative \
    must only be used to specify another type of date than a shipment date, for example invoice \
    date, from which the period for presentation begins.
    """
    days = extract_numeric_digits(value)    
    index = value.find(days)
    presentation_in_days = value[:index + len(days)]
    narrative = value[index+len(days):]
    
    content_validation_presentation_in_days = False
    length_validation_presentation_in_days = False
    
    content_validation_narrative = False
    length_validation_narrative = False    
    
    if re.match(character_set_json["n"]["regex"], presentation_in_days):
        content_validation_presentation_in_days = True
    
    if re.match(character_set_json["x"]["regex"], narrative):
        content_validation_narrative = True
    
    if len(presentation_in_days) <= character_length_map[mt_value][field_name]["days"]:
        length_validation_presentation_in_days = True
    
    if len(narrative) <= character_length_map[mt_value][field_name]["narrative"]:
        length_validation_narrative = True
            
    return presentation_in_days, narrative, \
        content_validation_presentation_in_days, length_validation_presentation_in_days,\
        content_validation_narrative, length_validation_narrative    
    
def handle_confirmation_instructions(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    Field 49: Confirmation Instructions
    """
    content_validation = False
    length_validation = False
    
    
    if re.match(character_set_json["x"]["regex"]) and value in ["CONFIRM", "MAY ADD", "WITHOUT"]:
        content_validation = True
    
    if len(value) <= character_length_map[mt_value][field_name]:
        length_validation = True
    
    return value, content_validation, length_validation   

def handle_requested_confirmation_party(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    Field 42a: Drawee
    """
    content_validation = False
    length_validation = False
        
    if field_name == "58A":
        
        # assumption is that party identifier and identifier code is separated by newline
        party_identifier, identifier_code = value.split("\n")[0],\
            " ".join([value.split("\n")[1:]])
        
        if len(party_identifier) <= character_length_map[mt_value][field_name]["party_identifier"]:
            length_validation =  True
        
        if len(identifier_code) <= 11:
            length_validation = True

        return party_identifier, identifier_code, content_validation, length_validation 
    
    elif field_name == "58D":
        party_identifier, name_and_address = value.split("\n")[0], " ".join([value.split("\n")[1:]])
        
        if len(party_identifier) <= character_length_map[mt_value][field_name]:
            length_validation = True        
        
        if len(name_and_address) <= character_length_map[mt_value][field_name]["name_and_address"]:
            length_validation = True
        
        return party_identifier, name_and_address, content_validation, length_validation


def handle_requested_confirmation_normal(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    Field 42a: Drawee
    """
    content_validation = False
    length_validation = False
        
    if field_name == "58A":
        
        # assumption is that party identifier and identifier code is separated by newline
        # party_identifier, identifier_code = value.split("\n")[0],\
        #     " ".join([value.split("\n")[1:]])
        
        # if len(party_identifier) <= character_length_map[mt_value][field_name]["party_identifier"]:
        #     length_validation =  True
        
        # if len(identifier_code) <= 11:
        #     length_validation = True

        # return party_identifier, identifier_code, content_validation, length_validation 
        return [value]
    elif field_name == "58D":
        # party_identifier, name_and_address = value.split("\n")[0], " ".join([value.split("\n")[1:]])
        
        # if len(party_identifier) <= character_length_map[mt_value][field_name]:
        #     length_validation = True        
        
        # if len(name_and_address) <= character_length_map[mt_value][field_name]["name_and_address"]:
        #     length_validation = True
        
        # return party_identifier, name_and_address, content_validation, length_validation
        return [value]

def handle_reimbursing_bank(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    # Not implemented
    return [value]

def handle_instructions_to_paying_accepting_negotiating_bank(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    # Not implemented
    return [value]

def handle_advise_through_bank(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    # Not implemented
    return [value]

def handle_sender_to_receiver_information(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    # Not implemented
    return [value]

############################ Finishing of the MT700############################
@pre_operation_decorator
def handle_receivers_reference(value: str,
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    Field 21: Receiver's Reference
    Usage Rules:
    ============
    If the Receiver's reference is not known, NONREF must be used in this field.    
    """
    value = value.strip()
    content_validation = False
    length_validation = False
    
    if re.match(character_set_json["x"]["regex"], value) and not value.startswith("/") and not \
        value.endswith("/") and value.__contains__("//"):
        content_validation = True
    
    if value and len(value) <=16:
        length_validation = True
    
    return value, content_validation, length_validation


@pre_operation_decorator
def handle_number_of_amendment(value, mt_value, field_name, character_set):
    content_validation = False
    length_validation = False    
    
    if re.match(character_set_json["n"]["regex"], value):
        content_validation = True    
    if len(value) <= character_length_map[mt_value][field_name]:
        length_validation = True
    
    assert bool(re.match(character_length_map[character_set], value)) == True
    
    return int(value), content_validation, length_validation

def month_validation(value):
    return 1 <= int("".join(value[2:4])) <= 12 

@pre_operation_decorator
def handle_date_amendment(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    Field 30: Date of Amendments
    """
    content_validation = False
    length_validation = False
    
    # same for date of amendment as well
    if len(value) <= character_length_map[mt_value][field_name]:
        length_validation = True 
    
    # if month_validation(value) == True:
    #     content_validation = True
    
    return value, content_validation, length_validation

def handling_issuing_bank(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    if field_name == "52A":
        party_identifier, identifier_code = value.split("\n")[0], " ".join([value.split("\n")[1:]])
        assert len(party_identifier) <= character_length_map[mt_value][field_name]
        return party_identifier, identifier_code
    elif field_name == "52D":
        party_identifier, name_and_address = value.split("\n")[0], " ".join([value.split("\n")[1:]])
        assert len(party_identifier) <= character_length_map[mt_value][field_name]        
        return party_identifier, name_and_address


def non_bank_issuer_of_the_credit(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    
    assert len(value) <= character_length_map[mt_value][field_name]    
    
    
    
    return [value]

def purpose_of_the_message(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    # same for form of documentary credit as well
    return [value]

def handle_cancellation_request(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    # Need to implement
    
    content_validation = False
    length_validation = False
    
    # same for date of amendment as well
    if len(value) <= character_length_map[mt_value][field_name]:
        length_validation = True 
    
    if month_validation(value) == True:
        content_validation = True
    
    return value, content_validation, length_validation


def increase_and_decrease_of_documentary_credit_amount(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    """
    1) extracting the increase or decrease documentary credit amount
    2) replace , comma with .
    3) there is also the configuration which we need to put the permissible charcater after decimal
    depending on the currency code
        * we need to maintain one map where the 
    """
    return [value]

def handle_amendment_charges_payable_by(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):
    # Need to implement
    # content_validation = False
    # length_validation = False
    
    # same for date of amendment as well
    # if len(value) <= character_length_map[mt_value][field_name]:
        # length_validation = True 
    
    # if month_validation(value) == True:
        # content_validation = True
    
    # return value, content_validation, length_validation
    return [value]

def handling_issuing_bank(value: str, 
                            mt_value: str = None, 
                            field_name: str = None,
                            inner_fields: Tuple = None):

    print(f"value: {value}")
    return [value]

########################### End of MT 707####################################








