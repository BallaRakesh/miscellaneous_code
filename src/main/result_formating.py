import json

# Predefined hints for identifying columns
hints = {
    "goods_description": ["description"],
    "quantity" : ['quantity'],
    "unit_price": ["rate", "price"],
    "amount": ["amount", "total"]
}

def contains_token(word_list, token):
    for word in word_list:
        if word in token:
            return True
    return False


def find_column_indices(headers, hints):
    # Initialize a dictionary to store the column indices for each key
    column_indices = {
        "goods_description": -1,
        "quantity": -1,
        "unit_price": -1,
        "amount": -1
    }
    
    # Iterate over the headers to find the indices based on hints 
    for header in headers:
        header_text = header['content'].lower()
        for key, hint_list in hints.items():
            if contains_token(hint_list, header_text):
                column_indices[key] = header['pos']
                break
    return column_indices

# Function to parse the input JSON and find the required column indices
def parse_json_and_find_columns(json_input):
    headers = json_input.get('headers', [])
    column_indices = find_column_indices(headers, hints)
    return column_indices

# Example JSON input (provided in the question)

def extract_goods_info(json_input, column_indices):
    cells = json_input['cells']
    goods_info = {}
    current_row = -1

    for cell in cells:
        row = cell['row_nums'][0]
        if row != current_row:
            # If this cell starts a new row, create a new goods entry
            current_row = row
            goods_key = f"goods{current_row + 1}"
            goods_info[goods_key] = {
                'goods_description': '',
                'quantity': '',
                'unit_price': '',
                'amount': ''
            }
        
        # Fill in the details based on the column number
        for key, col_num in column_indices.items():
            if col_num in cell['column_nums']:
                goods_info[goods_key][key] = cell['content']

    return goods_info

def aggregate_goods_info(goods_info):
    aggregated_info = {}
    for key, goods in goods_info.items():
        description = goods['goods_description']
        if description not in aggregated_info:
            aggregated_info[description] = {
                'quantity': 0,
                'amount': 0
            }

        # Add quantity if it is a number
        if goods['quantity'].replace('.', '', 1).isdigit():
            aggregated_info[description]['quantity'] += float(goods['quantity'])
        
        # Add amount if it is a number
        if goods['amount'].replace('.', '', 1).isdigit():
            aggregated_info[description]['amount'] += float(goods['amount'])

    # Convert aggregated_info back to goods_info format
    final_goods_info = {}
    counter = 1
    for description, data in aggregated_info.items():
        final_goods_info[f'goods{counter}'] = {
            'goods_description': description,
            'quantity': str(data['quantity']),
            'unit_price': '',
            'amount': str(data['amount'])
        }
        counter += 1

    return final_goods_info



#############################################################
#############################################################
# Example JSON input (provided in the question)
json_input = {
    "table_coords": [104, 898, 1594, 1281],
    "headers": [
        {"pos": 0, "content": "marks & nos", "bbox": [0, 0, 467, 27]},
        {"pos": 1, "content": "no & kind of pkgs description", "bbox": [467, 0, 838, 27]},
        {"pos": 2, "content": "description", "bbox": [838, 0, 1067, 27]},
        {"pos": 3, "content": "quantity", "bbox": [1067, 0, 1211, 27]},
        {"pos": 4, "content": "rate", "bbox": [1211, 0, 1316, 27]},
        {"pos": 5, "content": "amount", "bbox": [1316, 0, 1489, 27]}
    ],
    "cells": [
        {"content": "cold dough premix fdpz3004", "bbox": [0, 86, 467, 116], "column_nums": [0], "row_nums": [0]},
        {"content": "11 ctn cold dough premix", "bbox": [467, 86, 838, 116], "column_nums": [1], "row_nums": [0]},
        {"content": "", "bbox": [838, 86, 1067, 116], "column_nums": [2], "row_nums": [0]},
        {"content": "11.0000", "bbox": [1067, 86, 1211, 116], "column_nums": [3], "row_nums": [0]},
        {"content": "cif 38.324", "bbox": [1211, 86, 1316, 116], "column_nums": [4], "row_nums": [0]},
        {"content": "421.560", "bbox": [1316, 86, 1489, 116], "column_nums": [5], "row_nums": [0]},
        {"content": "cold dough premix fdpz3004 net wt : 38.75 kg batch no sb121288 dom : 22 august 2011 bbf 21 december 2011", "bbox": [0, 86, 467, 383], "column_nums": [0], "row_nums": [1]},
        {"content": "11 ctn cold dough premix packed in 775 g x 50 pkt x 1 sack 11 x 38.75 426.2500", "bbox": [467, 86, 838, 383], "column_nums": [1], "row_nums": [1]},
        {"content": "woven 426.2500 kgs", "bbox": [838, 86, 1067, 383], "column_nums": [2], "row_nums": [1]},
        {"content": "11.0000 ctn", "bbox": [1067, 86, 1211, 383], "column_nums": [3], "row_nums": [1]},
        {"content": "cif 38.324 per ctn", "bbox": [1211, 86, 1316, 383], "column_nums": [4], "row_nums": [1]},
        {"content": "421.560", "bbox": [1316, 86, 1489, 383], "column_nums": [5], "row_nums": [1]}
    ]
}

with open('/datadrive/table_res_trans/miscellaneous_code/src/samples/Covering_Schedule_254_page_5.json', 'r') as exp:
    json_input = json.load(exp)


# print(json_input)
# Find the column indices for "goods description", "unit price", and "amount"
column_indices = parse_json_and_find_columns(json_input)


# Column indices obtained earlier
# column_indices = {'goods_description': 2, 'quantity': 3, 'unit_price': 4, 'amount': 5}

# Extract goods information
print('column_indices', column_indices)
goods_info = extract_goods_info(json_input, column_indices)
print("Extracted Goods Info:")
print(goods_info)

# Aggregate goods information
aggregated_goods_result = aggregate_goods_info(goods_info)
print("Aggregated Goods Info:")
print(aggregated_goods_result)



