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

with open('/datadrive/table_res_trans/miscellaneous_code/src/samples/Covering_Schedule_254_page_5.json', 'r') as exp:
    json_input = json.load(exp)


print(json_input)
# Find the column indices for "goods description", "unit price", and "amount"
column_indices = parse_json_and_find_columns(json_input)


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


# Column indices obtained earlier
# column_indices = {'goods_description': 2, 'quantity': 3, 'unit_price': 4, 'amount': 5}

# Extract goods information
goods_info = extract_goods_info(json_input, column_indices)
print(goods_info)


