import pandas as pd
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

conn = sqlite3.connect('/home/ntlpt19/Desktop/TF_release/TradeGPT/DBs/14augPandas_testing.db')
cursor = conn.cursor()
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')

def prompt_template_relevant_tables3_ext(query: str, all_schemas):
    
    return f"""
    Based on the following list of table schemas: 
    {all_schemas}

    Identify all the relevant tables as per the users query. 
    User query: {query}

    The key of the returned response must be "relevant_tables". 
    Return only the table names in a list.
    
    If there are any null values in the response, leave the value empty.
    If there is no document type mentioned in the user query, leave the value empty.
    Example document type:CS: Covering Schedule, AWB: AirWay Bill, BOE: Bill of exchange, CI: Commerical Invoice, COO: Certificate of Origin \n
    IC: Insurance Certificate, PI: Performa Invoice, PL: Packing List, PO: Purchase order, LC:Letter of Credit
    
    I also require hints in the query, such as `image_name`, `page_no`, and `work_item_number` from the user's query. Note: Thoroughly analyze the user's query and return these parameters in a dictionary format. If any of the values are not available, leave them empty. 
    Note: 
    `image_name`: The name of the image, sometimes referred to as the document name and some times ends with .png. example: document name pdf11_7.png
    `page_no`: This is mentioned as "page number" in the query.
    `work_item_number`: This is mentioned as "work item number" in the query. 
    Response:
    """

def prompt_template_relevant_tables3_ext1(query: str, all_schemas):
    return f"""
    You are expert in identifing the revent tables in the domain of Trade Finance
    Based on the following list of table schemas: 
    {all_schemas}

    Identify all the relevant tables as per the users query. 
    User query: {query}

    The task is to identify the relevant tables based on the document type mentioned in the query and extract the fields to extract mentioned in the query.  

    Note:  
    1. Extract the document type mentioned in the query. analysis the query, what is the document type mentioned   
    2. Assume the document type is the relevant table.  
    3. Extract the fields mentioned for extraction.  

    The response must contain two keys:  
    - "relevant_tables"  
    - "fields_to_extract"  

    The value for "relevant_tables" is a list of relevant table names.  
    The value for "fields_to_extract" is a list of fields to extract mentioned in the query.  

    Example:  
    Query: "Identify the insurance certificate number in the Covering Schedule document named pdf16_7.png."  
    Relevant tables: ["CS"]  
    Fields to extract: ["insurance certificate number"]  
  
    If there are any null values in the response, leave the value empty.
    If there is no document type mentioned in the user query, leave the value empty.
    Example document type:CS: Covering Schedule, AWB: AirWay Bill, BOE: Bill of exchange, CI: Commerical Invoice, COO: Certificate of Origin \n
    IC: Insurance Certificate, PI: Performa Invoice, PL: Packing List, PO: Purchase order, LC:Letter of Credit
    
    I also require hints in the query, such as `image_name`, `page_no`, and `work_item_number` from the user's query. Note: Thoroughly analyze the user's query and return these parameters in a dictionary format. If any of the values are not available, leave them empty. 
    Note: 
    `image_name`: The name of the image, sometimes referred to as the document name and some times ends with .png. example: document name pdf11_7.png
    `page_no`: This is mentioned as "page number" in the query.
    `work_item_number`: This is mentioned as "work item number" in the query. 
    Response:
    """


def prompt_template_classification_relevant_tables2(query: str, all_schemas):
    return f"""    
    Based on the following list of table schemas: 
    {all_schemas}

    Identify top one relevant tables as per the users query. 
    User query: {query}
    
    The key of the returned response must be "relevant_tables" in a proper JSON format. 
    Return only the table names in a list.
    If there are any null values in the response, leave the value empty.
    
    To identify the relevant tables, I will follow these steps:
    1. Review the list of provided table schemas.
    2. Exclude any OCR-related database tables.
    3. If the query is related to classification, prioritize adding classification-related table schemas.
    4. Only select tables that are present in the provided list of schemas.
    example:
    "relevant_tables": ["BOL", "CS", "classification_table"]

    I also require hints in the query, such as `image_name`, `page_no`, and `work_item_number` from the user's query. Note: Thoroughly analyze the user's query and return these parameters in a dictionary format. If any of the values are not available, leave them empty. 
    Note: 
    `image_name`: The name of the image, sometimes referred to as the document name and some times ends with .png. example: document name pdf11_7.png
    `page_no`: This is mentioned as "page number" in the query.
    `work_item_number`: This is mentioned as "work item number" in the query. 
    
    the response should be in the proper json format and include the main responses as shown below:
        {{
            "relevant_tables" : [<list of table schemas>],
            "hints" : {{
                "image_name" : "<image_name>",
                "page_no" : "<page_no>",
                "work_item_number" : "<work_item_number>"
                }},
        }}
    Response:
    """

def tables_schema():
    # Execute a query to retrieve the list of tables  
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")  

    # Fetch the results  
    tables = cursor.fetchall()  

    # Print the table names  
    print("Tables in the database:")  
    table_list = []
    for table in tables:  
        print(table[0])  
        table_list.append(table[0])

    # Close the connection  
    # conn.close()
    print(table_list)
    return table_list



def fetch_table_schema(table_name):
    
    cursor.execute(f"PRAGMA table_info({table_name});")
    
    # self.cursor.execute(f"""SELECT column_name, data_type 
    #                     FROM information_schema.columns 
    #                     WHERE table_name = '{table_name}' 
    #                     AND table_schema = 'public';
    #                     """)
    res = cursor.fetchall()
    schema_string = f"The table {table_name} has the following columns and data types:\n"
    schema_string += '\n'.join([f"Column Name: {item[0]}, Data Type: {item[1]}" for item in res])

    return schema_string   

def get_all_schemas():
    # self.cursor.execute(f"""SELECT table_name, table_schema 
    #                             FROM information_schema.tables 
    #                             WHERE table_type = 'BASE TABLE' 
    #                             AND table_schema NOT IN ('pg_catalog', 'information_schema');
    #                             """)
    
    cursor.execute(f"""SELECT name AS table_name, 
                            sql AS table_schema 
                            FROM sqlite_master 
                            WHERE type = 'table';
                            """)
    
    
    res = cursor.fetchall()
    schemas = []
    for table in res:
        schema_string = fetch_table_schema(table[0])
        schemas.append([schema_string])

    return schemas

def generate_queries(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Check if the 'image_name' column exists in the CSV
    if 'image_name' not in df.columns:
        print("Error: 'image_name' column not found in the CSV file.")
        return
    
    # Iterate over the 'image_name' column and generate queries
    queries = []
    for image_name in df['image_name']:
        # query = f"The document named {image_name} seems to be misclassified. Can you confirm its document class?"
        query = f"The document named {image_name}. Can you confirm its document class?"
        queries.append(query)
    
    return queries


def get_relevant_tables(method_name, query):
    all_schemas = get_all_schemas()
    all_tables = tables_schema()
    print('all_tables')

    template_for_data_identification = eval(method_name)(query, all_schemas)
    data_response = llm.invoke(template_for_data_identification)
    return data_response
    


abbreviations = {
    'CS': 'Covering Schedule',
    'AWB': 'AirWay Bill',
    'BOE': 'Bill of Exchange',
    'CI': 'Commercial Invoice',
    'COO': 'Certificate of Origin',
    'IC': 'Insurance Certificate',
    'PL': 'Packing List'
}


def generate_extraction_queries(folder_path):
    queries = []
    # Iterate over each CSV file in the folder
    for filename, obb in abbreviations.items():
        # Construct the full file path
        file_path = os.path.join(folder_path, filename+'.csv')
        print(file_path)
        # Read the CSV file
        df = pd.read_csv(file_path)
        # Check if the 'image_name' column exists in the CSV
        if f'{filename}_document_name' not in df.columns:
            print(f"Error: 'image_name' column not found in {filename}.")
            continue
        # Get all columns except 'image_name' to use as keys
        possible_keys = [col for col in df.columns if col != f'{filename}_document_name']
        print(possible_keys)
        # Generate a query for each image in the CSV file
        for index, row in df.iterrows():
            # Choose a random key from the available columns
            if possible_keys:
                random_key = random.choice(possible_keys)
                random_key = ' '.join(random_key.split('_')[1:])
            else:
                print(f"No keys found in {filename}. Skipping this file.")
                continue
            # Get the image name
            image_name = row[f'{filename}_document_name']
            # Construct the query
            query = f"find out the {random_key} in the {obb} document for the document name {image_name}?"
            print(query)
            queries.append(query)
    return queries


def generate_extraction_sql_queries(folder_path):
    data = {
        'image_name': [],
        'query': [],
        'ground_truth': []
    }
    queries = []
    # Iterate over each CSV file in the folder
    for filename, obb in abbreviations.items():
        # Construct the full file path
        file_path = os.path.join(folder_path, filename+'.csv')
        print(file_path)
        # Read the CSV file
        df = pd.read_csv(file_path)
        # Check if the 'image_name' column exists in the CSV
        if f'{filename}_document_name' not in df.columns:
            print(f"Error: 'image_name' column not found in {filename}.")
            continue
        # Get all columns except 'image_name' to use as keys
        ignore_keys = [f'{filename}_document_name', f'{filename}_set_id', f'{filename}_pdf_page_no', f'{filename}_work_item_no']
        possible_keys = [col for col in df.columns if col not in ignore_keys]# f'{filename}_document_name']
        print(possible_keys)
        for keys_ in possible_keys:
            print(keys_)
            random_key = ' '.join(keys_.split('_')[1:])
            # Generate a query for each image in the CSV file
            for index, row in df.iterrows():
                # Choose a random key from the available columns
                # if possible_keys:
                #     random_key = random.choice(possible_keys)
                #     random_key = ' '.join(random_key.split('_')[1:])
                # else:
                #     print(f"No keys found in {filename}. Skipping this file.")
                #     continue
                # Get the image name
                image_name = row[f'{filename}_document_name']
                key_value = row[keys_]
                print(key_value)
                if not pd.isna(key_value) and key_value:
                    data['image_name'].append(image_name)
                    data['ground_truth'].append(key_value)
                    # Construct the query
                    query = f"extract the {random_key} in the {obb} document for the document name {image_name}?"
                    print(query)
                    data['query'].append(query)
                    queries.append(query)
                    
    # Convert dictionary to DataFrame
    df = pd.DataFrame(data)
    # Save DataFrame as CSV
    df.to_csv('output.csv', index=False)  # Set index=False to avoid saving the index
    return queries


# Load pretrained embedding model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define your query and list of tables (and optionally descriptions of the tables)
tables = [
    'CS: contains the covering schedule document information',
    'AWB: contains the airway bill document information',
    'BOE: contains the bill of exchange document information',
    'CI: contains the commercial invoice document information',
    'COO: contains the certificate of origin document information',
    'IC: contains the insurance certificate document information',
    'PL: contains the packing list document information',
    'classification_table: contains the document class information'
]



import csv
def save_queries_to_txt(queries_list, txt_file_path):
    with open(txt_file_path, mode='w', encoding='utf-8') as txtfile:
        for query in queries_list:
            txtfile.write(query + '\n')


import json
if __name__ == '__main__':
    # Example usage
    folder_path = '/home/ntlpt19/TF_testing_EXT/ITF_TESTING/Queries_Testing/pandas_df_aug27'
    
    possible_queries_ext = generate_extraction_sql_queries(folder_path) 
       
    # Example usage
    csv_file = '/home/ntlpt19/TF_testing_EXT/ITF_TESTING/Queries_Testing/pandas_df_aug27/classification_table.csv'
    # queries_clasify = generate_queries(csv_file)
    txt_file_path = 'outputfile.txt'  # Output text file path
    save_queries_to_txt(possible_queries_ext, txt_file_path)

    exit('OK')
    table_embeddings = model.encode(tables)
    
    df_1 = pd.read_csv('/home/ntlpt19/Desktop/TF_release/TradeGPT/ext_quries_sql_testing_relevent_table.csv')
    query_dict = {
    "user_query":list(df_1['updated_query'])
            }
    results = []
    for query in query_dict['user_query']:
        # query = "Determine the airway bill number in the Covering Schedule document named pdf11_8.png."
        # Generate embeddings for query and tables
        print(query)
        query_embedding = model.encode(query)
        # Compute cosine similarities between query and each table
        similarities = cosine_similarity([query_embedding], table_embeddings)[0]
        
        #classification
        rel_tables = get_relevant_tables('prompt_template_relevant_tables3_ext', query)
        print(rel_tables)
        res = json.loads(rel_tables.content) 
        final_rel_tables = res.get('relevant_tables', [])
        # Select top-K most relevant tables based on similarity scores
        top_k = 2  # Adjust K as needed
        selected_tables = sorted(zip(tables, similarities), key=lambda x: x[1], reverse=True)[:top_k]

        # Output the selected tables
        print("Selected tables:", selected_tables)
        result = {
            'user_query': query,
            'selected_tables_embedding': str(selected_tables),  # Convert list of tuples to string
            'rel_tables_gpt': str(final_rel_tables)  # Convert list to string
        }
        results.append(result)
        print(final_rel_tables)


# Define the CSV file path
csv_file_path = 'results.csv'

# Write results to the CSV file
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['user_query', 'selected_tables_embedding', 'rel_tables_gpt'])
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f'Results have been written to {csv_file_path}')