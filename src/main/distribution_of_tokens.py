import json

# Function to tokenize the content
def tokenize(content):
    return content.split()

# Initialize variables to keep track of the cell with the highest token count
def get_count(cells):
    max_tokens = 0
    # Analyze each cell
    for cell in cells:
        content = cell['content']
        # Tokenize the content
        tokens = tokenize(content)
        num_tokens = len(tokens)
        
        if num_tokens > max_tokens:
            max_tokens = num_tokens
    return max_tokens
    # Output the cell with the highest number of tokens


cells = [
    {
        "content": "cold dough premix fdpz3004",
        "bbox": [0, 86, 467, 116],
        "column_nums": [0],
        "row_nums": [0]
    },
    {
        "content": "11 ctn cold dough premix",
        "bbox": [467, 86, 838, 116],
        "column_nums": [1],
        "row_nums": [0]
    },
    {
        "content": "",
        "bbox": [838, 86, 1067, 116],
        "column_nums": [2],
        "row_nums": [0]
    }
]
print(get_count(cells))