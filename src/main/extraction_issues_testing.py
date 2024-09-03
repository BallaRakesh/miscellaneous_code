import re
from fuzzywuzzy import fuzz

import re
from typing import List

PRESENTATION_DATE_RE_PRECAL = r'\b(?:presented on|presentation at | presented)\s*([a-z]{3},\s*\d{1,2}(?:st|nd|rd|th)?,\s*\d{4})\b'
PRESENTATION_DATE_RE_PRECAL: str = r'\b(?:presented on|presentation at|presented)\s*((?:[A-Za-z]{3},\s*\d{1,2}(?:st|nd|rd|th)?,\s*\d{4})|(?:.{4,12}))\b'


def extract_presentation_date(ocr_data_all_text: str):
    pattern = re.compile(PRESENTATION_DATE_RE_PRECAL)
    matches: List = pattern.findall(ocr_data_all_text.lower())

    if matches:
        matches = matches[0]  # Now this should directly give you the date part
    else:
        matches = None

    print("matches")
    print("================================================")
    print(matches)  # This will give you the correct match
    return matches

# Example usage:
ocr_data_all_text = "presented on last Tuesday"
extract_presentation_date(ocr_data_all_text)
# Example 1 (Specific date): Oct 15th, 2023
# Example 2 (General format): last Tuesday
import re

PRESENTATION_DATE_RE_PRECAL = r'\b(?:presented on|presentation at|presented)\s*((?:(?:[A-Za-z]{3,9}\.?\s*,?\s*\d{1,2}(?:st|nd|rd|th)?\s*,?\s*\d{4})|(?:\d{1,2}(?:st|nd|rd|th)?\s*,?\s*[A-Za-z]{3,9}\.?\s*,?\s*\d{4}))|(?:.{4,12}))\b'
PRESENTATION_DATE_RE_PRECAL = r'\b(?:presented on|presentation at|presented)\s*((?:(?:[A-Za-z]{3,9}\.?\s*,?\s*\d{1,2}(?:st|nd|rd|th)?\s*,?\s*\d{4})|(?:\d{1,2}(?:st|nd|rd|th)?\s*,?\s*[A-Za-z]{3,9}\.?\s*,?\s*\d{4})|(?:[A-Za-z]{3,9}\.?\s*\d{1,2}(?:st|nd|rd|th)?\s*\d{4}))|(?:.{4,12}))\b'

test_cases = [
    "presented on jan 5th2003",
    "presented on Oct, 15th2023",
    "presented: Oct 15th2023 ",
    "presented on Oct, 15th, 2023",
    "presented on Oct 15th 2023",
    "presented on October 15, 2023",
    "presented on 15th Oct, 2023",
    "presented on 15 October 2023",
    "presented on 15th October, 2023",
    "presented last week",  # fallback case
]
print('++++++++++++++++++++============================++++++++++++++++++++++')
print('++++++++++++++++++++============================++++++++++++++++++++++')
for case in test_cases:
    match = re.search(PRESENTATION_DATE_RE_PRECAL, case)
    if match:
        print(f"Matched: {match.group(1)}")
    else:
        print(f"No match: {case}")
exit('????????')





def clean_text(text):
    return re.sub(r'~~{1,}', '~', re.sub(r',,{1,}', ',', text)).strip(',~').split("~")

    # Replace double commas with a single comma
    text = re.sub(r',,{1,}', ',', text)
    
    # Replace double tildes with a single tilde
    text = re.sub(r'~~{1,}', '~', text)
    
    # Remove any trailing comma or tilde
    text = text.strip(',~')
    
    return text

# Example usage
text = ",JANUARY,2016,~~,01/20/2016,,"
cleaned_text = clean_text(text)
print(cleaned_text)

exit('??????????')
ORIGINAL_CPY_TOKENS = ['original', 'org', 'copy', 'cpy']
def handle_original_cpy(input_text):
    for org_cpy_words in ORIGINAL_CPY_TOKENS:
        if org_cpy_words in input_text.lower() or fuzz.ratio(org_cpy_words, input_text.lower()) > 70:
            return org_cpy_words
    else:
        return ''
   
text = 'bioginal'

print(handle_original_cpy(text))







import re

# The text containing the date
text = "date 2003 09 25"

# Updated regular expression to include various date formats
EXTRACT_DATE_RE = r"""
\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}        # MM/DD/YYYY or similar
|\d{1,2}[./]\d{1,2}[./]\d{2,4}            # MM.DD.YYYY or similar
|\d{4}[-/]\d{1,2}[-/]\d{1,2}              # YYYY-MM-DD or similar
|\d{1,2}\s\d{1,2}\s\d{2,4}                # DD MM YYYY
|\d{4}\s\d{2}\s\d{2}                      # YYYY MM DD (new format)
|\d{1,2}[-/]\w{3,9}[-/]\d{2,4}            # 12-Mar-2022 or similar
|\w{3,9}[.]\d{1,2}[,]\d{1,4}              # Mar.12,2022 or similar
|\d{1,2}[ /]\w{3,9}[ /]\d{2,4}            # 12 Mar 2022 or similar
|\w{3,9}[-/]\d{1,2}[-/]\d{2,4}            # Mar-12-2022 or similar
|\w{3,9}\.? \d{1,2},? \d{2,4}             # Mar 12, 2022 or similar
|\d{1,2}(?:st|nd|rd|th) [A-Za-z]+ \d{4}   # 12th March 2022 or similar
|\d{1,2}(?:st|nd|rd|th) [A-Za-z]+, \d{4}  # 12th March, 2022 or similar
|[A-Za-z]+ \d{1,2},\d{4}                  # March 12,2022 or similar
|\d{1,2} [A-Za-z]{3,9} [.] [,] \d{2,4}    # 12 March . , 2022 or similar
|[A-Za-z]+ \d{1,2}, \d{4}                 # March 12, 2022 or similar
)\b"""

# Compile the pattern with re.VERBOSE to handle multiline regex
pattern = re.compile(EXTRACT_DATE_RE, re.VERBOSE)

# Find all matching dates in the text
dates = re.findall(pattern, text)

# Print out each matched date
if dates:
    for date in dates:
        print("Extracted Date:", date)
else:
    print("No dates found")





import re

# The text containing the date
text = "date 2003 09 25"
EXTRACT_DATE_RE : str = r"""\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{1,2}[./]\d{1,2}[ ./ ]\d{2,4}|\d{4}\s\d{2}\s\d{2}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}\d{1,2}\d{2,4}|\d{1,2}[-/]\w{3,9}[-/]\d{2,4}|\w{3,9}[.]\d{1,2}[,]\d{1,4}|\d{1,2}[ /]\w{3,9}[ /]\d{2,4}|\w{3,9}[-/]\d{1,2}[-/]\d{2,4}|\w{3,9}\.? \d{1,2},? \d{2,4}|\d{1,2}(?:st|nd|rd|th) [A-Za-z]+ \d{4}|\d{1,2}(?:st|nd|rd|th) [A-Za-z]+, \d{4}|[A-Za-z]+ \d{1,2},\d{4}|\d{1,2} [A-Za-z]{3,9} [.] [,] \d{2,4}|[A-Za-z]+ \d{1,2}, \d{4})\b"""
                                                                                                                             ###################                                      
# |\d{4}\s\d{2}\s\d{2})\b
pattern = EXTRACT_DATE_RE
new_format_dates=[]
actual_dates= []
dates = re.findall(pattern, text)
correct_dates= []
if len(dates)!=0:
    for index, date in enumerate(dates):
        print(date)


exit('?')
# Regular expression to find the date
date_match = re.search(r'\d{4} \d{2} \d{2}', text)

# Extract and print the date
if date_match:
    date = date_match.group()
    print("Extracted Date:", date)
else:
    print("Date not found")
