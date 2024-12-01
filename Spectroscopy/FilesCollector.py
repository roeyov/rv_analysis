import os
import json
from constants import *

def load_json_elements(json_file, json_key):
    with open(json_file, 'r') as f:
        data = json.load(f)
        print(data.keys())
    # Flatten the dictionary to a list of unique elements
    elements = set()
    elements.update(data[json_key])
    return elements


def find_files_with_strings(strings, directory, fits_suf):
    """
    Recursively finds all files in a directory whose names contain any of the given strings.

    Parameters:
        strings (list of str): The strings to search for in file names.
        directory (str): The root directory to start the search.

    Returns:
        dict: A dictionary where keys are the strings and values are lists of full paths of matching files.
    """
    # Initialize the dictionary to hold results
    result = {string: [] for string in strings}

    # Walk through the directory recursively
    for root, _, files in os.walk(directory):
        for file in files:
            for string in strings:
                if string in file and fits_suf in file :
                    # Add the full path of the file to the appropriate list
                    result[string].append(os.path.join(root, file))

    return result


def main():

    json_file_key = 'Sample O + 10 early BVs'  # Update to the directory you want to search
#    dict_keys(['Sample OeBe', 'Sample O + 10 early BVs', 'Sample early BVs + 10 early BIs', 'Sample early BIs', 'Sample late SGs + 10 early BIs'])
    fits_suf = FITS_SUF_COMBINED

    # Load elements from the JSON file
    elements = load_json_elements(OSTARS_IDS_JSON, json_file_key)
    matching_files = find_files_with_strings(list(elements), DATA_RELEASE_3_PATH, fits_suf)
    for k,v in matching_files.items():
        print(k,v[0])

if __name__ == '__main__':
    main()
