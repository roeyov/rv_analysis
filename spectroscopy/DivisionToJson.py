import os

import pandas as pd
import json

# Load the Excel file
file_path = r"C:\Users\roeyo\Documents\Roey's\Masters\Reasearch\Data\BLOeM Project Overview.xlsx"  # Update with your actual file path

file_path_without_extension = os.path.splitext(file_path)[0]

df = pd.read_excel(file_path, header=None)
# Convert to dictionary format
data_dict = {}
headers = df.iloc[0]
for i, header in enumerate(headers):
    if pd.isna(header):
        continue
    data_dict[header] = df.iloc[1:, i].dropna().tolist()

# Save to JSON
json_file = file_path_without_extension + ".json"
with open(json_file, 'w') as f:
    json.dump(data_dict, f, indent=4)
print(f"Data has been saved to {json_file}")
