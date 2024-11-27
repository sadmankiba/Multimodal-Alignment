import json
import random

# Load the JSON data
with open('data/llava_7b_v1_preference.json', 'r') as file:
    data = json.load(file)

MAX_OUTPUT_LEN = 40
# Filter entries with hallucination true and both outputs less than MAX_OUTPUT_LEN words
filtered_entries = [
    entry for entry in data
    if entry.get('hallucination') and
    len(entry['output_1']['value'].split()) < 40 and
    len(entry['output_2']['value'].split()) < 40
]

print(f"Found {len(filtered_entries)} entries with hallucination")

# Print a random entry from the filtered list
if filtered_entries:
    random_entry = random.choice(filtered_entries)
    print(json.dumps(random_entry, indent=4))
else:
    print("No entries found with the specified criteria.")
    
# Found id 22564