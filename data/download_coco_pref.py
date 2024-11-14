import os
import json
import requests

# Create the images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Load the JSON data
with open('llava_7b_v1_preference.json', 'r') as file:
    data = json.load(file)

# Base URL for COCO dataset images
base_url = 'http://images.cocodataset.org/train2017/'

print(f"Dataset has {len(data)} images...")

N_DOWNLOAD = 500

print(f"Downloading {N_DOWNLOAD} images...")

# Download each image
i = 0
for entry in data:
    image_name = entry['image']
    image_url = f"{base_url}{image_name}"
    image_path = os.path.join('images', image_name)
    
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(image_path, 'wb') as img_file:
            img_file.write(response.content)
        print(f"Downloaded {image_name}")
    else:
        print(f"Failed to download {image_name}")
        
    i += 1
    if i >= N_DOWNLOAD:
        break
    