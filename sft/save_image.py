import os
from PIL import Image
import requests
from io import BytesIO
from datasets import load_dataset

# Define the folder to save the images
output_folder = "downloaded_images"
os.makedirs(output_folder, exist_ok=True)

# Load dataset
dataset = load_dataset("csv", data_files="/home/dxm060/fm/Foundation-Model-Project/sft/processed_dataset.csv")['train']

# Save images to folder
def save_images(dataset, output_folder):
    for idx, entry in enumerate(dataset):
        image_url = entry['image_url']
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()  # Check for HTTP request errors
            image = Image.open(BytesIO(response.content)).convert("RGB")

            # Save the image with a unique name
            image_path = os.path.join(output_folder, f"image_{idx}.jpg")
            image.save(image_path)
            print(f"Saved: {image_path}")
        except Exception as e:
            print(f"Error saving image {idx} from {image_url}: {e}")

save_images(dataset, output_folder)
