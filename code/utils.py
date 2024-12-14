import os

import requests
from io import BytesIO

from PIL import Image

def get_coco_image(dataset_name, image_name):
    """
    Download the image or read from a saved directory
    
    Args:
        dataset_name: Example train2017
        image_name: Example 000000012345.jpg
    """
    dir = f'{os.path.expanduser("~")}/coco/{dataset_name}' 
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # If saved, return from local dir
    local_image_path = os.path.join(dir, image_name)
    if os.path.exists(local_image_path):
        image = Image.open(local_image_path)
        return image
    
    # Download image
    coco_url = f"http://images.cocodataset.org/{dataset_name}/{image_name}"
    response = requests.get(coco_url)
    
    image = Image.open(BytesIO(response.content))
    
    # Save image
    image.save(local_image_path)
    
    return image 