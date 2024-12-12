import torch
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Procedure
# 1. Load POPE dataset from HF and truncate. Load images.
# 2. Load BLIP2 model and processor
# 3. Run inference on POPE. Use model.generate(). 
# 4. Get metrics accuracy, precision, recall, f1 

from datasets import load_dataset

dataset = load_dataset("lmms-lab/POPE")
print(dataset)

# Sample POPE
# {'id': '0', 'question_id': '1', 'question': 'Is there a snowboard in the image?', 
# 'answer': 'yes', 'image_source': 'COCO_val2014_000000310196', 'category': 'adversarial',
# 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x427 at 0x7F29402C7250>}

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )
model = model.to(device)

responses = []
for i, entry in enumerate(tqdm(dataset)): 
    prompt = f"Question: {entry['question']} Answer:"
    image = entry['image']
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    output = model.generate(**inputs, max_new_tokens=20)
    
    decoded_output = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
    yes_no_response = None
    if 'yes' in decoded_output.lower():
        yes_no_response = 'yes'
    elif 'no' in decoded_output.lower():
        yes_no_response = 'no'
    else:
        yes_no_response = 'unknown'
    
    responses.append({
        'id': entry['id'],
        'question': entry['question'],
        'answer': entry['answer'],
        'image_source': entry['image_source'],
        'category': entry['category'],
        'response': yes_no_response
    })
    
    # Save every 1000 responses
    if (i + 1) % 5 == 0:
        print("Saving responses...")
        pd.DataFrame(responses).to_csv(f"../data/pope_blip2_responses_{i}.csv", index=False)
        
    if i == 10: 
        break