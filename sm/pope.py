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
processor = Blip2Processor.from_pretrained("laynad/dpo-checkpoint")

def get_blip2_model(path: str):
    model = Blip2ForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.float16
        )
    model = model.to(device)
    return model

model, name = get_blip2_model("laynad/dpo-checkpoint"), "blip2-base"

responses = []
for i, entry in enumerate(tqdm(dataset['test'])): 
    prompt = f"Question: {entry['question']} Answer:"
    image = entry['image']
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    # model, name = get_blip2_model("blip2-sft"), "blip2-sft"
    
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
    if (i + 1) % 1000 == 0:
        print("Saving responses...")
        pd.DataFrame(responses).to_csv(f"../data/pope_{name}_responses_{i+1}.csv", index=False)