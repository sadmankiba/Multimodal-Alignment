from PIL import Image
import requests
from datasets import Dataset, load_dataset
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import matplotlib.pyplot as plt

# Load dataset
dataset = load_dataset("zhiqings/LLaVA-Human-Preference-10K", split="train")

# Process each dataset entry
def process_entry(entry):
    questions = [conv['value'] for conv in entry['conversations'] if conv['from'] == 'human']
    question = questions[-1] if questions else ""
    print("Processing Question:", question)

    BASE_URL = "http://images.cocodataset.org/train2017/"
    image_url = BASE_URL + entry['image']
    return image_url, question

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor and model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
).to(device)

# List to store new dataset entries
new_dataset_entries = []

# Iterate through the dataset
for entry in dataset:
    # Process entry
    url, question = process_entry(entry)
    if "<image>" in question:
        question = question.replace("<image>", "").strip()

    prompt = f"Question: {question} Answer:"
    print(f"Prompt: {prompt}")

    # Load image with error handling
    try:
        image = Image.open(requests.get(url, stream=True).raw)
    except Exception as e:
        print(f"Error loading image: {e}")
        continue

    # Prepare inputs for the model
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    # Generate responses with different temperatures
    outputs = {}
    for temp in [0.7, 1.0]:
        generated_ids = model.generate(
            **inputs, temperature=temp, do_sample=True, max_new_tokens=50
        )
        decoded_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        outputs[temp] = decoded_output

    # Store the data in the new dataset structure
    new_entry = {
        "image_url": url,
        "prompt": prompt,
        "output_temp_0.7": outputs[0.7],
        "output_temp_1.0": outputs[1.0]
    }
    new_dataset_entries.append(new_entry)

# Create new dataset
new_dataset = Dataset.from_list(new_dataset_entries)

# Save the dataset to a file
new_dataset.save_to_disk("processed_llava_dataset")
print("New dataset saved to 'processed_llava_dataset'")
