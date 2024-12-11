from PIL import Image
import requests

from transformers import Blip2Processor, Blip2Model
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

url = "http://images.cocodataset.org/train2017/000000013249.jpg"
image = Image.open(requests.get(url, stream=True).raw)



device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

run_generation = True 

if run_generation:
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )
    model = model.to(device)

    prompt = "Question: Can you give possible reasons for the crowd of people to be walking in the street? Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)
else: 
    model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    model.to(device)

    prompt = "Question: how many cats are there? Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    outputs = model(**inputs)
    print(outputs.keys())


