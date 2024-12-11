from PIL import Image
import requests
from datasets import load_dataset, Dataset
from io import BytesIO
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import matplotlib.pyplot as plt

dataset = load_dataset("zhiqings/LLaVA-Human-Preference-10K", split="train")



def process_entry(entry):
    question="" 
   
    questions = [conv['value'] for conv in entry['conversations'] if conv['from'] == 'human']
    question=questions[-1]

   
    print("In the process",question)
    # Base URL for fetching the images
    BASE_URL = "http://images.cocodataset.org/train2017/"
    image = BASE_URL + entry['image']
    return(image,question)


# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)



device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

run_generation = True 

if run_generation:
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )
    model = model.to(device)


    for entry in dataset:
        url,question=process_entry(entry)
        if "<image>" in question:
            question=question.replace("<image>", "").strip() 
        prompt = f"Question: {question} Answer:"
        image = Image.open(requests.get(url, stream=True).raw)
        plt.imshow(image)
        plt.axis('off')
        plt.title("Image")
        plt.show()

        # Print the question
        print(f"Prompt: {prompt}")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

        output1 = model.generate(**inputs, temperature=0.7, do_sample=True, max_new_tokens=50)
        output2 = model.generate(**inputs, temperature=1.0, do_sample=True, max_new_tokens=50)

        # Decode outputs
        decoded_output1 = processor.batch_decode(output1, skip_special_tokens=True)[0].strip()
        decoded_output2 = processor.batch_decode(output2, skip_special_tokens=True)[0].strip()

        # generated_ids = model.generate(**inputs, max_new_tokens=50)
        # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # print("The generated text is",generated_text)
# else: 
#     model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
#     model.to(device)

#     prompt = "Question: how many cats are there? Answer:"
#     inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

#     outputs = model(**inputs)
#     print(outputs.keys())


