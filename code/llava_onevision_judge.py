import requests
from PIL import Image

import torch
import pandas as pd
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


def make_prompt_for_comparison(question, response1, response2):
    text = """
        Question: {question}
        Response 1: {response1}
        Response 2: {response2}
        
        Given the question, which response is more accurate based on content of 
        the image. If they are equally accurate, which is more helpful ? Give your
        answer in this format: "1" for the first response, "2" for the second response
        """
        
    return text

class LlavaOnevisionJudge:
    def __init__(self, model_id):
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(0)

        self.processor = AutoProcessor.from_pretrained(model_id)

    def get_judgement(self, image, question, response1, response2):
        prompt_text = make_prompt_for_comparison(question, response1, response2)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)

        output = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        response = self.processor.decode(output[0][2:], skip_special_tokens=True)
        
        preferred = response.strip().split("\n")[-1].strip() # Get the last line
        if preferred == "1" or preferred == "2":
            return int(preferred)
        else:
            print("Did not get a valid response. Skipping...")
            return 0


def test_judgement():
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    judge = LlavaOnevisionJudge(model_id)
    
    image_file = "http://images.cocodataset.org/train2017/000000039768.jpg"
    image = Image.open(requests.get(image_file, stream=True).raw)
    
    question = "What is the man doing?"
    response1 = "The man is giving speech in a crowd"
    response2 = "The man is giving a flower to a girl"
    preferred = judge.get_judgement(image, question, response1, response2)
    print(preferred)


if __name__ == "__main__":
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    judge = LlavaOnevisionJudge(model_id)
    
    # Read the CSV file
    blip2_generated_samples_file = "/users/Sadman/Foundation-Model-Project/data/processed_llava_dataset.csv"
    df = pd.read_csv(blip2_generated_samples_file)

    data_with_judgements = []
    for index, row in df.iterrows():
        image_url = row['image_url']
        image = Image.open(requests.get(image_url, stream=True).raw)
        
        question = row['prompt']
        response1 = row['output_temp_0.7']
        response2 = row['output_temp_1.0']
        
        judgement = judge.get_judgement(image, question, response1, response2)
        print(f"Judgement for row {index}: {judgement}")
        if judgement == 0:
            continue
        data_with_judgements.append({
            'image_url': image_url,
            'question': question,
            'response1': response2,
            'response2': response1,
            'judgement': judgement
        })
        
        if (index + 1) % 5 == 0:
            output_df = pd.DataFrame(data_with_judgements)
            output_df.to_csv(f"/users/Sadman/Foundation-Model-Project/data/onevision_judgement_{index + 1}.csv", index=False)
            
        if (index + 1) % 10 == 0:
            break
    
    
    



