import requests
from PIL import Image

import tqdm
import torch
import pandas as pd
from transformers import (
    AutoProcessor, 
    LlavaOnevisionForConditionalGeneration,
    Qwen2VLForConditionalGeneration
)


class LlavaOnevisionJudge:
    def __init__(self, model_id):
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(0)

        self.processor = AutoProcessor.from_pretrained(model_id)

    def make_prompt_for_comparison(self, question, response1, response2):
        text = f"""
            Question: {question}
            Response 1: {response1}
            Response 2: {response2}
            
            Given the question, which response is more accurate based on content of 
            the image. If they are equally accurate, which is more helpful ? Give your
            answer in this format: "1" for the first response, "2" for the second response
            """
        
        return text

    def get_judgement(self, image, question, response1, response2):
        prompt_text = self.make_prompt_for_comparison(question, response1, response2)
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


class Qwen2VLJudge:
    def __init__(self, model_id):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="auto", 
        ).to(0)

        self.processor = AutoProcessor.from_pretrained(model_id)
        
    def get_judgement(self, image, question, response1, response2):
        prompt_text = self.make_prompt_for_comparison(question, response1, response2)
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(images=[image], text=[prompt], padding=True, return_tensors='pt').to(self.model.device, torch.float16)

        output_ids = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        print("output_text", output_text)
        if output_text == "A" or '"A"' in output_text: 
            return 1
        elif output_text == "B" or '"B"' in output_text:
            return 2
        else:
            return 0

        
    def make_prompt_for_comparison(self, question, response1, response2):
        # Does not work
        text1 = f"""
            Question: {question}
            Response 1: {response1}
            Response 2: {response2}
            
            Given the question, which response is more accurate based on content of 
            the image. If they are equally accurate, which is more helpful ? Give your
            answer in this format: "1" for the first response, "2" for the second response
            """
        
        # Does not work
        text2 = f"""
            Which of the following response is more accurate? Give explanation first 
            and then choose one.
            
            Question: {question}
            Response 1: {response1}
            Response 2: {response2}
            """
        
        # Repeats the answer
        text3 = f"""
            Question: {question}
            Response 1: {response1}
            Response 2: {response2}
            
            Which of the following response is more accurate? "2" or "1"?
            """
        
        # Works
        text4 = f"""
            Question: {question}
            Response A: {response1}
            Response B: {response2}
            
            Which of the following response is more accurate? "B" or "A"?
            """
        
        # Does not work
        text5 = f"""
            Question: {question}
            Response 1: {response1}
            Response 2: {response2}
            
            Which of the following response is more accurate? Give your
            answer in this format: "1" for the first response, "2" for the second response
            """
        
        return text4
    
# Some responses
# output_text ['B']
# output_text ['B']
# output_text ['The answer is "B". The response "B" is more accurate as it is a question asking']

def test_llava_onevision_judgement():
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    judge = LlavaOnevisionJudge(model_id)
    
    image_file = "http://images.cocodataset.org/train2017/000000039768.jpg"
    image = Image.open(requests.get(image_file, stream=True).raw)
    
    question = "What is the man doing?"
    response1 = "The man is giving speech in a crowd"
    response2 = "The man is giving a flower to a girl"
    preferred = judge.get_judgement(image, question, response1, response2)
    print(preferred)


def test_qwen2_vl_judgement():
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    judge = Qwen2VLJudge(model_id)
    
    image_file = "http://images.cocodataset.org/train2017/000000039768.jpg"
    image = Image.open(requests.get(image_file, stream=True).raw)
    
    question = "What is the man doing?"
    response1 = "There is a bird in the sky"
    response2 = "The man is giving a flower to a girl"
    preferred = judge.get_judgement(image, question, response1, response2)
    print(preferred)


if __name__ == "__main__":
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    judge = Qwen2VLJudge(model_id)
    
    # Read the CSV file
    blip2_generated_samples_file = "../data/processed_llava_dataset.csv"
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
        if judgement == 1 or judgement == 2:    
            data_with_judgements.append({
                'image_url': image_url,
                'question': question,
                'response1': response2,
                'response2': response1,
                'judgement': judgement
            })
        
        if (index + 1) % 500 == 0:
            output_df = pd.DataFrame(data_with_judgements)
            output_df.to_csv(f"../data/onevision_judgement_{index + 1}.csv", index=False)
    
    
    



