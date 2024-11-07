import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration # requires torch >= 2.4

model_id = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

class LlavaOnevision:
    def __init__(self):
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(0)

        self.processor = AutoProcessor.from_pretrained(model_id)
        
        
    def __call__(self, image, question):
        # Each value in "content" has to be a list of dicts with types ("text", "image") 
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                    ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)

        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        text_output = self.processor.decode(output[0][2:], skip_special_tokens=True)
        return text_output[text_output.find("assistant")+len("assistant")+1:]
        
    
def build_llava_onevision():
    return LlavaOnevision()

def test_llava_onevision():
    model = build_llava_onevision()
    
    # from web
    # image_file = "http://images.cocodataset.org/train2017/000000039768.jpg"
    # image = Image.open(requests.get(image_file, stream=True).raw)
          
    image_file = "images/2519330533_597840098a_o.jpg"
    image = Image.open(image_file).convert('RGB')  # PIL already opens in RGB mode
    response = model(image, "What are inside this?")
    print(response)

if __name__ == '__main__':
    test_llava_onevision()