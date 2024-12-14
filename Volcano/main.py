import json
import io
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import requests
from io import BytesIO
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoProcessor, AutoModelForImageTextToText
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX

from transformers import BitsAndBytesConfig, LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import Blip2Processor, Blip2ForConditionalGeneration

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

df = pd.read_parquet("pope.parquet", engine="pyarrow")[:100]

ov_model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", torch_dtype="float16", device_map='auto')
ov_processor = LlavaOnevisionProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
ov_processor.tokenizer.padding_side = "left"


# qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
# # qwen_model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
# qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )

blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")

def load_image(image_file):
    image = None
    try:
      if image_file.startswith('http') or image_file.startswith('https'):
          response = requests.get(image_file)
          image = Image.open(BytesIO(response.content)).convert('RGB')
      else:
          image = Image.open(image_file).convert('RGB')
    except:
      print("There was an error reading the image file", image_file)
    return image

def generate_llama_onevision(model, processor, image_path, text):
    image = load_image(image_path)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image"}
            ]
        }
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=[image], text=[prompt], padding=True, return_tensors="pt").to(model.device, torch.float16)
    generate_kwargs = {"max_new_tokens": 50, "do_sample": True, "top_p": 0.9}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    response = generated_text[0].split("assistant\n")[1]
    return response

def generate_qwenvl(model, processor, image_path, text):
  conversation = [
      {
          "role": "user",
          "content": [
              {
                  "type": "image",
                  "image": f"{image_path}",
              },
              {"type": "text", "text": text},
          ],
      }
  ]

  text = processor.apply_chat_template(
      conversation, tokenize=False, add_generation_prompt=True
  )
  image_inputs, video_inputs = process_vision_info(conversation)
  inputs = processor(
      text=[text],
      images=image_inputs,
      videos=video_inputs,
      padding=True,
      return_tensors="pt",
  )
  inputs = inputs.to(model.device)

  # Inference: Generation of the output
  generated_ids = model.generate(**inputs, max_new_tokens=64)
  generated_ids_trimmed = [
      out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
  ]
  output_text = processor.batch_decode(
      generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
  )
  return output_text[0]

def generate_blip(model, processor, image_path, text):
    image = load_image(image_path)

    prompt = text + " Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    try:
        return generated_text.split("Answer:")[1]
    except:
        return generated_text

def inference(model, image_path, text):
    if model == "qwen":
        result = generate_qwenvl(qwen_model, qwen_processor, image_path, text)
        # print(result)
        return result
    if model == "ov":
        return generate_llama_onevision(ov_model, ov_processor, image_path, text)
    if model == "blip":
        return generate_blip(blip_model, blip_processor, image_path, text)
    

def eval_model(json_file, output, model, captions=False):
    # Load input JSON data
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    logs = []
    iter_cnt = {1: 0, 2: 0, 3: 0}
    # print(json_data)

    for idx, line in tqdm(enumerate(json_data)):
        image_path = line['image_src']
        image = load_image(image_path)
        if image == None:
          continue
        question = line['question']
        qs = question

        if captions:
            captions = inference(model, image_path, "Caption this image.")
            question = f"{question}\nYou are also provided image captions below to help you answer the question\n{captions}" 

        # Initial question generation
        initial_answer = inference(model, image_path, question)
        outputs = initial_answer
        revision = None

        # print('Initial output:', outputs)

        # Feedback and refinement loop
        for i in range(3):
            fqs = (
                "Generate the feedback given initial answer referring to question and image.\n"
                f"Question: {question}\nInitial answer: {outputs}"
            )
            feedback1 = inference(model, image_path, fqs)
            # print('Feedback:', feedback1)

            rqs = (
                "Adjust the initial response considering the feedback and image.\n"
                f"Question: {question}\nInitial answer: {outputs}\nFeedback: {feedback1}"
            )
            revision = inference(model, image_path, rqs)
            # print("Revision:", revision)

            comparison_prompt = (
                f"{question}\nA. {outputs}\nB. {revision}\n"
                "Answer with the option's letter from the given choices directly."
            )
            comparison_result = inference(model, image_path, comparison_prompt)

            # print('Answer comparison:', comparison_result)

            if 'b' in comparison_result.lower():
                outputs = revision
            elif 'a' in comparison_result.lower():
                break

        iter_cnt[i + 1] += 1

        # Logging results
        logs.append({
            'question': question,
            'gold answer': line['gt_answer'],
            'initial answer': initial_answer,
            'feedback': feedback1,
            'revision': revision,
        })

        json_data[idx]['model_answer'] = outputs
        print('---------------------------------')
        print('question: ', question)
        print('answer: ', line['gt_answer'])
        print('initial pred: ', initial_answer)
        print('pred: ', outputs)
        print('---------------------------------')

    with open(output, 'w') as f:
        json.dump(json_data, f, indent="\t")

    return logs

def eval_model_pope(json_file, output, model, captions=False):
    # Load input JSON data
    # with open(json_file, 'r') as f:
    #     json_data = json.load(f)

    logs = []
    iter_cnt = {1: 0, 2: 0, 3: 0}
    # print(json_data)

    for idx, (i, row) in tqdm(enumerate(df.iterrows())):
        # print(row)
        image = Image.open(io.BytesIO(row["image"]["bytes"]))
        image_path = "test.jpg"
        image.save("test.jpg")
        question = row["question"]
        answer = row["answer"]
        if image == None:
            continue
            print("No Image")
        qs = question

        if captions:
            captions = inference(model, image_path, "Caption this image.")
            question = f"{question}\nYou are also provided image captions below to help you answer the question\n{captions}" 

        # Initial question generation
        initial_answer = inference(model, image_path, question)
        outputs = initial_answer
        revision = None

        # print('Initial output:', outputs)

        # Feedback and refinement loop
        for i in range(3):
            fqs = (
                "Generate the feedback given initial answer referring to question and image.\n"
                f"Question: {question}\nInitial answer: {outputs}"
            )
            feedback1 = inference(model, image_path, fqs)
            # print('Feedback:', feedback1)

            rqs = (
                "Adjust the initial response considering the feedback and image.\n"
                f"Question: {question}\nInitial answer: {outputs}\nFeedback: {feedback1}"
            )
            revision = inference(model, image_path, rqs)
            # print("Revision:", revision)

            comparison_prompt = (
                f"{question}\nA. {outputs}\nB. {revision}\n"
                "Answer with the option's letter from the given choices directly."
            )
            comparison_result = inference(model, image_path, comparison_prompt)

            # print('Answer comparison:', comparison_result)

            if 'b' in comparison_result.lower():
                outputs = revision
            elif 'a' in comparison_result.lower():
                break

        iter_cnt[i + 1] += 1

        # Logging results
        logs.append({
            'question': question,
            'gt_answer': answer,
            'model_answer': outputs
        })

        # json_data[idx]['model_answer'] = outputs
        print('---------------------------------')
        print('question: ', question)
        print('answer: ', answer)
        print('initial pred: ', initial_answer)
        print('pred: ', outputs)
        print('---------------------------------')

    with open(output, 'w') as f:
        json.dump(logs, f, indent="\t")

    return logs

for model in ["ov", "blip"]:
    eval_model_pope("response_template.json", f"results_POPE/{model}_response_no_captions.json", model, False)
    eval_model_pope("response_template.json", f"results_POPE/{model}_response_captions.json", model, True)
