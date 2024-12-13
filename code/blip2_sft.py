
import json 
import requests
import os
from functools import partial
from io import BytesIO

import torch
from PIL import Image
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from trl import SFTConfig, SFTTrainer
from collections import namedtuple

from utils import get_coco_image

# SFT for BLIP2 on LLaVa Instruct 

# Procedure 
# 1. Load LLaVa Instruct dataset from HF and truncate. Load images.
# 2. Load BLIP2 model and processor
# 3. Config SFT trainer. 
# 4. Train the model and save it.

def get_llava_instruct_dataset(num_items=100):
    llava_instruct_filepath = os.path.expanduser('~/Foundation-Model-Project/data/llava_instruct_150k.json')
    
    with open(llava_instruct_filepath, 'r') as file:
        llava_instruct_data = json.load(file)
    
    print("Dataset len", len(llava_instruct_data))
    print(llava_instruct_data[0])
    
    data_list = []
    for idx, row in enumerate(llava_instruct_data):
        image_filename = row['image']
        image = get_coco_image("train2017", image_filename)
        for i in range(0, len(row['conversations'])-1, 2):
            assert row['conversations'][i]['from'] == 'human'
            assert row['conversations'][i+1]['from'] == 'gpt'
            question = row['conversations'][i]['value']
            answer = row['conversations'][i+1]['value']
            data_list.append({
                'images': image,
                'question': question,
                'answer': answer
            })
            if len(data_list) == num_items:
                break
        
        print("Processed", idx)
        
    def gen_data():
        for data in data_list:
            yield data
            
    dataset = Dataset.from_generator(gen_data)
    return dataset
    

def load_blip2_model(args):
    # Load processor and model
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    ).to(args.device)
    
    return model, processor

def _collate_fn(examples, processor):
    # Get the texts and images, and apply the template
    print("Examples", examples)
    texts = [] 
    mask_texts = []
    for example in examples:
        texts.append(f"Question: {example['question']} Answer: {example['answer']}")
        mask_texts.append(f"Question: {example['question']} Answer:")

    images = [example["images"][0] for example in examples]
    print("Images", images)

    # Tokenize the texts and process the images
    input_encoding = processor(images=images, text=texts, return_tensors="pt", padding=True)
    mask_encoding = processor(images=images, text=mask_texts, return_tensors="pt", padding=True)
    print("input_encoding keys", input_encoding.keys())
    print("input_encoding input_ids", input_encoding["input_ids"])
    print("mask_encoding input_ids", mask_encoding["input_ids"])
    
    # The labels are the input_ids, and we mask the query and question tokens in the loss computation
    labels = input_encoding["input_ids"].clone()
    labels[input_encoding["input_ids"] == mask_encoding["input_ids"]] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    print("labels", labels)
    input_encoding["labels"] = labels
    exit()
    return input_encoding


def run_sft(model, processor, train_dataset, eval_dataset, args):
    training_args = SFTConfig(
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        fp16=True,
        gradient_checkpointing=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        torch_compile=True,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        logging_dir=args.logging_dir,
        logging_strategy="steps",
        push_to_hub=True,
        hub_private_repo=True,
        max_steps=args.max_steps,
        save_strategy="no",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=partial(_collate_fn, processor=processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.tokenizer,
    )
    
    trainer.train()

if __name__ == "__main__":
    Args = namedtuple('Args', ['device', 'max_seq_length', 'max_steps', 
            'batch_size', 'num_epochs', 'logging_steps', 
            'gradient_accumulation_steps', 'output_dir', 'logging_dir'])
    args = Args(device='cuda', max_seq_length=512, max_steps=1, 
        batch_size=1, num_epochs=3, logging_steps=1, 
        gradient_accumulation_steps=1, output_dir="blip2-sft", logging_dir="../logs")
    
    dataset = get_llava_instruct_dataset(num_items=40)
    dataset = dataset.map()
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    print("Dataset length", len(dataset))
    print("Dataset[0]", dataset[0])
    print("Train dataset length", len(train_dataset))
    print("Eval dataset length", len(eval_dataset))
    
    model, processor = load_blip2_model(args)
    
    
    run_sft(model, processor, train_dataset, eval_dataset, args)

    model.save_pretrained("sft_blip2_llava_instruct")
