import requests
from collections import namedtuple
from utils import get_coco_image

import torch
import pandas as pd
from datasets import Dataset
from PIL import Image
from transformers import Blip2ForConditionalGeneration, AutoProcessor
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig

# Procedure
# 1. Prepare dataset from BLIP2 sample judgements
# 2. Load BLIP2 model and processor 
# 3. Set training arguments. Save locally and push to hub. 
# 4. Train model with DPO. 

# Login to HF
# pip install huggingface_hub
# huggingface-cli login

def get_blip2_pref_dataset(num_items=100):
    blip2_sample_judgements_file = "../data/blip2_sample_judge_9000.csv"
    df = pd.read_csv(blip2_sample_judgements_file)

    data_list = []
    coco_dataset_name = df.iloc[0]['image_url'].split('/')[-2]
    for idx, row in df.iterrows():
        if row['judgement'] == 1:
            chosen = row['response1']
            rejected = row['response2']
        else:
            chosen = row['response2']
            rejected = row['response1']
        
        image_file = row['image_url'].split('/')[-1]
        image = get_coco_image(coco_dataset_name, image_file)
        data_list.append({
            'images': image,
            'prompt': row['question'],
            'chosen': str(chosen),
            'rejected': str(rejected)
        })
        
        if idx == num_items:
            break

    def gen_data():
        for data in data_list:
            yield data
            
    dataset = Dataset.from_generator(gen_data)
    return dataset

def load_blip2_model(args):
    model_id = "Salesforce/blip2-opt-2.7b"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, 
                torch_dtype=torch.float16).to(args.device)
    
    return model, processor

def train_model_dpo(model, processor, train_dataset, eval_dataset, args):
    training_args = DPOConfig(
        output_dir=args.output_dir,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_chekpointing,
        max_steps=1, # testing purpose
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        logging_dir=args.logging_dir,
        logging_strategy="steps",
        save_strategy="steps",
        push_to_hub=True,
        hub_private_repo=True,
        remove_unused_columns=False
    )
    trainer = DPOTrainer(
        model,
        ref_model=None,  # not needed when using peft
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=LoraConfig(),
    )

    trainer.train()

if __name__ == "__main__": 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Args = namedtuple('Args', ['batch_size', 'num_epochs', 'logging_steps', 
                'gradient_accumulation_steps', 'gradient_chekpointing', 
                'bf16', 'output_dir', 'logging_dir', 'device'])
    
    args = Args(
        batch_size=1,
        num_epochs=3,
        logging_steps=1,
        gradient_accumulation_steps=4,
        gradient_chekpointing=True,
        bf16=True,
        output_dir="blip2-dpo",
        logging_dir="../logs",
        device=device
    )
    print(args)
    
    dataset = get_blip2_pref_dataset()
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    print("Prepared dataset")
    
    model, processor = load_blip2_model(args)
    
    print("Starting training")
    train_model_dpo(model, processor, train_dataset, eval_dataset, args)