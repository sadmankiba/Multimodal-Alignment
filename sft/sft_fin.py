from PIL import Image
import requests
from datasets import load_dataset
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from io import BytesIO
import os

# Load dataset
#dataset = load_dataset("zhiqings/LLaVA-Human-Preference-10K", split="train")
dataset = load_dataset("csv", data_files="processed_dataset.csv")['train']

#dataset = load_dataset("csv", data_files="/users/Sadman/Foundation-Model-Project/FM_Reward/processed_dataset.csv")
# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

# Load the pre-trained model for fine-tuning
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
model = model.to(device)

# Fine-tuning configuration
learning_rate = 5e-5
batch_size = 4
num_epochs = 3

# Optimizer setup
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Data preparation
def process_entry(entry, idx):
    output_folder = "downloaded_images" 
    # print(type(entry))
    question = entry['question']

    # Base URL for fetching the images
    #BASE_URL = "http://images.cocodataset.org/train2017/"
    image_url = entry['image_url']
    
    # Fetch image
    try:
        image = Image.open(os.path.join(output_folder, f"image_{idx}.jpg")).convert("RGB")
    except Exception as e:
        print(f"Error fetching or processing image: {e}")
        image = None

    answer = entry["Chosen"]  # This is the ground truth answer
    return image, question, answer

# Prepare dataset
def preprocess_data(dataset):
    images = []
    questions = []
    answers = []

    for idx, entry in enumerate(dataset):
        image, question, answer = process_entry(entry, idx)
        if image:
            images.append(image)
            questions.append(question)
            answers.append(answer)

    return images, questions, answers

images, questions, answers = preprocess_data(dataset)

# Create a DataLoader for batching
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, questions, answers, processor):
        self.images = images
        self.questions = questions
        self.answers = answers
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        question = self.questions[idx]
        answer = self.answers[idx]

        prompt = f"Question: {question} Answer:"
        # Preprocess the inputs and labels
        inputs = self.processor(images=image, text_input=prompt, return_tensors="pt", padding=True, truncation=True)
        print(inputs.keys())
        labels = self.processor.tokenizer(answer, return_tensors="pt", padding=True, truncation=True).input_ids

        # Convert tensors to the right shape and return
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }

train_dataset = CustomDataset(images, questions, answers, processor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Fine-tuning loop
def train():
    model.train()
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)))

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # Compute loss
            loss = outputs.loss
            loss.backward()

            # Update weights
            optimizer.step()

            progress_bar.set_postfix({"loss": loss.item()})
            progress_bar.update(batch_size)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Start training
train()

# Save the fine-tuned model
model.save_pretrained("blip2_finetuned")
processor.save_pretrained("blip2_finetuned")

