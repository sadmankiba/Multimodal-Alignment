from transformers import BlipForConditionalGeneration, BlipProcessor
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from PIL import Image
import os

# Load your dataset
dataset = load_dataset("zhiqings/LLaVA-Human-Preference-10K")

# Function to process each sample
def process_sample(sample):
    # Determine the chosen and rejected responses
    chosen = sample["output_2"]["value"] if sample["preference"] == 2 else sample["output_1"]["value"]
    rejected = sample["output_1"]["value"] if sample["preference"] == 2 else sample["output_2"]["value"]
    
    # Return the updated sample with chosen and rejected
    return {"chosen": chosen, "rejected": rejected}

# Apply the processing function to the 'train' split
dataset['train'] = dataset['train'].map(process_sample)

# Load BLIP model and processor
model_name = "Salesforce/blip-image-captioning-base"
model = BlipForConditionalGeneration.from_pretrained(model_name)
processor = BlipProcessor.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


import requests
from PIL import Image
from io import BytesIO
import os

# Function to download an image using its URL and save it to a local path
# def download_image(image_url, save_path):
#     # Send a GET request to the image URL
#     response = requests.get(image_url)
    
#     # Check if the request was successful
#     if response.status_code == 200:
#         # Open the image from the response content
#         image = Image.open(BytesIO(response.content))
#         image.save(save_path)  # Save the image to the specified path
#         print(f"Image saved to {save_path}")
#     else:
#         print(f"Failed to download image from {image_url}")



# def process_sample_with_image(sample, image_dir):
#     # Construct the full path or URL for the image
#     image_id = sample["image"]  # Assuming the sample has an 'image_id' field
#     # image_filename = f"{image_id}.jpg"
#     image_url = f"http://images.cocodataset.org/train2017/{image_id}"  # If fetching remotely
    

#     # Generate a local save path for the image
#     save_path = os.path.join(image_dir, image_id)

#     # Download and save the image
#     download_image(image_url, save_path)

#     # Process the chosen and rejected responses
#     chosen = sample["output_2"]["value"] if sample["preference"] == 2 else sample["output_1"]["value"]
#     rejected = sample["output_1"]["value"] if sample["preference"] == 2 else sample["output_2"]["value"]

#     # Return the updated sample with chosen, rejected, and the local image path
#     return {"chosen": chosen, "rejected": rejected, "image_path": save_path}

# # Apply the function to the 'train' split
# dataset['train'] = dataset['train'].map(lambda x: process_sample_with_image(x, "./images" ))





def preprocess_function(example):
    image = Image.open(example["image_path"]).convert("RGB")
    inputs_chosen = processor(images=image, text=example["chosen"], return_tensors="pt", padding="max_length", truncation=True)
    inputs_rejected = processor(images=image, text=example["rejected"], return_tensors="pt", padding="max_length", truncation=True)
    return {
        "pixel_values": inputs_chosen["pixel_values"].squeeze(0),
        "input_ids_chosen": inputs_chosen["input_ids"].squeeze(0),
        "input_ids_rejected": inputs_rejected["input_ids"].squeeze(0),
    }

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Define reward model
class BlipRewardModel(nn.Module):
    def __init__(self, base_model):
        super(BlipRewardModel, self).__init__()
        self.blip = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)  # Predict a single reward score

    def forward(self, pixel_values, input_ids):
        outputs = self.blip(pixel_values=pixel_values, input_ids=input_ids)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # Use CLS token output
        reward = self.reward_head(last_hidden_state)
        return reward

# Initialize the reward model
reward_model = BlipRewardModel(model).to(device)

# Prepare DataLoader
train_dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)

# Define optimizer
optimizer = AdamW(reward_model.parameters(), lr=1e-5)

# Training loop
reward_model.train()
epochs = 3

for epoch in range(epochs):
    loop = tqdm(train_dataloader, leave=True)
    for batch in loop:
        optimizer.zero_grad()

        # Move batch to device
        pixel_values = batch["pixel_values"].to(device)
        input_ids_chosen = batch["input_ids_chosen"].to(device)
        input_ids_rejected = batch["input_ids_rejected"].to(device)

        # Forward pass
        reward_chosen = reward_model(pixel_values, input_ids_chosen)
        reward_rejected = reward_model(pixel_values, input_ids_rejected)

        # Compute loss (preference reward loss)
        loss = torch.nn.functional.relu(1.0 - reward_chosen + reward_rejected).mean()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update progress bar
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

# Save the reward model
reward_model_save_path = "./blip_reward_model"
torch.save(reward_model.state_dict(), reward_model_save_path)
print(f"Reward model saved to {reward_model_save_path}")
