from datasets import load_dataset

# Specify the save directory
save_dir = './data/MMHal-Bench'        # not necessary to create this directory

# Download the dataset
dataset = load_dataset('Shengcao1006/MMHal-Bench')   

# Save to disk
dataset.save_to_disk(save_dir)   

# Print the save directory where the dataset files are stored
print(f"Dataset files are stored in: {save_dir}")