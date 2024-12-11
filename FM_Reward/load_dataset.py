from datasets import Dataset

# Load the saved dataset
dataset = Dataset.load_from_disk("Foundation-Model-Project/FM_Reward/processed_llava_dataset")
# Save to CSV
dataset.to_csv("processed_llava_dataset.csv")
# Inspect the first entry
print(dataset[0])
