import os
import json
from sklearn.metrics import accuracy_score, f1_score

# Function to process a single JSON file
def process_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    all_gt_answers = []
    all_model_answers = []
    correct_count = 0
    total_count = 0

    # Iterate over each question-answer block in the list
    for entry in data:
        gt_answer = entry.get('gt_answer', '').strip().lower()
        model_answer = entry.get('model_answer', '').strip().lower()

        # Check if the answers match (for accuracy)
        is_correct = gt_answer == model_answer
        correct_count += is_correct
        total_count += 1

        # Append the answers for F1 score calculation
        all_gt_answers.append(gt_answer)
        all_model_answers.append(model_answer)

    # Calculate accuracy for this file
    accuracy = correct_count / total_count if total_count > 0 else 0
    # Calculate F1 score for this file
    # f1 = f1_score(all_gt_answers, all_model_answers, average='binary', pos_label='yes')

    return accuracy, 0

# Function to process the entire directory
def process_directory(directory_path):
    # Iterate through all JSON files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            accuracy, f1 = process_json_file(file_path)
            
            # Print results for this file
            print(f"File: {filename} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

# Example usage:
directory_path = 'results_POPE'  # Change this to your directory path
process_directory(directory_path)
