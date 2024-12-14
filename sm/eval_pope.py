import os
import pandas as pd

# Directory containing your CSV files
directory = '../data'

# Initialize an empty list to hold DataFrames
dfs = []

# Loop through the directory and read matching CSV files
for filename in os.listdir(directory):
    if filename.startswith("pope_blip2-base_responses") and filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Ensure that 'answer' and 'response' columns are lowercase and stripped of whitespace
combined_df['answer_clean'] = combined_df['answer'].str.lower().str.strip()
combined_df['response_clean'] = combined_df['response'].str.lower().str.strip()

# Compare the cleaned 'answer' and 'response' columns and count the matches
matching_count = (combined_df['answer_clean'] == combined_df['response_clean']).sum()

# Calculate the percentage
total_count = len(combined_df)
percentage_matching = (matching_count / total_count) * 100

# Display the result
print(f'Percentage of cases where the answer and response are the same: {percentage_matching:.2f}%')


# Optionally, you can save the combined DataFrame to a new CSV file
# combined_df.to_csv('/path/to/your/combined_file.csv', index=False)

# Display the combined DataFrame
# print(combined_df)
