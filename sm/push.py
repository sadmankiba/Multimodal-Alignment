from huggingface_hub import HfApi, Repository
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import Repository

api = HfApi()

repo_name = "dpo-checkpoint"

# api.create_repo(repo_name, private=False)
 
# Path to your local checkpoint directory
local_checkpoint_path = "./blip2-dpo/checkpoint-1350"
 
# Clone the repository to a local directory
repo_name = "laynad/dpo-checkpoint" # Replace with your username and repo name
repo = Repository(local_dir=local_checkpoint_path, clone_from=f"https://huggingface.co/{repo_name}")
 
# Add all files in the checkpoint directory and push
repo.push_to_hub(commit_message="Initial commit of DPO checkpoints")