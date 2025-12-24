from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="/home/jiajjiao/rocm-agent/training_data",
    repo_id="jsonjiao/rocm-agent-traj-v1",
    repo_type="dataset",
    ignore_patterns=["*.log"],
)