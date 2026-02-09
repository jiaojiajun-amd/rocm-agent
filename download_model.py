from huggingface_hub import snapshot_download

repo_id = "zai-org/GLM-4.6"
# repo_id = "moonshotai/Kimi-K2-Thinking"


model_path = snapshot_download(
    repo_id=repo_id,
    cache_dir="/models/",
    resume_download=True,  # 断点续传
    local_files_only=False
)
print(f"Model downloaded to: {model_path}")