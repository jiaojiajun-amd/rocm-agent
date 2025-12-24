from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="jsonjiao/rocm-agent-traj-v1",
    repo_type="dataset",
    local_dir="./training_data_downloaded",
)

