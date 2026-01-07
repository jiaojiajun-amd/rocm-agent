#!/usr/bin/env python3
"""Upload Qwen format training data to Huggingface Hub."""

import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login
import typer
from rich.console import Console
from rich.table import Table

console = Console()


def load_qwen_data(file_path: Path) -> list[dict]:
    """Load Qwen format JSON data."""
    data = json.loads(file_path.read_text())
    return data


def main(
    hf_token: str = typer.Option(..., "--token", "-t", envvar="HF_TOKEN", help="Huggingface API token"),
    repo_name: str = typer.Option("rocm-agent-sft", "--repo", "-r", help="Dataset repository name"),
    data_dir: Path = typer.Option(
        Path("/home/jiajjiao/rocm-agent/training_data_v2/llamafactory"),
        "--data-dir", "-d",
        help="Directory containing the Qwen format JSON files"
    ),
    private: bool = typer.Option(True, "--private/--public", help="Make the dataset private"),
):
    """Upload Qwen format SFT data to Huggingface Hub."""
    
    # Login to Huggingface
    console.print("[cyan]Logging in to Huggingface...[/cyan]")
    login(token=hf_token)
    
    api = HfApi()
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{repo_name}"
    
    console.print(f"[green]Logged in as: {username}[/green]")
    console.print(f"[cyan]Will upload to: {repo_id}[/cyan]")
    
    # Find all Qwen format files
    main_files = sorted(data_dir.glob("*_v2_qwen.json"))
    aux_files = sorted(data_dir.glob("*_v2_qwen_auxiliary.json"))
    
    console.print(f"\n[cyan]Found {len(main_files)} main files and {len(aux_files)} auxiliary files[/cyan]")
    
    # Load and merge all main conversation data
    console.print("\n[cyan]Loading main conversation data...[/cyan]")
    all_main_data = []
    for f in main_files:
        data = load_qwen_data(f)
        console.print(f"  → {f.name}: {len(data)} samples")
        all_main_data.extend(data)
    
    # Load and merge all auxiliary data
    console.print("\n[cyan]Loading auxiliary data...[/cyan]")
    all_aux_data = []
    for f in aux_files:
        data = load_qwen_data(f)
        console.print(f"  → {f.name}: {len(data)} samples")
        all_aux_data.extend(data)
    
    # Create datasets
    console.print("\n[cyan]Creating datasets...[/cyan]")
    
    # For main data, we need to flatten the messages for the Dataset
    main_dataset = Dataset.from_list(all_main_data)
    aux_dataset = Dataset.from_list(all_aux_data)
    
    # Combine into DatasetDict
    dataset_dict = DatasetDict({
        "train": main_dataset,
        "auxiliary": aux_dataset,
    })
    
    console.print(f"  → Main train split: {len(main_dataset)} samples")
    console.print(f"  → Auxiliary split: {len(aux_dataset)} samples")
    
    # Upload to Huggingface
    console.print(f"\n[cyan]Uploading to {repo_id}...[/cyan]")
    
    dataset_dict.push_to_hub(
        repo_id,
        private=private,
        commit_message="Upload ROCm Agent SFT training data (Qwen format)",
    )
    
    # Display summary
    table = Table(title="Upload Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Repository", repo_id)
    table.add_row("Main Samples", str(len(all_main_data)))
    table.add_row("Auxiliary Samples", str(len(all_aux_data)))
    table.add_row("Private", str(private))
    table.add_row("URL", f"https://huggingface.co/datasets/{repo_id}")
    
    console.print("\n")
    console.print(table)
    console.print(f"\n[bold green]✓ Successfully uploaded to https://huggingface.co/datasets/{repo_id}[/bold green]")


if __name__ == "__main__":
    typer.run(main)

