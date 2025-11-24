"""Process and filter training data for agent continuous pretraining.

This script provides utilities to:
- Filter successful examples
- Extract dialogue trajectories
- Format data for different training frameworks
- Generate statistics and reports
"""

import json
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


def load_training_data(input_file: Path) -> dict:
    with open(input_file) as f:
        return json.load(f)


def filter_successful_examples(data: dict, min_reward: float = 0.0) -> list[dict]:
    return [
        ex for ex in data["examples"]
        if ex["success"] and ex["reward"] >= min_reward
    ]


def format_for_supervised_finetuning(examples: list[dict]) -> list[dict]:
    """Format examples for supervised fine-tuning (SFT).
    
    Returns list of conversations in format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ],
        "metadata": {...}
    }
    """
    formatted = []
    for ex in examples:
        formatted.append({
            "messages": ex["messages"],
            "metadata": {
                "instance_id": ex["instance_id"],
                "reward": ex["reward"],
                "speedup": ex["speedup"],
                "model_calls": ex["model_calls"],
            }
        })
    return formatted


def format_for_trajectory_learning(examples: list[dict]) -> list[dict]:
    """Format examples for trajectory-based learning.
    
    Includes full trajectories with actions, observations, and rewards.
    """
    trajectories = []
    for ex in examples:
        trajectory = {
            "task": ex["problem_statement"],
            "instance_id": ex["instance_id"],
            "messages": ex["messages"],
            "git_diff": ex["git_diff"],
            "final_reward": ex["reward"],
            "success": ex["success"],
            "metadata": ex.get("metadata", {}),
        }
        trajectories.append(trajectory)
    return trajectories


def extract_action_observation_pairs(messages: list[dict]) -> list[dict]:
    """Extract action-observation pairs from message history."""
    pairs = []
    i = 0
    while i < len(messages):
        if messages[i]["role"] == "assistant":
            action = messages[i]["content"]
            observation = messages[i + 1]["content"] if i + 1 < len(messages) else None
            if observation:
                pairs.append({
                    "action": action,
                    "observation": observation,
                })
            i += 2
        else:
            i += 1
    return pairs


def generate_statistics(data: dict) -> dict:
    examples = data["examples"]
    successful = [ex for ex in examples if ex["success"]]
    
    stats = {
        "total_examples": len(examples),
        "successful_examples": len(successful),
        "success_rate": len(successful) / len(examples) if examples else 0,
        "average_reward": sum(ex["reward"] for ex in examples) / len(examples) if examples else 0,
        "average_reward_successful": sum(ex["reward"] for ex in successful) / len(successful) if successful else 0,
        "average_model_calls": sum(ex["model_calls"] for ex in examples) / len(examples) if examples else 0,
        "total_model_calls": sum(ex["model_calls"] for ex in examples),
        "average_trajectory_length": sum(len(ex["messages"]) for ex in successful) / len(successful) if successful else 0,
    }
    
    reward_distribution = {}
    for ex in examples:
        reward_bucket = int(ex["reward"] * 10) / 10
        reward_distribution[reward_bucket] = reward_distribution.get(reward_bucket, 0) + 1
    stats["reward_distribution"] = reward_distribution
    
    return stats


@app.command()
def analyze(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input training data JSON"),
):
    """Analyze training data and show statistics."""
    console.print(f"[cyan]Loading training data from {input_file}...[/cyan]\n")
    data = load_training_data(input_file)
    
    metadata = data.get("metadata", {})
    if metadata:
        console.print("[bold blue]Metadata[/bold blue]")
        meta_table = Table(show_header=False)
        meta_table.add_column("Key", style="cyan")
        meta_table.add_column("Value", style="yellow")
        for key, value in metadata.items():
            meta_table.add_row(key.replace("_", " ").title(), str(value))
        console.print(meta_table)
        console.print()
    
    stats = generate_statistics(data)
    
    stats_table = Table(title="Training Data Statistics", show_header=True, header_style="bold magenta")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Total Examples", str(stats["total_examples"]))
    stats_table.add_row("Successful Examples", f"{stats['successful_examples']} ({stats['success_rate']*100:.1f}%)")
    stats_table.add_row("Average Reward", f"{stats['average_reward']:.4f}")
    stats_table.add_row("Avg Reward (Successful)", f"{stats['average_reward_successful']:.4f}")
    stats_table.add_row("Average Model Calls", f"{stats['average_model_calls']:.1f}")
    stats_table.add_row("Total Model Calls", str(stats["total_model_calls"]))
    stats_table.add_row("Avg Trajectory Length", f"{stats['average_trajectory_length']:.1f}")
    
    console.print(stats_table)
    console.print()
    
    if stats["reward_distribution"]:
        console.print("[bold blue]Reward Distribution[/bold blue]")
        dist_table = Table(show_header=True)
        dist_table.add_column("Reward Range", style="cyan")
        dist_table.add_column("Count", style="green")
        for reward, count in sorted(stats["reward_distribution"].items()):
            dist_table.add_row(f"{reward:.1f}", str(count))
        console.print(dist_table)


@app.command()
def filter_data(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input training data JSON"),
    output_file: Path = typer.Option(..., "--output", "-o", help="Output filtered JSON"),
    min_reward: float = typer.Option(0.0, "--min-reward", help="Minimum reward threshold"),
    successful_only: bool = typer.Option(True, "--successful-only", help="Only keep successful examples"),
):
    """Filter training data based on criteria."""
    console.print(f"[cyan]Loading training data from {input_file}...[/cyan]")
    data = load_training_data(input_file)
    
    examples = data["examples"]
    console.print(f"Original examples: {len(examples)}")
    
    if successful_only:
        examples = [ex for ex in examples if ex["success"]]
        console.print(f"After filtering successful: {len(examples)}")
    
    if min_reward > 0:
        examples = [ex for ex in examples if ex["reward"] >= min_reward]
        console.print(f"After reward filter (>= {min_reward}): {len(examples)}")
    
    filtered_data = {
        "metadata": data.get("metadata", {}),
        "filter_criteria": {
            "min_reward": min_reward,
            "successful_only": successful_only,
        },
        "examples": examples,
        "summary": {
            "total_examples": len(examples),
            "original_count": len(data["examples"]),
            "filter_rate": len(examples) / len(data["examples"]) if data["examples"] else 0,
        }
    }
    
    with open(output_file, "w") as f:
        json.dump(filtered_data, f, indent=2)
    
    console.print(f"\n[bold green]✓ Filtered data saved to {output_file}[/bold green]")
    console.print(f"Kept {len(examples)} out of {len(data['examples'])} examples")


@app.command()
def export_sft(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input training data JSON"),
    output_file: Path = typer.Option(..., "--output", "-o", help="Output SFT format JSONL"),
    min_reward: float = typer.Option(0.0, "--min-reward", help="Minimum reward threshold"),
):
    """Export data in supervised fine-tuning (SFT) format."""
    console.print(f"[cyan]Loading training data from {input_file}...[/cyan]")
    data = load_training_data(input_file)
    
    examples = filter_successful_examples(data, min_reward)
    console.print(f"Using {len(examples)} successful examples (min_reward >= {min_reward})")
    
    formatted = format_for_supervised_finetuning(examples)
    
    with open(output_file, "w") as f:
        for item in formatted:
            f.write(json.dumps(item) + "\n")
    
    console.print(f"[bold green]✓ SFT data saved to {output_file}[/bold green]")
    console.print(f"Format: JSONL with {len(formatted)} conversations")


@app.command()
def export_trajectory(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input training data JSON"),
    output_file: Path = typer.Option(..., "--output", "-o", help="Output trajectory format JSON"),
    min_reward: float = typer.Option(0.0, "--min-reward", help="Minimum reward threshold"),
):
    """Export data in trajectory learning format."""
    console.print(f"[cyan]Loading training data from {input_file}...[/cyan]")
    data = load_training_data(input_file)
    
    examples = filter_successful_examples(data, min_reward)
    console.print(f"Using {len(examples)} successful examples (min_reward >= {min_reward})")
    
    trajectories = format_for_trajectory_learning(examples)
    
    output_data = {
        "metadata": data.get("metadata", {}),
        "trajectories": trajectories,
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    console.print(f"[bold green]✓ Trajectory data saved to {output_file}[/bold green]")
    console.print(f"Format: JSON with {len(trajectories)} trajectories")


@app.command()
def show_example(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input training data JSON"),
    index: int = typer.Option(0, "--index", help="Example index to show"),
):
    """Show a detailed view of a single training example."""
    data = load_training_data(input_file)
    examples = data["examples"]
    
    if index >= len(examples):
        console.print(f"[red]Error: Index {index} out of range (0-{len(examples)-1})[/red]")
        raise typer.Exit(1)
    
    ex = examples[index]
    
    console.print(f"\n[bold blue]Example {index}: {ex['instance_id']}[/bold blue]\n")
    
    info_table = Table(show_header=False)
    info_table.add_column("Field", style="cyan")
    info_table.add_column("Value", style="yellow")
    info_table.add_row("Instance ID", ex["instance_id"])
    info_table.add_row("Success", "✓ Yes" if ex["success"] else "✗ No")
    info_table.add_row("Reward", f"{ex['reward']:.4f}")
    info_table.add_row("Speedup", f"{ex['speedup']:.4f}")
    info_table.add_row("Model Calls", str(ex["model_calls"]))
    info_table.add_row("Exit Status", ex["exit_status"])
    info_table.add_row("Message Count", str(len(ex["messages"])))
    console.print(info_table)
    console.print()
    
    console.print("[bold cyan]Problem Statement:[/bold cyan]")
    console.print(ex["problem_statement"][:500] + "..." if len(ex["problem_statement"]) > 500 else ex["problem_statement"])
    console.print()
    
    console.print("[bold cyan]Message History:[/bold cyan]")
    for i, msg in enumerate(ex["messages"][:10]):
        role = msg["role"]
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        console.print(f"[{i}] {role.upper()}: {content}")
    
    if len(ex["messages"]) > 10:
        console.print(f"... and {len(ex['messages']) - 10} more messages")
    console.print()
    
    if ex.get("git_diff"):
        console.print("[bold cyan]Git Diff:[/bold cyan]")
        diff = ex["git_diff"]
        console.print(diff[:500] + "..." if len(diff) > 500 else diff)
    else:
        console.print("[yellow]No git diff available[/yellow]")


if __name__ == "__main__":
    app()

