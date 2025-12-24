#!/usr/bin/env python3
"""Re-process trajectory files to extract auxiliary calls (observation_reasoning, history_summarization).

This script reads existing trajectory files and generates new SFT data with auxiliary turns included.
"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

console = Console()


def extract_conversation_from_trajectory(trajectory: dict) -> dict:
    """Extract conversation with auxiliary turns from a trajectory file."""
    all_model_calls = trajectory.get("all_model_calls", [])
    
    if not all_model_calls:
        return {"system": "", "turns": [], "auxiliary_turns": []}
    
    main_queries = [c for c in all_model_calls if c.get("type") == "main_query"]
    auxiliary_calls = [c for c in all_model_calls if c.get("type") in ["observation_reasoning", "history_summarization"]]
    
    # Extract system prompt from first main query
    system_prompt = ""
    if main_queries:
        first_messages = main_queries[0].get("messages", [])
        for msg in first_messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
                break
    
    # Build main conversation turns
    turns = []
    for call in main_queries:
        messages = call.get("messages", [])
        response = call.get("response", {})
        user_input = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_input = msg.get("content", "")
                break
        assistant_output = response.get("content", "")
        if user_input and assistant_output:
            turns.append({"user": user_input, "assistant": assistant_output})
    
    # Build auxiliary turns
    auxiliary_turns = []
    for call in auxiliary_calls:
        call_type = call.get("type", "")
        messages = call.get("messages", [])
        response = call.get("response", {})
        
        aux_system = ""
        aux_user = ""
        for msg in messages:
            if msg.get("role") == "system":
                aux_system = msg.get("content", "")
            elif msg.get("role") == "user":
                aux_user = msg.get("content", "")
        
        aux_output = response.get("content", "")
        if aux_user and aux_output:
            auxiliary_turns.append({
                "type": call_type,
                "system": aux_system,
                "user": aux_user,
                "assistant": aux_output,
            })
    
    return {
        "system": system_prompt,
        "turns": turns,
        "auxiliary_turns": auxiliary_turns,
    }


def main(
    trajectory_dir: Path = typer.Argument(..., help="Directory containing trajectory JSON files"),
    output_file: Path = typer.Option(..., "--output", "-o", help="Output JSON file for SFT data"),
    filter_success: bool = typer.Option(True, "--filter-success/--no-filter-success", help="Only include successful examples"),
    min_reward: Optional[float] = typer.Option(None, "--min-reward", help="Minimum reward threshold"),
):
    """Re-process trajectory files to extract SFT data with auxiliary turns."""
    
    # Load index file
    index_file = trajectory_dir / "index.json"
    if not index_file.exists():
        console.print(f"[red]Index file not found: {index_file}[/red]")
        raise typer.Exit(1)
    
    index = json.loads(index_file.read_text())
    files = index.get("files", [])
    metadata = index.get("metadata", {})
    
    console.print(f"[cyan]Found {len(files)} trajectory files[/cyan]")
    
    conversations = []
    total_turns = 0
    total_auxiliary = 0
    skipped = 0
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), console=console) as progress:
        task = progress.add_task("Processing trajectories...", total=len(files))
        
        for filename in files:
            filepath = trajectory_dir / filename
            if not filepath.exists():
                progress.advance(task)
                continue
            
            trajectory = json.loads(filepath.read_text())
            
            # Apply filters
            if filter_success and not trajectory.get("success", False):
                skipped += 1
                progress.advance(task)
                continue
            
            reward = trajectory.get("reward", 0)
            if min_reward is not None and reward < min_reward:
                skipped += 1
                progress.advance(task)
                continue
            
            # Extract conversation
            conv = extract_conversation_from_trajectory(trajectory)
            
            if not conv.get("turns") and not conv.get("auxiliary_turns"):
                skipped += 1
                progress.advance(task)
                continue
            
            conversations.append({
                "instance_id": trajectory.get("instance_id"),
                "repeat_id": trajectory.get("repeat_id"),
                "system": conv.get("system", ""),
                "turns": conv.get("turns", []),
                "auxiliary_turns": conv.get("auxiliary_turns", []),
                "reward": reward,
                "success": trajectory.get("success", True),
                "num_turns": len(conv.get("turns", [])),
                "num_auxiliary": len(conv.get("auxiliary_turns", [])),
            })
            
            total_turns += len(conv.get("turns", []))
            total_auxiliary += len(conv.get("auxiliary_turns", []))
            
            progress.advance(task)
    
    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": metadata,
        "conversations": conversations,
        "summary": {
            "total_conversations": len(conversations),
            "total_turns": total_turns,
            "total_auxiliary_turns": total_auxiliary,
            "skipped": skipped,
        }
    }
    output_file.write_text(json.dumps(data, indent=2))
    
    # Display summary
    table = Table(title="Re-processing Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Files", str(len(files)))
    table.add_row("Processed", str(len(conversations)))
    table.add_row("Skipped", str(skipped))
    table.add_row("Main Turns", str(total_turns))
    table.add_row("Auxiliary Turns", str(total_auxiliary))
    
    console.print("\n")
    console.print(table)
    console.print(f"\n[bold green]✓ Saved to {output_file}[/bold green]")


if __name__ == "__main__":
    typer.run(main)

