#!/usr/bin/env python3
"""Convert SFT training data to LLaMA-Factory compatible formats.

Supports:
- ShareGPT format (multi-turn conversations)
- Alpaca format (single-turn instruction-following)
"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def convert_to_qwen(conversations: list[dict], filter_success: bool = True, min_reward: Optional[float] = None, include_auxiliary: bool = True) -> tuple[list[dict], list[dict]]:
    """Convert conversations to Qwen3 format (user/assistant roles).
    
    Returns:
    - main_data: main conversation data
    - auxiliary_data: observation_reasoning and history_summarization data
    """
    main_data = []
    auxiliary_data = []
    
    for conv in conversations:
        if filter_success and not conv.get("success", False):
            continue
        if min_reward is not None and conv.get("reward", 0) < min_reward:
            continue
        
        # Main conversation turns
        turns = conv.get("turns", [])
        if turns:
            messages = []
            system_content = conv.get("system", "")
            if system_content:
                messages.append({"role": "system", "content": system_content})
            
            for turn in turns:
                user_msg = turn.get("user", "")
                assistant_msg = turn.get("assistant", "")
                if user_msg and assistant_msg:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})
            
            if len(messages) > (1 if system_content else 0):
                entry = {
                    "messages": messages,
                    "instance_id": conv.get("instance_id"),
                    "repeat_id": conv.get("repeat_id"),
                    "reward": conv.get("reward", 0),
                    "type": "main",
                }
                main_data.append(entry)
        
        # Auxiliary turns (observation_reasoning, history_summarization)
        if include_auxiliary:
            aux_turns = conv.get("auxiliary_turns", [])
            for aux in aux_turns:
                aux_messages = []
                aux_system = aux.get("system", "")
                if aux_system:
                    aux_messages.append({"role": "system", "content": aux_system})
                aux_messages.append({"role": "user", "content": aux.get("user", "")})
                aux_messages.append({"role": "assistant", "content": aux.get("assistant", "")})
                
                aux_entry = {
                    "messages": aux_messages,
                    "instance_id": conv.get("instance_id"),
                    "repeat_id": conv.get("repeat_id"),
                    "reward": conv.get("reward", 0),
                    "type": aux.get("type", "auxiliary"),
                }
                auxiliary_data.append(aux_entry)
    
    return main_data, auxiliary_data


def convert_to_alpaca(conversations: list[dict], filter_success: bool = True, min_reward: Optional[float] = None) -> list[dict]:
    """Convert conversations to Alpaca format (each turn as separate instruction).
    
    Each turn becomes one instruction-input-output triplet.
    """
    alpaca_data = []
    
    for conv in conversations:
        if filter_success and not conv.get("success", False):
            continue
        if min_reward is not None and conv.get("reward", 0) < min_reward:
            continue
        
        system_content = conv.get("system", "You are a helpful assistant.")
        turns = conv.get("turns", [])
        
        for idx, turn in enumerate(turns):
            user_msg = turn.get("user", "")
            assistant_msg = turn.get("assistant", "")
            
            if user_msg and assistant_msg:
                alpaca_data.append({
                    "instruction": system_content,
                    "input": user_msg,
                    "output": assistant_msg,
                    "instance_id": conv.get("instance_id"),
                    "repeat_id": conv.get("repeat_id"),
                    "turn_idx": idx,
                    "reward": conv.get("reward", 0),
                })
    
    return alpaca_data


def main(
    input_file: Path = typer.Argument(..., help="Input SFT JSON file"),
    output_file: Path = typer.Option(..., "--output", "-o", help="Output JSON file"),
    format: str = typer.Option("sharegpt", "--format", "-f", help="Output format: sharegpt or alpaca"),
    filter_success: bool = typer.Option(True, "--filter-success/--no-filter-success", help="Only include successful samples"),
    min_reward: Optional[float] = typer.Option(None, "--min-reward", help="Minimum reward threshold"),
    include_auxiliary: bool = typer.Option(True, "--include-auxiliary/--no-auxiliary", help="Include auxiliary (observation_reasoning, history_summarization) data"),
    dataset_info: Optional[Path] = typer.Option(None, "--dataset-info", help="Generate dataset_info.json for LLaMA-Factory"),
):
    """Convert SFT training data to LLaMA-Factory compatible format."""
    
    console.print(f"[cyan]Loading {input_file}...[/cyan]")
    data = json.loads(input_file.read_text())
    
    # Support both old format (samples) and new format (conversations)
    if "conversations" in data:
        conversations = data.get("conversations", [])
        console.print(f"  → {len(conversations)} conversations loaded (new format)")
    elif "samples" in data:
        # Old format compatibility - convert samples to conversations
        samples = data.get("samples", [])
        console.print(f"  → {len(samples)} samples loaded (old format, converting...)")
        from collections import defaultdict
        grouped = defaultdict(list)
        for s in samples:
            key = (s.get("instance_id"), s.get("repeat_id"))
            grouped[key].append(s)
        
        conversations = []
        for (instance_id, repeat_id), group in grouped.items():
            group.sort(key=lambda x: x.get("sample_idx", 0))
            system_content = ""
            first_messages = group[0].get("messages", []) if group else []
            for msg in first_messages:
                if msg.get("role") == "system":
                    system_content = msg.get("content", "")
                    break
            
            turns = []
            for s in group:
                messages = s.get("messages", [])
                response = s.get("response", {})
                user_msg = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        user_msg = msg.get("content", "")
                        break
                assistant_msg = response.get("content", "")
                if user_msg and assistant_msg:
                    turns.append({"user": user_msg, "assistant": assistant_msg})
            
            if turns:
                conversations.append({
                    "instance_id": instance_id,
                    "repeat_id": repeat_id,
                    "system": system_content,
                    "turns": turns,
                    "reward": group[-1].get("reward", 0),
                    "success": group[-1].get("success", True),
                })
        console.print(f"  → Converted to {len(conversations)} conversations")
    else:
        console.print("[red]Unknown data format[/red]")
        raise typer.Exit(1)
    
    auxiliary_data = []
    if format == "sharegpt":
        converted, auxiliary_data = convert_to_qwen(conversations, filter_success, min_reward, include_auxiliary)
        console.print(f"[green]Converted to Qwen3 format: {len(converted)} conversations, {len(auxiliary_data)} auxiliary[/green]")
    elif format == "qwen":
        converted, auxiliary_data = convert_to_qwen(conversations, filter_success, min_reward, include_auxiliary)
        console.print(f"[green]Converted to Qwen3 format: {len(converted)} conversations, {len(auxiliary_data)} auxiliary[/green]")
    elif format == "alpaca":
        converted = convert_to_alpaca(conversations, filter_success, min_reward)
        console.print(f"[green]Converted to Alpaca format: {len(converted)} samples[/green]")
    else:
        console.print(f"[red]Unknown format: {format}. Use 'sharegpt' or 'alpaca'[/red]")
        raise typer.Exit(1)
    
    # Save main converted data
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(converted, indent=2, ensure_ascii=False))
    console.print(f"[bold green]✓ Main data saved to {output_file}[/bold green]")
    
    # Save auxiliary data if present
    if auxiliary_data and include_auxiliary:
        aux_output = output_file.parent / f"{output_file.stem}_auxiliary.json"
        aux_output.write_text(json.dumps(auxiliary_data, indent=2, ensure_ascii=False))
        console.print(f"[bold green]✓ Auxiliary data saved to {aux_output}[/bold green]")
    
    # Generate dataset_info.json if requested
    if dataset_info:
        dataset_name = output_file.stem
        if format in ["sharegpt", "qwen"]:
            info = {
                dataset_name: {
                    "file_name": str(output_file.name),
                    "formatting": "sharegpt",
                    "columns": {"messages": "messages"}
                }
            }
        else:
            info = {
                dataset_name: {
                    "file_name": str(output_file.name),
                    "formatting": format,
                    "columns": {}
                }
            }
        if format == "alpaca":
            info[dataset_name]["columns"] = {
                "prompt": "instruction",
                "query": "input", 
                "response": "output",
            }
        
        # Add auxiliary dataset info
        if auxiliary_data and include_auxiliary:
            aux_name = f"{dataset_name}_auxiliary"
            info[aux_name] = {
                "file_name": f"{output_file.stem}_auxiliary.json",
                "formatting": "sharegpt",
                "columns": {"messages": "messages"}
            }
        
        dataset_info.parent.mkdir(parents=True, exist_ok=True)
        
        # Merge with existing if present
        if dataset_info.exists():
            existing = json.loads(dataset_info.read_text())
            existing.update(info)
            info = existing
        
        dataset_info.write_text(json.dumps(info, indent=2))
        console.print(f"[bold green]✓ Dataset info saved to {dataset_info}[/bold green]")
    
    # Display summary
    table = Table(title="Conversion Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Input Conversations", str(len(conversations)))
    table.add_row("Output Format", format)
    table.add_row("Main Conversations", str(len(converted)))
    if format in ["sharegpt", "qwen"]:
        total_turns = sum((len(c["messages"]) - 1) // 2 for c in converted)
        table.add_row("Main Turns", str(total_turns))
    table.add_row("Auxiliary Samples", str(len(auxiliary_data)))
    if min_reward is not None:
        table.add_row("Min Reward Filter", str(min_reward))
    table.add_row("Filter Success", str(filter_success))
    
    console.print("\n")
    console.print(table)


if __name__ == "__main__":
    typer.run(main)
