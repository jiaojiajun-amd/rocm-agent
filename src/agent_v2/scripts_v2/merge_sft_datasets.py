#!/usr/bin/env python3
"""Merge multiple SFT training datasets into one."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def main(
    inputs: list[Path] = typer.Argument(..., help="Input SFT JSON files to merge"),
    output: Path = typer.Option(..., "--output", "-o", help="Output merged JSON file"),
    filter_success: bool = typer.Option(False, "--filter-success", help="Only include successful samples"),
    min_reward: Optional[float] = typer.Option(None, "--min-reward", help="Minimum reward threshold"),
):
    """Merge multiple SFT training datasets into one."""
    
    all_samples = []
    all_metadata = []
    
    for input_path in inputs:
        console.print(f"[cyan]Loading {input_path}...[/cyan]")
        data = json.loads(input_path.read_text())
        
        samples = data.get("samples", [])
        metadata = data.get("metadata", {})
        
        # Apply filters
        if filter_success:
            samples = [s for s in samples if s.get("success", False)]
        if min_reward is not None:
            samples = [s for s in samples if s.get("reward", 0) >= min_reward]
        
        console.print(f"  → {len(samples)} samples after filtering")
        
        all_samples.extend(samples)
        all_metadata.append({
            "source": str(input_path),
            "samples_count": len(samples),
            **metadata
        })
    
    # Deduplicate by (instance_id, repeat_id, sample_idx)
    seen = set()
    unique_samples = []
    for s in all_samples:
        key = (s.get("instance_id"), s.get("repeat_id"), s.get("sample_idx"))
        if key not in seen:
            seen.add(key)
            unique_samples.append(s)
    
    duplicates_removed = len(all_samples) - len(unique_samples)
    
    # Build merged output
    merged = {
        "metadata": {
            "merged_from": all_metadata,
            "total_sources": len(inputs),
        },
        "samples": unique_samples,
        "summary": {
            "total_samples": len(unique_samples),
            "duplicates_removed": duplicates_removed,
            "successful_samples": sum(1 for s in unique_samples if s.get("success", False)),
            "average_reward": sum(s.get("reward", 0) for s in unique_samples) / len(unique_samples) if unique_samples else 0,
        }
    }
    
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2))
    
    # Display summary
    table = Table(title="Merge Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Input Files", str(len(inputs)))
    table.add_row("Total Samples", str(len(unique_samples)))
    table.add_row("Duplicates Removed", str(duplicates_removed))
    table.add_row("Successful Samples", str(merged["summary"]["successful_samples"]))
    table.add_row("Average Reward", f"{merged['summary']['average_reward']:.4f}")
    
    console.print("\n")
    console.print(table)
    console.print(f"\n[bold green]✓ Merged data saved to {output}[/bold green]")


if __name__ == "__main__":
    typer.run(main)

