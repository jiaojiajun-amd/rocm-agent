"""Visualize training data statistics and quality metrics."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress
from rich.table import Table

app = typer.Typer()
console = Console()


def load_data(input_file: Path) -> dict:
    with open(input_file) as f:
        return json.load(f)


def create_histogram(values: list[float], bins: int = 10, width: int = 50) -> str:
    """Create a text-based histogram."""
    if not values:
        return "No data"
    
    min_val = min(values)
    max_val = max(values)
    bin_width = (max_val - min_val) / bins if max_val > min_val else 1
    
    histogram = [0] * bins
    for val in values:
        if max_val == min_val:
            bin_idx = 0
        else:
            bin_idx = min(int((val - min_val) / bin_width), bins - 1)
        histogram[bin_idx] += 1
    
    max_count = max(histogram) if histogram else 1
    
    lines = []
    for i, count in enumerate(histogram):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        bar_length = int((count / max_count) * width)
        bar = "█" * bar_length
        lines.append(f"{bin_start:.2f}-{bin_end:.2f} │{bar} {count}")
    
    return "\n".join(lines)


@app.command()
def overview(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input training data JSON"),
):
    """Show a visual overview of training data."""
    console.print("\n[bold cyan]Loading training data...[/bold cyan]")
    data = load_data(input_file)
    
    examples = data.get("examples", [])
    metadata = data.get("metadata", {})
    
    # Header
    console.print(Panel.fit(
        f"[bold]Training Data Overview[/bold]\n"
        f"File: {input_file}\n"
        f"Model: {metadata.get('model_name', 'N/A')}",
        border_style="cyan"
    ))
    
    # Basic statistics
    total = len(examples)
    successful = sum(1 for ex in examples if ex.get("success", False))
    failed = total - successful
    
    stats_table = Table(title="📊 Basic Statistics", show_header=True, header_style="bold magenta")
    stats_table.add_column("Metric", style="cyan", width=25)
    stats_table.add_column("Value", style="green", width=20)
    stats_table.add_column("Visual", style="yellow")
    
    success_rate = successful / total if total > 0 else 0
    success_bar = "█" * int(success_rate * 20)
    stats_table.add_row("Total Examples", str(total), "")
    stats_table.add_row("Successful", str(successful), success_bar + f" {success_rate*100:.1f}%")
    stats_table.add_row("Failed", str(failed), "")
    
    console.print(stats_table)
    console.print()
    
    # Reward distribution
    if examples:
        rewards = [ex.get("reward", 0.0) for ex in examples]
        avg_reward = sum(rewards) / len(rewards)
        max_reward = max(rewards)
        min_reward = min(rewards)
        
        reward_table = Table(title="🎯 Reward Statistics", show_header=True, header_style="bold magenta")
        reward_table.add_column("Metric", style="cyan", width=25)
        reward_table.add_column("Value", style="green")
        
        reward_table.add_row("Average Reward", f"{avg_reward:.4f}")
        reward_table.add_row("Max Reward", f"{max_reward:.4f}")
        reward_table.add_row("Min Reward", f"{min_reward:.4f}")
        
        # Reward buckets
        high = sum(1 for r in rewards if r >= 0.8)
        medium = sum(1 for r in rewards if 0.5 <= r < 0.8)
        low = sum(1 for r in rewards if r < 0.5)
        
        reward_table.add_row("High Quality (≥0.8)", f"{high} ({high/total*100:.1f}%)")
        reward_table.add_row("Medium Quality (0.5-0.8)", f"{medium} ({medium/total*100:.1f}%)")
        reward_table.add_row("Low Quality (<0.5)", f"{low} ({low/total*100:.1f}%)")
        
        console.print(reward_table)
        console.print()
        
        # Reward histogram
        console.print(Panel(
            create_histogram(rewards, bins=10, width=40),
            title="📈 Reward Distribution",
            border_style="yellow"
        ))
        console.print()
    
    # Model efficiency
    if examples:
        model_calls = [ex.get("model_calls", 0) for ex in examples]
        avg_calls = sum(model_calls) / len(model_calls) if model_calls else 0
        total_calls = sum(model_calls)
        
        efficiency_table = Table(title="🤖 Model Efficiency", show_header=True, header_style="bold magenta")
        efficiency_table.add_column("Metric", style="cyan", width=25)
        efficiency_table.add_column("Value", style="green")
        
        efficiency_table.add_row("Total Model Calls", str(total_calls))
        efficiency_table.add_row("Average Calls/Task", f"{avg_calls:.1f}")
        
        # Trajectory lengths
        trajectory_lengths = [len(ex.get("messages", [])) for ex in examples]
        avg_length = sum(trajectory_lengths) / len(trajectory_lengths) if trajectory_lengths else 0
        max_length = max(trajectory_lengths) if trajectory_lengths else 0
        
        efficiency_table.add_row("Avg Trajectory Length", f"{avg_length:.1f} messages")
        efficiency_table.add_row("Max Trajectory Length", f"{max_length} messages")
        
        console.print(efficiency_table)
        console.print()
    
    # Quality summary
    if examples:
        successful_examples = [ex for ex in examples if ex.get("success", False)]
        if successful_examples:
            successful_rewards = [ex.get("reward", 0.0) for ex in successful_examples]
            avg_successful_reward = sum(successful_rewards) / len(successful_rewards)
            
            console.print(Panel(
                f"[bold green]✓ Success Rate: {success_rate*100:.1f}%[/bold green]\n"
                f"[bold yellow]⭐ Avg Reward (Successful): {avg_successful_reward:.4f}[/bold yellow]\n"
                f"[bold cyan]📝 Avg Trajectory Length: {avg_length:.1f}[/bold cyan]",
                title="🎉 Quality Summary",
                border_style="green"
            ))


@app.command()
def compare(
    file1: Path = typer.Option(..., "--file1", help="First data file"),
    file2: Path = typer.Option(..., "--file2", help="Second data file"),
    label1: str = typer.Option("Dataset 1", "--label1", help="Label for first dataset"),
    label2: str = typer.Option("Dataset 2", "--label2", help="Label for second dataset"),
):
    """Compare two training datasets side by side."""
    console.print("\n[bold cyan]Loading datasets...[/bold cyan]")
    data1 = load_data(file1)
    data2 = load_data(file2)
    
    examples1 = data1.get("examples", [])
    examples2 = data2.get("examples", [])
    
    # Comparison table
    table = Table(title="📊 Dataset Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column(label1, style="green", width=20)
    table.add_column(label2, style="yellow", width=20)
    table.add_column("Diff", style="blue", width=15)
    
    # Total examples
    total1 = len(examples1)
    total2 = len(examples2)
    table.add_row("Total Examples", str(total1), str(total2), f"{total2-total1:+d}")
    
    # Success rate
    success1 = sum(1 for ex in examples1 if ex.get("success", False))
    success2 = sum(1 for ex in examples2 if ex.get("success", False))
    rate1 = success1 / total1 if total1 > 0 else 0
    rate2 = success2 / total2 if total2 > 0 else 0
    table.add_row(
        "Success Rate",
        f"{rate1*100:.1f}%",
        f"{rate2*100:.1f}%",
        f"{(rate2-rate1)*100:+.1f}%"
    )
    
    # Average reward
    rewards1 = [ex.get("reward", 0.0) for ex in examples1]
    rewards2 = [ex.get("reward", 0.0) for ex in examples2]
    avg1 = sum(rewards1) / len(rewards1) if rewards1 else 0
    avg2 = sum(rewards2) / len(rewards2) if rewards2 else 0
    table.add_row(
        "Average Reward",
        f"{avg1:.4f}",
        f"{avg2:.4f}",
        f"{avg2-avg1:+.4f}"
    )
    
    # Model calls
    calls1 = [ex.get("model_calls", 0) for ex in examples1]
    calls2 = [ex.get("model_calls", 0) for ex in examples2]
    avg_calls1 = sum(calls1) / len(calls1) if calls1 else 0
    avg_calls2 = sum(calls2) / len(calls2) if calls2 else 0
    table.add_row(
        "Avg Model Calls",
        f"{avg_calls1:.1f}",
        f"{avg_calls2:.1f}",
        f"{avg_calls2-avg_calls1:+.1f}"
    )
    
    console.print(table)


@app.command()
def quality_report(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input training data JSON"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Save report to file"),
):
    """Generate a detailed quality report."""
    data = load_data(input_file)
    examples = data.get("examples", [])
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("TRAINING DATA QUALITY REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Overall statistics
    total = len(examples)
    successful = sum(1 for ex in examples if ex.get("success", False))
    rewards = [ex.get("reward", 0.0) for ex in examples]
    
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("-" * 70)
    report_lines.append(f"Total Examples: {total}")
    report_lines.append(f"Successful: {successful} ({successful/total*100:.1f}%)")
    report_lines.append(f"Failed: {total-successful}")
    report_lines.append("")
    
    # Quality tiers
    report_lines.append("QUALITY TIERS")
    report_lines.append("-" * 70)
    excellent = sum(1 for r in rewards if r >= 0.9)
    good = sum(1 for r in rewards if 0.7 <= r < 0.9)
    fair = sum(1 for r in rewards if 0.5 <= r < 0.7)
    poor = sum(1 for r in rewards if r < 0.5)
    
    report_lines.append(f"Excellent (≥0.9): {excellent} ({excellent/total*100:.1f}%)")
    report_lines.append(f"Good (0.7-0.9): {good} ({good/total*100:.1f}%)")
    report_lines.append(f"Fair (0.5-0.7): {fair} ({fair/total*100:.1f}%)")
    report_lines.append(f"Poor (<0.5): {poor} ({poor/total*100:.1f}%)")
    report_lines.append("")
    
    # Top performers
    report_lines.append("TOP 5 EXAMPLES")
    report_lines.append("-" * 70)
    sorted_examples = sorted(examples, key=lambda x: x.get("reward", 0.0), reverse=True)
    for i, ex in enumerate(sorted_examples[:5], 1):
        report_lines.append(
            f"{i}. {ex.get('instance_id', 'N/A')}: "
            f"reward={ex.get('reward', 0):.4f}, "
            f"calls={ex.get('model_calls', 0)}"
        )
    report_lines.append("")
    
    # Recommendations
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 70)
    
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    if avg_reward >= 0.8:
        report_lines.append("✓ Excellent data quality - ready for training")
    elif avg_reward >= 0.6:
        report_lines.append("! Good data quality - consider filtering low performers")
    else:
        report_lines.append("⚠ Low data quality - review data generation process")
    
    if successful / total < 0.7:
        report_lines.append("⚠ Low success rate - check agent configuration")
    
    report_lines.append("")
    report_lines.append("=" * 70)
    
    report = "\n".join(report_lines)
    
    console.print(report)
    
    if output_file:
        output_file.write_text(report)
        console.print(f"\n[green]✓ Report saved to {output_file}[/green]")


if __name__ == "__main__":
    app()

