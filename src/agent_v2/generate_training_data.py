"""Generate training data for agent continuous pretraining."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import typer
import yaml
from jinja2 import StrictUndefined, Template
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from eval_utils import evaluate_info
from minisweagent.agents.mini import MiniAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments.docker_remote import RemoteDockerEnvironment
from minisweagent.models.litellm_amd_model import LiteLLMAMDModel
from minisweagent.utils.log import add_file_handler, logger

console = Console()


@dataclass
class TrainingExample:
    instance_id: str
    repeat_id: int
    problem_statement: str
    conversation: dict  # Main training data: {system, turns} - one conversation per example
    git_diff: Optional[str]
    exit_status: str
    reward: float
    speedup: float
    success: bool
    model_calls: int
    evaluation_info: dict
    # Detailed data for trajectory analysis (saved separately)
    full_messages: list[dict] = field(default_factory=list)
    all_model_calls: list[dict] = field(default_factory=list)
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def extract_sft_conversation(all_model_calls: list[dict]) -> dict:
    """Extract conversations from model calls, including auxiliary calls.
    
    Returns a dict with:
    - system: system prompt (from first call)
    - turns: list of main conversation turns
    - auxiliary_turns: list of observation_reasoning and history_summarization turns
    """
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
    
    # Build auxiliary turns (observation/history summarization)
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


def get_rocm_environment(config: dict, instance: dict, server_url: str) -> RemoteDockerEnvironment:
    env_config = config.setdefault("environment", {})
    env_config["image"] = instance.get("image_name", "rocm-lib")
    env = RemoteDockerEnvironment(server_url=server_url, **env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        out = env.execute(Template(startup_command, undefined=StrictUndefined).render(**instance))
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def get_git_diff(env, instance_id: str) -> Optional[str]:
    for cmd in ["git diff --cached", "git diff", "git diff HEAD"]:
        result = env.execute(cmd)
        if result["returncode"] == 0 and result["output"].strip():
            return result["output"]
    return None


async def generate_single_example(
    instance: dict,
    model: LiteLLMAMDModel,
    config: dict,
    docker_server_url: str,
    eval_server_url: str,
    repeat_id: int = 0,
) -> TrainingExample:
    instance_id = instance["instance_id"]
    try:
        env = get_rocm_environment(config, instance, docker_server_url)
        agent = MiniAgent(model=model, env=env, **config.get("agent", {}))
        exit_status, result = agent.run(instance["problem_statement"])
        
        reward, speedup, evaluation_info = await evaluate_info(
            exit_status, result, agent.env.container_id, instance_id,
            instance.get("dataset_name", "SWE-bench/SWE-bench_Lite"),
            instance.get("split", "test"), eval_server_url,
        )
        
        all_model_calls = agent.get_all_model_calls()
        return TrainingExample(
            instance_id=instance_id,
            repeat_id=repeat_id,
            problem_statement=instance["problem_statement"],
            conversation=extract_sft_conversation(all_model_calls),
            git_diff=get_git_diff(agent.env, instance_id),
            exit_status=exit_status,
            reward=float(reward),
            speedup=float(speedup),
            success=True,
            model_calls=model.get_template_vars().get("n_model_calls", 0),
            evaluation_info=evaluation_info,
            full_messages=agent.get_full_messages(),
            all_model_calls=all_model_calls,
            metadata={"dataset_name": instance.get("dataset_name"), "split": instance.get("split"), "image_name": instance.get("image_name")},
        )
    except Exception as e:
        logger.exception(f"Error generating data for {instance_id}_repeat{repeat_id}: {e}")
        return TrainingExample(
            instance_id=instance_id, repeat_id=repeat_id, problem_statement=instance.get("problem_statement", ""),
            conversation={"system": "", "turns": []}, git_diff=None, exit_status="error", reward=0.0, speedup=0.0, success=False,
            model_calls=0, evaluation_info={"meta": {"success": False, "error": str(e)}}, error=str(e),
        )


def save_sft_data(examples: list[TrainingExample], output_path: Path, metadata: dict):
    """Save SFT training data in conversation format (more efficient, no redundant context)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Each example is one conversation with multiple turns
    conversations = []
    total_turns = 0
    total_auxiliary = 0
    for ex in examples:
        conv = ex.conversation
        if not conv.get("turns") and not conv.get("auxiliary_turns"):
            continue
        conversations.append({
            "instance_id": ex.instance_id,
            "repeat_id": ex.repeat_id,
            "system": conv.get("system", ""),
            "turns": conv.get("turns", []),
            "auxiliary_turns": conv.get("auxiliary_turns", []),
            "reward": ex.reward,
            "success": ex.success,
            "num_turns": len(conv.get("turns", [])),
            "num_auxiliary": len(conv.get("auxiliary_turns", [])),
        })
        total_turns += len(conv.get("turns", []))
        total_auxiliary += len(conv.get("auxiliary_turns", []))
    
    data = {
        "metadata": metadata,
        "conversations": conversations,
        "summary": {
            "total_conversations": len(conversations),
            "total_turns": total_turns,
            "total_auxiliary_turns": total_auxiliary,
            "successful": sum(1 for ex in examples if ex.success),
            "failed": sum(1 for ex in examples if not ex.success),
            "average_reward": sum(ex.reward for ex in examples) / len(examples) if examples else 0,
            "total_model_calls": sum(ex.model_calls for ex in examples),
        }
    }
    output_path.write_text(json.dumps(data, indent=2))
    logger.info(f"SFT training data saved to {output_path}")


def save_trajectory_data(examples: list[TrainingExample], output_dir: Path, metadata: dict):
    """Save detailed trajectory data (full_messages, all_model_calls, etc.) to separate folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each example as a separate file for easier inspection
    for ex in examples:
        filename = f"{ex.instance_id}_repeat{ex.repeat_id}.json"
        filepath = output_dir / filename
        trajectory = {
            "instance_id": ex.instance_id,
            "repeat_id": ex.repeat_id,
            "problem_statement": ex.problem_statement,
            "full_messages": ex.full_messages,
            "all_model_calls": ex.all_model_calls,
            "git_diff": ex.git_diff,
            "exit_status": ex.exit_status,
            "reward": ex.reward,
            "speedup": ex.speedup,
            "success": ex.success,
            "model_calls": ex.model_calls,
            "evaluation_info": ex.evaluation_info,
            "error": ex.error,
            "metadata": ex.metadata,
        }
        filepath.write_text(json.dumps(trajectory, indent=2))
    
    # Save index file
    index = {
        "metadata": metadata,
        "files": [f"{ex.instance_id}_repeat{ex.repeat_id}.json" for ex in examples],
        "summary": {
            "total_examples": len(examples),
            "successful": sum(1 for ex in examples if ex.success),
            "failed": sum(1 for ex in examples if not ex.success),
            "average_reward": sum(ex.reward for ex in examples) / len(examples) if examples else 0,
        }
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2))
    logger.info(f"Trajectory data saved to {output_dir}")


def main(
    dataset_file: Path = typer.Option(..., "--dataset", "-d", help="Dataset JSON file"),
    api_key: str = typer.Option(..., "--api-key", help="API key"),
    model_name: str = typer.Option("Qwen/Qwen3-8B", "--model", "-m", help="Model name"),
    docker_server: str = typer.Option("10.67.77.184:9527", "--docker-server", help="Docker server"),
    eval_server: str = typer.Option("10.67.77.184:9528", "--eval-server", help="Eval server"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file"),
    output_file: Path = typer.Option(..., "--output", "-o", help="Output JSON file for SFT data"),
    max_tasks: Optional[int] = typer.Option(None, "--max-tasks", help="Max number of tasks"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Log file path"),
    temperature: float = typer.Option(1.0, "--temperature", help="Sampling temperature"),
    max_tokens: int = typer.Option(8000, "--max-tokens", help="Max tokens"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers"),
    repeats: int = typer.Option(1, "--repeats", "-r", help="Number of times to repeat the entire dataset"),
):
    """Generate training data: outer loop = repeats, inner loop = dataset instances.
    
    Outputs:
    - Main output (-o): SFT training data with real model input/output pairs
    - Trajectory folder: Detailed data in {output}_trajectories/ folder
    """
    if log_file:
        add_file_handler(log_file)
    
    config_path = config_file or get_config_path(Path(builtin_config_dir / "mini.yaml"))
    config = yaml.safe_load(config_path.read_text())
    
    dataset = json.loads(dataset_file.read_text())
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a JSON array")
    if max_tasks:
        dataset = dataset[:max_tasks]
    
    total_tasks = len(dataset)
    total_samples = total_tasks * repeats
    
    # Trajectory folder path
    trajectory_dir = output_file.parent / f"{output_file.stem}_trajectories"
    
    console.print(f"[bold blue]Generating training data[/bold blue]")
    console.print(f"Model: {model_name}, Tasks: {total_tasks}, Repeats: {repeats}, Total: {total_samples}, Workers: {workers}")
    console.print(f"SFT output: {output_file}")
    console.print(f"Trajectory folder: {trajectory_dir}")
    
    docker_server_url, eval_server_url = f"http://{docker_server}", f"http://{eval_server}"
    
    def worker(args: tuple[dict, int]) -> TrainingExample:
        instance, repeat_id = args
        model = LiteLLMAMDModel(model_name=model_name, api_key=api_key, temperature=temperature, max_tokens=max_tokens, top_k=40)
        return asyncio.run(generate_single_example(instance, model, config, docker_server_url, eval_server_url, repeat_id))
    
    examples: list[TrainingExample] = []
    tasks_to_run = [(instance, repeat_id) for repeat_id in range(repeats) for instance in dataset]
    
    generation_metadata = {
        "model_name": model_name, "temperature": temperature, "max_tokens": max_tokens,
        "config_file": str(config_path), "dataset_file": str(dataset_file),
        "workers": workers, "repeats": repeats, "total_tasks": total_tasks, "total_samples": total_samples,
    }
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), console=console) as progress:
        ptask = progress.add_task(f"Generating {total_samples} samples...", total=total_samples)
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_info = {executor.submit(worker, task): task for task in tasks_to_run}
            
            for future in as_completed(future_to_info):
                instance, repeat_id = future_to_info[future]
                try:
                    example = future.result()
                except Exception as e:
                    logger.exception(f"Error: {e}")
                    example = TrainingExample(
                        instance_id=instance.get("instance_id", "unknown"), repeat_id=repeat_id,
                        problem_statement=instance.get("problem_statement", ""), conversation={"system": "", "turns": []}, git_diff=None,
                        exit_status="error", reward=0.0, speedup=0.0, success=False, model_calls=0,
                        evaluation_info={"meta": {"success": False, "error": str(e)}}, error=str(e),
                    )
                
                examples.append(example)
                progress.update(ptask, description=f"[cyan]Completed {len(examples)}/{total_samples}[/cyan]")
                progress.advance(ptask)
                
                # Save intermediate results
                save_sft_data(examples, output_file, generation_metadata)
                save_trajectory_data(examples, trajectory_dir, generation_metadata)
    
    successful = sum(1 for ex in examples if ex.success)
    total_turns = sum(len(ex.conversation.get("turns", [])) for ex in examples)
    
    table = Table(title="Training Data Generation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Conversations", str(len(examples)))
    table.add_row("Total Turns", str(total_turns))
    table.add_row("Successful", f"{successful} ({successful/len(examples)*100:.1f}%)")
    table.add_row("Failed", f"{len(examples) - successful} ({(len(examples) - successful)/len(examples)*100:.1f}%)")
    table.add_row("Average Reward", f"{sum(ex.reward for ex in examples) / len(examples):.4f}")
    table.add_row("Total Model Calls", str(sum(ex.model_calls for ex in examples)))
    console.print("\n")
    console.print(table)
    console.print(f"\n[bold green]✓ SFT training data saved to {output_file}[/bold green]")
    console.print(f"[bold green]✓ Trajectory data saved to {trajectory_dir}[/bold green]")


if __name__ == "__main__":
    typer.run(main)
