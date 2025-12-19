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
    messages: list[dict]
    git_diff: Optional[str]
    exit_status: str
    reward: float
    speedup: float
    success: bool
    model_calls: int
    evaluation_info: dict
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


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
        
        return TrainingExample(
            instance_id=instance_id,
            repeat_id=repeat_id,
            problem_statement=instance["problem_statement"],
            messages=agent.get_full_messages(),
            git_diff=get_git_diff(agent.env, instance_id),
            exit_status=exit_status,
            reward=float(reward),
            speedup=float(speedup),
            success=True,
            model_calls=model.get_template_vars().get("n_model_calls", 0),
            evaluation_info=evaluation_info,
            metadata={"dataset_name": instance.get("dataset_name"), "split": instance.get("split"), "image_name": instance.get("image_name")},
        )
    except Exception as e:
        logger.exception(f"Error generating data for {instance_id}_repeat{repeat_id}: {e}")
        return TrainingExample(
            instance_id=instance_id, repeat_id=repeat_id, problem_statement=instance.get("problem_statement", ""),
            messages=[], git_diff=None, exit_status="error", reward=0.0, speedup=0.0, success=False,
            model_calls=0, evaluation_info={"meta": {"success": False, "error": str(e)}}, error=str(e),
        )


def save_training_data(examples: list[TrainingExample], output_path: Path, metadata: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": metadata,
        "examples": [
            {k: getattr(ex, k) for k in ["instance_id", "repeat_id", "problem_statement", "messages", "git_diff", 
             "exit_status", "reward", "speedup", "success", "model_calls", "evaluation_info", "error", "metadata"]}
            for ex in examples
        ],
        "summary": {
            "total_examples": len(examples),
            "successful": sum(1 for ex in examples if ex.success),
            "failed": sum(1 for ex in examples if not ex.success),
            "average_reward": sum(ex.reward for ex in examples) / len(examples) if examples else 0,
            "total_model_calls": sum(ex.model_calls for ex in examples),
        }
    }
    output_path.write_text(json.dumps(data, indent=2))
    logger.info(f"Training data saved to {output_path}")


def main(
    dataset_file: Path = typer.Option(..., "--dataset", "-d", help="Dataset JSON file"),
    api_key: str = typer.Option(..., "--api-key", help="API key"),
    model_name: str = typer.Option("Qwen/Qwen3-8B", "--model", "-m", help="Model name"),
    docker_server: str = typer.Option("10.67.77.184:9527", "--docker-server", help="Docker server"),
    eval_server: str = typer.Option("10.67.77.184:9528", "--eval-server", help="Eval server"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file"),
    output_file: Path = typer.Option(..., "--output", "-o", help="Output JSON file"),
    max_tasks: Optional[int] = typer.Option(None, "--max-tasks", help="Max number of tasks"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Log file path"),
    temperature: float = typer.Option(1.0, "--temperature", help="Sampling temperature"),
    max_tokens: int = typer.Option(8000, "--max-tokens", help="Max tokens"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers"),
    repeats: int = typer.Option(1, "--repeats", "-r", help="Number of times to repeat the entire dataset"),
):
    """Generate training data: outer loop = repeats, inner loop = dataset instances."""
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
    
    console.print(f"[bold blue]Generating training data[/bold blue]")
    console.print(f"Model: {model_name}, Tasks: {total_tasks}, Repeats: {repeats}, Total: {total_samples}, Workers: {workers}")
    
    docker_server_url, eval_server_url = f"http://{docker_server}", f"http://{eval_server}"
    
    def worker(args: tuple[dict, int]) -> TrainingExample:
        instance, repeat_id = args
        model = LiteLLMAMDModel(model_name=model_name, api_key=api_key, temperature=temperature, max_tokens=max_tokens, top_k=40)
        return asyncio.run(generate_single_example(instance, model, config, docker_server_url, eval_server_url, repeat_id))
    
    examples: list[TrainingExample] = []
    # Outer loop: repeats, Inner loop: dataset instances
    tasks_to_run = [(instance, repeat_id) for repeat_id in range(repeats) for instance in dataset]
    
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
                        problem_statement=instance.get("problem_statement", ""), messages=[], git_diff=None,
                        exit_status="error", reward=0.0, speedup=0.0, success=False, model_calls=0,
                        evaluation_info={"meta": {"success": False, "error": str(e)}}, error=str(e),
                    )
                
                examples.append(example)
                progress.update(ptask, description=f"[cyan]Completed {len(examples)}/{total_samples}[/cyan]")
                progress.advance(ptask)
                
                save_training_data(examples, output_file, {
                    "model_name": model_name, "temperature": temperature, "max_tokens": max_tokens,
                    "config_file": str(config_path), "dataset_file": str(dataset_file),
                    "workers": workers, "repeats": repeats, "total_tasks": total_tasks, "total_samples": total_samples,
                })
    
    successful = sum(1 for ex in examples if ex.success)
    table = Table(title="Training Data Generation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Examples", str(len(examples)))
    table.add_row("Successful", f"{successful} ({successful/len(examples)*100:.1f}%)")
    table.add_row("Failed", f"{len(examples) - successful} ({(len(examples) - successful)/len(examples)*100:.1f}%)")
    table.add_row("Average Reward", f"{sum(ex.reward for ex in examples) / len(examples):.4f}")
    table.add_row("Total Model Calls", str(sum(ex.model_calls for ex in examples)))
    console.print("\n")
    console.print(table)
    console.print(f"\n[bold green]✓ Training data saved to {output_file}[/bold green]")


if __name__ == "__main__":
    typer.run(main)
