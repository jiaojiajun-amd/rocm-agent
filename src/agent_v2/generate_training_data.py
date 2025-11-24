"""Generate training data for agent continuous pretraining using mini agent.

This script runs mini agent on tasks and collects:
- Complete dialogue trajectories (all messages)
- Code changes (git diffs)
- Evaluation results
- Task metadata
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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

app = typer.Typer()
console = Console()


@dataclass
class TrainingExample:
    """Single training example for agent continuous pretraining."""
    instance_id: str
    sample_id: int
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


def get_rocm_bench_docker_image_name(instance: dict) -> str:
    return instance.get("image_name", "rocm-lib")


def get_rocm_environment(
    config: dict,
    instance: dict,
    server_url: str
) -> RemoteDockerEnvironment:
    env_config = config.setdefault("environment", {})
    image_name = get_rocm_bench_docker_image_name(instance)
    env_config["image"] = image_name
    logger.info(f"Creating environment with image: {image_name}")
    env = RemoteDockerEnvironment(server_url=server_url, **env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def get_mini_agent(
    instance: dict,
    config: dict,
    server_url: str,
    model: LiteLLMAMDModel,
) -> MiniAgent:
    env = get_rocm_environment(config, instance, server_url)
    return MiniAgent(
        model=model,
        env=env,
        **config.get("agent", {})
    )


def get_git_diff(env, instance_id: str) -> Optional[str]:
    try:
        logger.info(f"Getting git diff for {instance_id}")
        diff_result = env.execute("git diff --cached")
        if diff_result["returncode"] == 0 and diff_result["output"].strip():
            logger.info(f"Git diff (cached) obtained for {instance_id}")
            return diff_result["output"]
        diff_result = env.execute("git diff")
        if diff_result["returncode"] == 0 and diff_result["output"].strip():
            logger.info(f"Git diff obtained for {instance_id}")
            return diff_result["output"]
        diff_result = env.execute("git diff HEAD")
        if diff_result["returncode"] == 0 and diff_result["output"].strip():
            logger.info(f"Git diff HEAD obtained for {instance_id}")
            return diff_result["output"]
        logger.warning(f"No git changes found for {instance_id}")
        return None
    except Exception as e:
        logger.error(f"Error getting git diff for {instance_id}: {e}")
        return None


async def generate_single_example(
    instance: dict,
    model: LiteLLMAMDModel,
    config: dict,
    docker_server_url: str,
    eval_server_url: str,
    sample_id: int = 0,
) -> TrainingExample:
    instance_id = instance["instance_id"]
    full_id = f"{instance_id}_sample{sample_id}"
    logger.info(f"Generating training data for: {full_id}")
    
    try:
        agent = get_mini_agent(instance, config, docker_server_url, model)
        problem = instance["problem_statement"]
        
        logger.info(f"Starting agent execution for {instance_id}")
        exit_status, result = agent.run(problem)
        container_id = agent.env.container_id
        
        logger.info(f"Agent completed for {instance_id}, exit_status: {exit_status}")
        
        git_diff = get_git_diff(agent.env, instance_id)
        
        dataset_name = instance.get("dataset_name", "SWE-bench/SWE-bench_Lite")
        split = instance.get("split", "test")
        
        logger.info(f"Evaluating {instance_id}")
        reward, speedup, evaluation_info = await evaluate_info(
            exit_status,
            result,
            container_id,
            instance_id,
            dataset_name,
            split,
            eval_server_url,
        )
        
        logger.info(f"Task {instance_id} completed with reward: {reward}")
        
        model_stats = model.get_template_vars()
        
        return TrainingExample(
            instance_id=instance_id,
            sample_id=sample_id,
            problem_statement=problem,
            messages=agent.messages.copy(),
            git_diff=git_diff,
            exit_status=exit_status,
            reward=float(reward),
            speedup=float(speedup),
            success=True,
            model_calls=model_stats.get("n_model_calls", 0),
            evaluation_info=evaluation_info,
            error=None,
            metadata={
                "dataset_name": dataset_name,
                "split": split,
                "image_name": instance.get("image_name"),
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating data for {full_id}: {e}")
        import traceback
        traceback.print_exc()
        
        return TrainingExample(
            instance_id=instance_id,
            sample_id=sample_id,
            problem_statement=instance.get("problem_statement", ""),
            messages=[],
            git_diff=None,
            exit_status="error",
            reward=0.0,
            speedup=0.0,
            success=False,
            model_calls=0,
            evaluation_info={"meta": {"success": False, "error": str(e)}},
            error=str(e),
        )


def save_training_data(
    examples: list[TrainingExample],
    output_path: Path,
    metadata: dict,
):
    data = {
        "metadata": metadata,
        "examples": [
            {
                "instance_id": ex.instance_id,
                "sample_id": ex.sample_id,
                "problem_statement": ex.problem_statement,
                "messages": ex.messages,
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
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Training data saved to {output_path}")


@app.command()
def generate_single(
    instance_file: Path = typer.Option(
        ...,
        "--instance",
        "-i",
        help="Path to JSON file containing a single task instance"
    ),
    api_key: str = typer.Option(..., "--api-key", help="API key"),
    model_name: str = typer.Option("Qwen/Qwen3-8B", "--model", "-m", help="Model name"),
    docker_server: str = typer.Option("10.67.77.184:9527", "--docker-server", help="Docker server"),
    eval_server: str = typer.Option("10.67.77.184:9528", "--eval-server", help="Eval server"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file"),
    output_file: Path = typer.Option(..., "--output", "-o", help="Output JSON file"),
    temperature: float = typer.Option(1.0, "--temperature", help="Sampling temperature"),
    max_tokens: int = typer.Option(8000, "--max-tokens", help="Max tokens"),
):
    """Generate training data from a single task."""
    console.print(f"[cyan]Loading instance from {instance_file}...[/cyan]")
    with open(instance_file) as f:
        instance = json.load(f)
    
    if config_file:
        config_path = config_file
    else:
        config_spec = Path(builtin_config_dir / "mini.yaml")
        config_path = get_config_path(config_spec)
    
    config = yaml.safe_load(config_path.read_text())
    logger.info(f"Loaded config from {config_path}")
    
    console.print(f"[cyan]Initializing {model_name} model...[/cyan]")
    model = LiteLLMAMDModel(
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    docker_server_url = f"http://{docker_server}"
    eval_server_url = f"http://{eval_server}"
    
    with console.status("[bold green]Generating training data..."):
        example = asyncio.run(
            generate_single_example(
                instance,
                model,
                config,
                docker_server_url,
                eval_server_url,
            )
        )
    
    metadata = {
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "config_file": str(config_path),
    }
    
    save_training_data([example], output_file, metadata)
    
    console.print(f"[bold green]✓ Training data saved to {output_file}[/bold green]")
    console.print(f"Success: {example.success}, Reward: {example.reward:.4f}")


@app.command()
def generate_multi(
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
    samples_per_task: int = typer.Option(1, "--samples-per-task", help="Number of samples to generate per task"),
):
    """Generate training data from multiple tasks using multi-threading.
    
    Each task can generate multiple samples (controlled by --samples-per-task) to create
    diverse training data through sampling (requires temperature > 0).
    """
    if log_file:
        add_file_handler(log_file)
        console.print(f"[cyan]Logging to {log_file}[/cyan]")
    
    if config_file:
        config_path = config_file
    else:
        config_spec = Path(builtin_config_dir / "mini.yaml")
        config_path = get_config_path(config_spec)
    
    config = yaml.safe_load(config_path.read_text())
    logger.info(f"Loaded config from {config_path}")
    
    with open(dataset_file) as f:
        dataset = json.load(f)
    
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a JSON array")
    
    if max_tasks:
        dataset = dataset[:max_tasks]
    
    total_tasks = len(dataset)
    total_samples = total_tasks * samples_per_task
    logger.info(f"Loaded {total_tasks} tasks from {dataset_file}")
    logger.info(f"Will generate {samples_per_task} samples per task, total {total_samples} samples")
    
    console.print(f"[bold blue]Generating training data with mini agent[/bold blue]")
    console.print(f"Model: {model_name}")
    console.print(f"Dataset: {dataset_file}")
    console.print(f"Tasks: {total_tasks}")
    console.print(f"Samples per task: {samples_per_task}")
    console.print(f"Total samples: {total_samples}")
    console.print(f"Workers: {workers}")
    console.print(f"Output: {output_file}\n")
    
    docker_server_url = f"http://{docker_server}"
    eval_server_url = f"http://{eval_server}"
    
    def worker(instance_and_sample: tuple[dict, int]) -> TrainingExample:
        instance, sample_id = instance_and_sample
        try:
            model = LiteLLMAMDModel(
                model_name=model_name,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=40,
            )
            return asyncio.run(
                generate_single_example(
                    instance=instance,
                    model=model,
                    config=config,
                    docker_server_url=docker_server_url,
                    eval_server_url=eval_server_url,
                    sample_id=sample_id,
                )
            )
        except Exception as e:
            logger.exception(f"Worker error for {instance.get('instance_id')} sample{sample_id}: {e}")
            return TrainingExample(
                instance_id=instance.get("instance_id", "unknown"),
                sample_id=sample_id,
                problem_statement=instance.get("problem_statement", ""),
                messages=[],
                git_diff=None,
                exit_status="error",
                reward=0.0,
                speedup=0.0,
                success=False,
                model_calls=0,
                evaluation_info={"meta": {"success": False, "error": str(e)}},
                error=str(e),
            )
    
    examples: list[TrainingExample] = []
    
    # Create task list: (instance, sample_id) for each sample
    tasks_to_run = []
    for ins in dataset:
        for sample_id in range(samples_per_task):
            tasks_to_run.append((ins, sample_id))
    
    task_info_map = {}  # future -> (task_idx, sample_id)
    for task_idx, (ins, sample_id) in enumerate(tasks_to_run):
        task_info_map[task_idx] = (ins.get("instance_id", f"task_{task_idx}"), sample_id)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        ptask = progress.add_task(f"Generating {total_samples} samples...", total=total_samples)
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task_idx = {
                executor.submit(worker, task_data): task_idx 
                for task_idx, task_data in enumerate(tasks_to_run)
            }
            
            for future in as_completed(future_to_task_idx):
                task_idx = future_to_task_idx[future]
                ins_id, sample_id = task_info_map[task_idx]
                full_id = f"{ins_id}_sample{sample_id}"
                try:
                    example = future.result()
                except Exception as e:
                    logger.exception(f"Error getting result for {full_id}: {e}")
                    example = TrainingExample(
                        instance_id=ins_id,
                        sample_id=sample_id,
                        problem_statement="",
                        messages=[],
                        git_diff=None,
                        exit_status="error",
                        reward=0.0,
                        speedup=0.0,
                        success=False,
                        model_calls=0,
                        evaluation_info={"meta": {"success": False, "error": str(e)}},
                        error=str(e),
                    )
                
                examples.append(example)
                progress.update(
                    ptask,
                    description=f"[cyan]Completed {len(examples)}/{total_samples} (last: {full_id})[/cyan]"
                )
                progress.advance(ptask)
                
                metadata = {
                    "model_name": model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "config_file": str(config_path),
                    "dataset_file": str(dataset_file),
                    "workers": workers,
                    "samples_per_task": samples_per_task,
                    "total_tasks": total_tasks,
                    "total_samples": total_samples,
                }
                save_training_data(examples, output_file, metadata)
    
    successful = sum(1 for ex in examples if ex.success)
    failed = len(examples) - successful
    avg_reward = sum(ex.reward for ex in examples) / len(examples) if examples else 0
    
    table = Table(title="Training Data Generation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Examples", str(len(examples)))
    table.add_row("Successful", f"{successful} ({successful/len(examples)*100:.1f}%)")
    table.add_row("Failed", f"{failed} ({failed/len(examples)*100:.1f}%)")
    table.add_row("Average Reward", f"{avg_reward:.4f}")
    table.add_row("Total Model Calls", str(sum(ex.model_calls for ex in examples)))
    
    console.print("\n")
    console.print(table)
    console.print(f"\n[bold green]✓ Training data saved to {output_file}[/bold green]")


if __name__ == "__main__":
    app()

