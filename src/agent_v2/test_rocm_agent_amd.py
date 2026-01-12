# Copyright (c) Microsoft. All rights reserved.

"""Test script for ROCm agent with AMD internal GPT-5 model.

Supports both single task testing and full dataset testing.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import yaml
from datasets import load_dataset
from jinja2 import StrictUndefined, Template
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table

from concurrent.futures import ThreadPoolExecutor, as_completed

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments.docker_remote import RemoteDockerEnvironment
from minisweagent.models.litellm_amd_model import LiteLLMAMDModel
from minisweagent.utils.log import add_file_handler, logger
from eval_utils import evaluate,evaluate_info

app = typer.Typer()
console = Console()


def get_rocm_bench_docker_image_name(instance: dict) -> str:
    """Get Docker image name for the instance."""
    return instance.get("image_name", "rocm-lib")


def get_rocm_environment(
    config: dict, 
    instance: dict, 
    server_url: str
) -> RemoteDockerEnvironment:
    """Create a remote Docker environment for the ROCm instance."""
    env_config = config.setdefault("environment", {})
    image_name = get_rocm_bench_docker_image_name(instance)
    
    env_config["image"] = image_name
    logger.info(f"Creating environment with image: {image_name}")
    
    env = RemoteDockerEnvironment(server_url=server_url, **env_config)
    
    # Execute startup command if specified
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    
    return env


def get_agent(
    instance: dict,
    config: dict,
    server_url: str,
    model: LiteLLMAMDModel,
) -> DefaultAgent:
    """Create an agent for the given instance."""
    env = get_rocm_environment(config, instance, server_url)
    use_which_agent = config.get("use_which_agent", "rocm")
    if use_which_agent == "rocm":
        agent = DefaultAgent(
            model=model,
            env=env,
            **config.get("agent", {}),
        )
    else:
        from minisweagent.agents.mini import MiniAgent
        agent = MiniAgent(
            model = model,
            env=env,
            **config.get("agent", {})
        )
        
    return agent


def get_git_diff(env, instance_id: str) -> Optional[str]:
    """
    Get git diff from the environment.
    Tries multiple strategies:
    1. git diff --cached (staged changes)
    2. git diff (unstaged changes)
    3. git diff HEAD (all changes vs HEAD)
    """
    try:
        logger.info(f"Getting git diff for {instance_id}")
        
        # First try to get cached (staged) changes
        diff_result = env.execute("git diff --cached")
        if diff_result["returncode"] == 0 and diff_result["output"].strip():
            logger.info(f"Git diff (cached) obtained successfully for {instance_id}")
            return diff_result["output"]
        
        # If no cached changes, try regular diff
        diff_result = env.execute("git diff")
        if diff_result["returncode"] == 0 and diff_result["output"].strip():
            logger.info(f"Git diff obtained successfully for {instance_id}")
            return diff_result["output"]
        
        # Try diff against HEAD to capture all changes
        diff_result = env.execute("git diff HEAD")
        if diff_result["returncode"] == 0 and diff_result["output"].strip():
            logger.info(f"Git diff HEAD obtained successfully for {instance_id}")
            return diff_result["output"]
        
        logger.warning(f"No git changes found for {instance_id}")
        return None
        
    except Exception as e:
        logger.error(f"Error getting git diff for {instance_id}: {e}")
        return None


async def run_single_task(
    instance: dict,
    model: LiteLLMAMDModel,
    config: dict,
    docker_server_url: str,
    eval_server_url: str,
) -> Dict[str, Any]:
    """Run a single task and return results."""
    instance_id = instance["instance_id"]
    logger.info(f"Running task: {instance_id}")
    agent = None
    
    try:
        agent = get_agent(instance, config, docker_server_url, model)
        problem = instance["problem_statement"]
        
        # Run the agent
        container_id = agent.env.container_id
        logger.info(f"Starting agent execution for {instance_id}")
        exit_status, result = agent.run(problem, instance_id=instance_id, eval_server_url=eval_server_url, container_id=container_id)
        actions = getattr(agent, "actions", [])
        
        logger.info(f"Agent execution completed for {instance_id}, exit_status: {exit_status}")
        
        # Get git diff
        git_diff = get_git_diff(agent.env, instance_id)
        
        # Evaluate
        dataset_name = instance.get("dataset_name", "SWE-bench/SWE-bench_Lite")
        split = instance.get("split", "test")
        
        logger.info(f"Starting evaluation for {instance_id}")
        reward, speedup = await evaluate(
            exit_status, 
            result, 
            container_id, 
            instance_id, 
            dataset_name, 
            split, 
            eval_server_url
        )
        
        logger.info(f"Task {instance_id} completed with reward: {reward}")
        
        # Get model statistics
        model_stats = model.get_template_vars()
        
        return {
            "instance_id": instance_id,
            "exit_status": exit_status,
            "reward": float(reward),
            "speedup": float(speedup),
            "success": True,
            "error": None,
            "model_calls": model_stats.get("n_model_calls", 0),
            "model_cost": model_stats.get("model_cost", 0.0),
            "git_diff": git_diff,
            "actions": actions,
        }
        
    except Exception as e:
        logger.error(f"Error running task {instance_id}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "instance_id": instance_id,
            "exit_status": "error",
            "reward": 0.0,
            "success": False,
            "error": str(e),
            "model_calls": 0,
            "model_cost": 0.0,
            "git_diff": None,
        }
    finally:
        # Manually cleanup docker container after task completes
        if agent and hasattr(agent, "env") and hasattr(agent.env, "cleanup"):
            try:
                agent.env.cleanup()
                logger.info(f"Container cleaned up for {instance_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup container for {instance_id}: {e}")


from typing import Any, Dict

async def run_single_task_with_evaluation_info(
    instance: dict,
    model: LiteLLMAMDModel,
    config: dict,
    docker_server_url: str,
    eval_server_url: str,
) -> Dict[str, Any]:
    """Run a single task and return results."""
    instance_id = instance["instance_id"]
    logger.info(f"Running task: {instance_id}")
    agent = None

    try:
        agent = get_agent(instance, config, docker_server_url, model)
        problem = instance["problem_statement"]

        # Run the agent
        container_id = agent.env.container_id
        logger.info(f"Starting agent execution for {instance_id}")
        exit_status, result = agent.run(problem, instance_id=instance_id, eval_server_url=eval_server_url, container_id=container_id)
        actions = getattr(agent, "actions", [])

        logger.info(f"Agent execution completed for {instance_id}, exit_status: {exit_status}")

        # Get git diff
        git_diff = get_git_diff(agent.env, instance_id)

        # Evaluate
        dataset_name = instance.get("dataset_name", "SWE-bench/SWE-bench_Lite")
        split = instance.get("split", "test")

        logger.info(f"Starting evaluation for {instance_id}")
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

        # Get model statistics
        model_stats = model.get_template_vars()

        return {
            "instance_id": instance_id,
            "exit_status": exit_status,
            "reward": float(reward),
            "speedup": float(speedup),
            "success": True,
            "error": None,
            "model_calls": model_stats.get("n_model_calls", 0),
            "model_cost": model_stats.get("model_cost", 0.0),
            "git_diff": git_diff,
            "actions": actions,
            "evaluation_info": evaluation_info,
        }

    except Exception as e:
        logger.error(f"Error running task {instance_id}: {e}")
        import traceback
        traceback.print_exc()

        return {
            "instance_id": instance_id,
            "exit_status": "error",
            "reward": 0.0,
            "success": False,
            "error": str(e),
            "model_calls": 0,
            "model_cost": 0.0,
            "git_diff": None,
            "actions": [],
            "evaluation_info": {
                "meta": {
                    "success": False,
                    "error": str(e),
                }
            },
        }
    finally:
        # Manually cleanup docker container after task completes
        if agent and hasattr(agent, "env") and hasattr(agent.env, "cleanup"):
            try:
                agent.env.cleanup()
                logger.info(f"Container cleaned up for {instance_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup container for {instance_id}: {e}")


async def run_all_tasks(
    dataset_path: Path,
    model: LiteLLMAMDModel,
    config: dict,
    docker_server_url: str,
    eval_server_url: str,
    output_file: Optional[Path] = None,
    max_tasks: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run all tasks from a JSON dataset."""
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a JSON array of task instances")
    
    # Limit tasks if specified
    if max_tasks:
        dataset = dataset[:max_tasks]
    
    logger.info(f"Loaded {len(dataset)} tasks from {dataset_path}")
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Running {len(dataset)} tasks...", 
            total=len(dataset)
        )
        
        for idx, instance in enumerate(dataset):
            instance_id = instance.get("instance_id", f"task_{idx}")
            progress.update(
                task, 
                description=f"[cyan]Task {idx+1}/{len(dataset)}: {rich_escape(instance_id)}[/cyan]"
            )
            model.n_calls = 0 
            result = await run_single_task_with_evaluation_info(
                instance,
                model,
                config,
                docker_server_url,
                eval_server_url,
            )
            
            results.append(result)
            progress.advance(task)
            
            # Save intermediate results
            if output_file:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Saved intermediate results to {output_file}")
    
    # Calculate statistics
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    total_reward = sum(r["reward"] for r in results)
    avg_reward = total_reward / len(results) if results else 0
    total_calls = sum(r.get("model_calls", 0) for r in results)
    total_cost = sum(r.get("model_cost", 0.0) for r in results)
    
    # Display results table
    table = Table(title="Results Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Tasks", str(len(results)))
    table.add_row("Successful", str(successful))
    table.add_row("Failed", str(failed))
    table.add_row("Success Rate", f"{(successful/len(results)*100):.2f}%")
    table.add_row("Average Reward", f"{avg_reward:.4f}")
    table.add_row("Total Reward", f"{total_reward:.4f}")
    table.add_row("Total Model Calls", str(total_calls))
    table.add_row("Total Cost", f"${total_cost:.4f}")
    
    console.print("\n")
    console.print(table)
    
    return results


@app.command()
def test_single(
    instance_file: Path = typer.Option(
        ..., 
        "--instance", 
        "-i",
        help="Path to JSON file containing a single task instance"
    ),
    api_key: str = typer.Option(
        "c1f7f3ee59064fc0a5fad8c2586f1bd9",
        "--api-key",
        help="AMD API subscription key"
    ),
    model_name: str = typer.Option(
        "gpt-5",
        "--model",
        "-m",
        help="Model name (e.g., gpt-5)"
    ),
    docker_server: str = typer.Option(
        "10.67.77.184:9527",
        "--docker-server",
        help="Docker server address (IP:PORT)"
    ),
    eval_server: str = typer.Option(
        "10.67.77.184:9528",
        "--eval-server",
        help="Evaluation server address (IP:PORT)"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Custom config file (defaults to builtin rocm config)"
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results (JSON)"
    ),
    temperature: float = typer.Option(
        1.0,
        "--temperature",
        "-t",
        help="Sampling temperature"
    ),
    max_tokens: int = typer.Option(
        8000,
        "--max-tokens",
        help="Maximum tokens to generate"
    ),
):
    """Test a single ROCm task with AMD GPT-5 model."""
    
    # Load instance
    console.print(f"[cyan]Loading instance from {instance_file}...[/cyan]")
    with open(instance_file, 'r') as f:
        instance = json.load(f)
    
    # Load config
    if config_file:
        config_path = config_file
    else:
        config_spec = Path(builtin_config_dir / "rocm" / "config.yaml")
        config_path = get_config_path(config_spec)
    
    config = yaml.safe_load(config_path.read_text())
    logger.info(f"Loaded config from {config_path}")
    
    # Initialize model
    console.print(f"[cyan]Initializing {model_name} model...[/cyan]")

    model = LiteLLMAMDModel(
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    console.print(f"[bold blue]Testing single task with {model_name}[/bold blue]")
    console.print(f"Instance ID: {rich_escape(instance.get('instance_id', 'N/A'))}")
    
    # Run task
    docker_server_url = f"http://{docker_server}"
    eval_server_url = f"http://{eval_server}"
    
    with console.status("[bold green]Running task..."):
        result = asyncio.run(
            run_single_task(
                instance,
                model,
                config,
                docker_server_url,
                eval_server_url,
            )
        )
    
    # Display results
    console.print("\n[bold green]═══ Results ═══[/bold green]")
    
    result_table = Table(show_header=False, box=None)
    result_table.add_column("Key", style="cyan")
    result_table.add_column("Value", style="yellow")
    
    result_table.add_row("Instance ID", rich_escape(result['instance_id']))
    result_table.add_row("Success", "✓ Yes" if result['success'] else "✗ No")
    result_table.add_row("Reward", f"{result['reward']:.4f}")
    result_table.add_row("Speedup", f"{result['speedup']:.4f}")
    result_table.add_row("Exit Status", rich_escape(result['exit_status']))
    result_table.add_row("Model Calls", str(result.get('model_calls', 0)))
    result_table.add_row("Model Cost", f"${result.get('model_cost', 0.0):.4f}")
    
    if result['error']:
        result_table.add_row("Error", f"[red]{rich_escape(result['error'])}[/red]")
    
    console.print(result_table)
    
    # Display git diff if available
    if result.get('git_diff'):
        console.print("\n[bold cyan]═══ Git Diff ═══[/bold cyan]")
        console.print(result['git_diff'], markup=False)
    else:
        console.print("\n[yellow]No git diff available[/yellow]")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        console.print(f"\n[green]✓ Results saved to {output_file}[/green]")


@app.command()
def test_all(
    dataset_file: Path = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to JSON file containing all task instances"
    ),
    api_key: str = typer.Option(
        "c1f7f3ee59064fc0a5fad8c2586f1bd9",
        "--api-key",
        help="AMD API subscription key"
    ),
    model_name: str = typer.Option(
        "gpt-5",
        "--model",
        "-m",
        help="Model name (e.g., gpt-5)"
    ),
    docker_server: str = typer.Option(
        "10.67.77.184:9527",
        "--docker-server",
        help="Docker server address (IP:PORT)"
    ),
    eval_server: str = typer.Option(
        "10.67.77.184:9528",
        "--eval-server",
        help="Evaluation server address (IP:PORT)"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Custom config file (defaults to builtin rocm config)"
    ),
    output_file: Path = typer.Option(
        "results.json",
        "--output",
        "-o",
        help="Output file for results (JSON)"
    ),
    max_tasks: Optional[int] = typer.Option(
        None,
        "--max-tasks",
        help="Maximum number of tasks to run (for testing)"
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        help="Log file path"
    ),
    temperature: float = typer.Option(
        1.0,
        "--temperature",
        "-t",
        help="Sampling temperature"
    ),
    max_tokens: int = typer.Option(
        8000,
        "--max-tokens",
        help="Maximum tokens to generate"
    ),
):
    """Test all ROCm tasks from a dataset with AMD GPT-5 model."""
    
    # Setup logging
    if log_file:
        add_file_handler(logger, log_file)
        console.print(f"[cyan]Logging to {log_file}[/cyan]")
    
    # Load config
    if config_file:
        config_path = config_file
    else:
        config_spec = Path(builtin_config_dir / "rocm" / "config_amd_v1.yaml")
        config_path = get_config_path(config_spec)
    
    config = yaml.safe_load(config_path.read_text())
    logger.info(f"Loaded config from {config_path}")
    
    # Initialize model
    console.print(f"[cyan]Initializing {model_name} model...[/cyan]")
    model = LiteLLMAMDModel(
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        top_k=40,
    )
    
    console.print(f"[bold blue]Testing dataset with {model_name}[/bold blue]")
    console.print(f"Dataset: {dataset_file}")
    if max_tasks:
        console.print(f"Max tasks: {max_tasks}")
    console.print(f"Output: {output_file}\n")
    
    # Run all tasks
    docker_server_url = f"http://{docker_server}"
    eval_server_url = f"http://{eval_server}"
    
    results = asyncio.run(
        run_all_tasks(
            dataset_file,
            model,
            config,
            docker_server_url,
            eval_server_url,
            output_file,
            max_tasks,
        )
    )
    
    # Save final results with metadata
    final_output = {
        "metadata": {
            "model_name": model_name,
            "dataset_file": str(dataset_file),
            "total_tasks": len(results),
            "config_file": str(config_path),
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        "results": results,
        "summary": {
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "average_reward": sum(r["reward"] for r in results) / len(results) if results else 0,
            "total_reward": sum(r["reward"] for r in results),
            "total_model_calls": sum(r.get("model_calls", 0) for r in results),
            "total_cost": sum(r.get("model_cost", 0.0) for r in results),
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    console.print(f"\n[bold green]✓ All results saved to {output_file}[/bold green]")


def generate_simple_report(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate a simplified report with key information only."""
    simple_results = []
    for r in results:
        simple_item = {
            "instance_id": r.get("instance_id"),
            "success": r.get("success"),
            "reward": r.get("reward"),
            "speedup": r.get("speedup"),
            "exit_status": r.get("exit_status"),
            "error": r.get("error"),
            "git_diff": r.get("git_diff"),
            "model_calls": r.get("model_calls", 0),
        }
        eval_info = r.get("evaluation_info", {})
        if eval_info:
            meta = eval_info.get("meta", {})
            simple_item["eval_success"] = meta.get("success")
            simple_item["eval_reason"] = meta.get("reason")
            simple_item["eval_error"] = meta.get("error")
            extracted = eval_info.get("extracted", {})
            if extracted:
                simple_item["eval_exit_code"] = extracted.get("exit_code")
                simple_item["eval_timed_out"] = extracted.get("timed_out")
        simple_results.append(simple_item)
    return simple_results


@app.command("test_all_multi_thread")
def test_all_multi_thread(
    dataset_file: Path = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to JSON file containing all task instances"
    ),
    api_key: str = typer.Option(
        "c1f7f3ee59064fc0a5fad8c2586f1bd9",
        "--api-key",
        help="AMD API subscription key"
    ),
    model_name: str = typer.Option(
        "gpt-5",
        "--model",
        "-m",
        help="Model name (e.g., gpt-5)"
    ),
    docker_server: str = typer.Option(
        "10.67.77.184:9527",
        "--docker-server",
        help="Docker server address (IP:PORT)"
    ),
    eval_server: str = typer.Option(
        "10.67.77.184:9528",
        "--eval-server",
        help="Evaluation server address (IP:PORT)"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Custom config file (defaults to builtin rocm config)"
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Directory to write outputs (output.json, simple_output.json, actions.json)"
    ),
    max_tasks: Optional[int] = typer.Option(
        None,
        "--max-tasks",
        help="Maximum number of tasks to run (for testing)"
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        help="Log file path"
    ),
    temperature: float = typer.Option(
        1.0,
        "--temperature",
        "-t",
        help="Sampling temperature"
    ),
    max_tokens: int = typer.Option(
        8000,
        "--max-tokens",
        help="Maximum tokens to generate"
    ),
    workers: int = typer.Option(
        max(1, os.cpu_count() or 4),
        "--workers",
        "-w",
        help="Number of threads for parallel task execution"
    ),
    use_which_agent: str = typer.Option(
        "rocm",
        "--use-rocm-agent",
        help="which agent"
    )
):
    """
    测试数据集中的所有 ROCm 任务，使用多线程并发执行（每线程独立模型实例）。
    """

    # Setup logging
    if log_file:
        add_file_handler(logger, log_file)
        console.print(f"[cyan]Logging to {log_file}[/cyan]")

    # Load config
    if config_file:
        config_path = config_file
    else:
        config_spec = Path(builtin_config_dir / "rocm" / "config_amd_v1.yaml")
        config_path = get_config_path(config_spec)

    config = yaml.safe_load(config_path.read_text())
    config["use_which_agent"] = use_which_agent
    logger.info(f"Loaded config from {config_path}")

    console.print(f"[bold blue]Testing dataset with {model_name} (multi-thread)[/bold blue]")
    console.print(f"Dataset: {dataset_file}")
    if max_tasks:
        console.print(f"Max tasks: {max_tasks}")
    console.print(f"Output dir: {output_dir}")
    console.print(f"Workers: {workers}\n")

    docker_server_url = f"http://{docker_server}"
    eval_server_url = f"http://{eval_server}"

    # Load dataset (必须是 list[instance])
    with open(dataset_file, "r") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a JSON array of task instances")
    if max_tasks:
        dataset = dataset[:max_tasks]

    total = len(dataset)
    logger.info(f"Loaded {total} tasks from {dataset_file}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "output.json"
    simple_output_file = output_dir / "simple_output.json"
    actions_output_file = output_dir / "actions.json"

    # 子线程 worker：每个线程里创建自己的模型实例，并执行单个任务
    def worker(instance: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 每线程独立模型，避免线程安全问题
            model = LiteLLMAMDModel(
                model_name=model_name,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=40,
            )
            # 若需要重置统计字段，可在此设置
            # model.n_calls = 0

            # 调用异步单任务函数
            return asyncio.run(
                run_single_task_with_evaluation_info(
                    instance=instance,
                    model=model,
                    config=config,
                    docker_server_url=docker_server_url,
                    eval_server_url=eval_server_url,
                )
            )
        except Exception as e:
            logger.exception(f"Worker error for instance {instance.get('instance_id')}: {e}")
            return {
                "instance_id": instance.get("instance_id"),
                "exit_status": "error",
                "reward": 0.0,
                "success": False,
                "error": str(e),
                "model_calls": 0,
                "model_cost": 0.0,
                "git_diff": None,
                "evaluation_info": {
                    "meta": {
                        "success": False,
                        "error": str(e),
                    }
                },
            }

    results: List[Dict[str, Any]] = []
    instance_ids: List[str] = [ins.get("instance_id", f"task_{i}") for i, ins in enumerate(dataset)]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        ptask = progress.add_task(f"Running {total} tasks in parallel...", total=total)

        # 提交任务
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {executor.submit(worker, ins): i for i, ins in enumerate(dataset)}

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                ins_id = instance_ids[idx]
                try:
                    res = future.result()
                except Exception as e:
                    logger.exception(f"Unhandled exception in future for {ins_id}: {e}")
                    res = {
                        "instance_id": ins_id,
                        "exit_status": "error",
                        "reward": 0.0,
                        "success": False,
                        "error": str(e),
                        "model_calls": 0,
                        "model_cost": 0.0,
                        "git_diff": None,
                        "evaluation_info": {
                            "meta": {
                                "success": False,
                                "error": str(e),
                            }
                        },
                    }

                results.append(res)
                # 更新进度描述
                progress.update(
                    ptask,
                    description=f"[cyan]Completed {len(results)}/{total} (last: {rich_escape(ins_id)})[/cyan]"
                )
                progress.advance(ptask)

                # 保存中间结果（可选）
                try:
                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=2)
                    logger.info(f"Saved intermediate results to {output_file} ({len(results)}/{total})")
                except Exception as e:
                    logger.warning(f"Failed to save intermediate results: {e}")

    # 汇总统计
    successful = sum(1 for r in results if r.get("success"))
    failed = len(results) - successful
    total_reward = sum(r.get("reward", 0.0) for r in results)
    avg_reward = (total_reward / len(results)) if results else 0.0
    total_calls = sum(r.get("model_calls", 0) for r in results)
    total_cost = sum(r.get("model_cost", 0.0) for r in results)

    # 显示结果表
    table = Table(title="Results Summary (Multi-thread)", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Tasks", str(len(results)))
    table.add_row("Successful", str(successful))
    table.add_row("Failed", str(failed))
    table.add_row("Success Rate", f"{(successful/len(results)*100):.2f}%")
    table.add_row("Average Reward", f"{avg_reward:.4f}")
    table.add_row("Total Reward", f"{total_reward:.4f}")
    table.add_row("Total Model Calls", str(total_calls))
    table.add_row("Total Cost", f"${total_cost:.4f}")

    console.print("\n")
    console.print(table)

    # 保存最终带元数据的结果
    final_output = {
        "metadata": {
            "mode": "multi-thread",
            "model_name": model_name,
            "dataset_file": str(dataset_file),
            "total_tasks": len(results),
            "config_file": str(config_path),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "workers": workers,
        },
        "results": results,
        "summary": {
            "successful": successful,
            "failed": failed,
            "average_reward": avg_reward,
            "total_reward": total_reward,
            "total_model_calls": total_calls,
            "total_cost": total_cost,
        },
    }

    try:
        with open(output_file, "w") as f:
            json.dump(final_output, f, indent=2)
        console.print(f"\n[bold green]✓ Full results saved to {output_file}[/bold green]")
    except Exception as e:
        logger.error(f"Failed to write final results to {output_file}: {e}")

    # Generate and save simple report
    simple_results = generate_simple_report(results)
    simple_output = {
        "metadata": final_output["metadata"],
        "results": simple_results,
        "summary": final_output["summary"],
    }
    try:
        with open(simple_output_file, "w") as f:
            json.dump(simple_output, f, indent=2)
        console.print(f"[bold green]✓ Simple results saved to {simple_output_file}[/bold green]")
    except Exception as e:
        logger.error(f"Failed to write simple results to {simple_output_file}: {e}")

    # Save action sequences separately
    try:
        actions_payload = [
            {"instance_id": r.get("instance_id"), "actions": r.get("actions", [])}
            for r in results
        ]
        with open(actions_output_file, "w") as f:
            json.dump({"metadata": final_output["metadata"], "actions": actions_payload}, f, indent=2)
        console.print(f"[bold green]✓ Action sequences saved to {actions_output_file}[/bold green]")
    except Exception as e:
        logger.error(f"Failed to write action sequences to {actions_output_file}: {e}")

@app.command()
def test_connection(
    api_key: str = typer.Option(
        "c1f7f3ee59064fc0a5fad8c2586f1bd9",
        "--api-key",
        help="AMD API subscription key"
    ),
    model_name: str = typer.Option(
        "gpt-5",
        "--model",
        "-m",
        help="Model name to test"
    ),
    temperature: float = typer.Option(
        1.0,
        "--temperature",
        help="Sampling temperature"
    ),
):
    """Test connection to AMD LLM API."""
    
    console.print(f"[bold blue]Testing connection to AMD LLM API[/bold blue]")
    console.print(f"Model: {model_name}")
    console.print(f"Temperature: {temperature}\n")
    
    try:
        with console.status("[bold green]Initializing model..."):
            model = LiteLLMAMDModel(
                model_name=model_name,
                api_key=api_key,
                temperature=temperature,
                top_k=40,
            )
        
        console.print("[green]✓ Model initialized successfully[/green]")
        
        messages = [
            {"role": "user", "content": "Hello! Please respond with 'OK' if you can read this."}
        ]
        
        with console.status("[bold yellow]Sending test message..."):
            response = model.query(messages)
        
        console.print("\n[bold green]✓ Connection successful![/bold green]")
        console.print(f"\n[cyan]Response:[/cyan]")
        console.print(response['content'], markup=False)
        console.print()
        
        # Display model stats
        stats = model.get_template_vars()
        stats_table = Table(title="Model Statistics", show_header=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="yellow")
        
        stats_table.add_row("Model Calls", str(stats.get('n_model_calls', 0)))
        stats_table.add_row("Total Cost", f"${stats.get('model_cost', 0.0):.4f}")
        
        console.print(stats_table)
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Connection failed![/bold red]")
        console.print(f"[red]Error: {rich_escape(str(e))}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def generate_simple(
    results_file: Path = typer.Option(
        ...,
        "--results",
        "-r",
        help="Path to full results JSON file"
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for simple results (defaults to input_simple.json)"
    ),
):
    """Generate a simple report from an existing full results file."""
    console.print(f"[cyan]Loading results from {results_file}...[/cyan]")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Handle both list format and dict format
    if isinstance(data, list):
        results = data
        metadata = {}
        summary = {}
    else:
        results = data.get("results", data)
        metadata = data.get("metadata", {})
        summary = data.get("summary", {})
    
    simple_results = generate_simple_report(results)
    
    # Compute summary if not present
    if not summary:
        successful = sum(1 for r in results if r.get("success"))
        failed = len(results) - successful
        total_reward = sum(r.get("reward", 0.0) for r in results)
        summary = {
            "successful": successful,
            "failed": failed,
            "average_reward": total_reward / len(results) if results else 0,
            "total_reward": total_reward,
        }
    
    simple_output = {
        "metadata": metadata,
        "results": simple_results,
        "summary": summary,
    }
    
    if output_file is None:
        output_file = Path(str(results_file).replace(".json", "_simple.json"))
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(simple_output, f, indent=2)
    
    console.print(f"[bold green]✓ Simple report saved to {output_file}[/bold green]")
    console.print(f"[cyan]Total tasks: {len(simple_results)}[/cyan]")


@app.command()
def analyze_results(
    results_file: Path = typer.Option(
        ...,
        "--results",
        "-r",
        help="Path to results JSON file"
    ),
):
    """Analyze results from a previous test run."""
    
    console.print(f"[cyan]Loading results from {results_file}...[/cyan]\n")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Check if it's the new format with metadata
    if "metadata" in data and "results" in data:
        metadata = data["metadata"]
        results = data["results"]
        summary = data.get("summary", {})
    else:
        # Old format - just a list of results
        metadata = {}
        results = data
        summary = {}
    
    # Display metadata
    if metadata:
        console.print("[bold blue]Metadata[/bold blue]")
        meta_table = Table(show_header=False, box=None)
        meta_table.add_column("Key", style="cyan")
        meta_table.add_column("Value", style="yellow")
        
        for key, value in metadata.items():
            meta_table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(meta_table)
        console.print()
    
    # Calculate statistics
    total = len(results)
    successful = sum(1 for r in results if r.get("success", False))
    failed = total - successful
    
    rewards = [r.get("reward", 0.0) for r in results]
    avg_reward = sum(rewards) / total if total > 0 else 0
    max_reward = max(rewards) if rewards else 0
    min_reward = min(rewards) if rewards else 0
    
    # Display summary
    summary_table = Table(title="Results Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Total Tasks", str(total))
    summary_table.add_row("Successful", f"{successful} ({successful/total*100:.1f}%)")
    summary_table.add_row("Failed", f"{failed} ({failed/total*100:.1f}%)")
    summary_table.add_row("Average Reward", f"{avg_reward:.4f}")
    summary_table.add_row("Max Reward", f"{max_reward:.4f}")
    summary_table.add_row("Min Reward", f"{min_reward:.4f}")
    
    if summary:
        summary_table.add_row("Total Model Calls", str(summary.get("total_model_calls", "N/A")))
        summary_table.add_row("Total Cost", f"${summary.get('total_cost', 0.0):.4f}")
    
    console.print(summary_table)
    console.print()
    
    # Display top performers
    sorted_results = sorted(results, key=lambda x: x.get("reward", 0.0), reverse=True)
    
    top_table = Table(title="Top 5 Performers", show_header=True, header_style="bold green")
    top_table.add_column("Rank", style="cyan", width=6)
    top_table.add_column("Instance ID", style="yellow")
    top_table.add_column("Reward", style="green", justify="right")
    top_table.add_column("Status", style="magenta")
    
    for i, result in enumerate(sorted_results[:5], 1):
        status = "✓" if result.get("success", False) else "✗"
        top_table.add_row(
            str(i),
            rich_escape(result.get("instance_id", "N/A")),
            f"{result.get('reward', 0.0):.4f}",
            status
        )
    
    console.print(top_table)
    console.print()
    
    # Display failures if any
    failures = [r for r in results if not r.get("success", False)]
    if failures:
        fail_table = Table(title=f"Failed Tasks ({len(failures)})", show_header=True, header_style="bold red")
        fail_table.add_column("Instance ID", style="yellow")
        fail_table.add_column("Error", style="red")
        
        for result in failures[:10]:  # Show first 10 failures
            error_msg = result.get("error", "Unknown error")
            # Truncate long error messages
            if len(error_msg) > 60:
                error_msg = error_msg[:57] + "..."
            fail_table.add_row(
                rich_escape(result.get("instance_id", "N/A")),
                rich_escape(error_msg)
            )
        
        if len(failures) > 10:
            fail_table.add_row("...", f"and {len(failures) - 10} more failures")
        
        console.print(fail_table)


if __name__ == "__main__":
    app()



