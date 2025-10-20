


from minisweagent.environments.docker_remote import RemoteDockerEnvironment
from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import add_file_handler, logger
from pathlib import Path

import threading, json, traceback

import typer
import yaml
from datasets import load_dataset
from jinja2 import StrictUndefined, Template
from rich.live import Live



from agentlightning import LLM, LitAgent, NamedResources, Trainer, configure_logger, reward
from typing import Any, cast

from minisweagent.models.qwen import QwenModel

import requests 

_OUTPUT_FILE_LOCK = threading.Lock()





def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


# MODIFIED: This function now handles remote environment creation
def get_sb_environment(config: dict, instance: dict, server_url: str | None = None) -> Environment:
    """
    Get the environment for a SWE-bench instance.
    If environment_class is 'docker_remote', it creates a RemoteDockerEnvironment.
    Otherwise, it falls back to the default local environment creation.
    """
    env_config = config.setdefault("environment", {})
    env_class = env_config.get("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)

    if env_class == "docker_remote":
        if RemoteDockerEnvironment is None:
            raise RuntimeError("docker_remote environment requested, but 'RemoteDockerEnvironment' could not be imported.")
        if not server_url:
            raise ValueError("The 'docker_remote' environment class requires a --server-url or a 'server_url' in the config.")
        
        # All original env_config keys will be passed to RemoteDockerEnvironment
        env_config["image"] = image_name
        print(f"env_config{env_config}")
        env = RemoteDockerEnvironment(server_url=server_url, **env_config)
    else:
        # Original logic for local docker/singularity
        if env_class == "docker":
            env_config["image"] = image_name
        elif env_class == "singularity":
            env_config["image"] = "docker://" + image_name
        env = get_environment(env_config)

    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))

def get_agent(instance: dict,
    config: dict,
    server_url: str | None,
    model_config):


    # model = get_model(config=config.get("model", {}))
    model = QwenModel(**model_config)
    agent = None

    env = get_sb_environment(config, instance, server_url=server_url)
    agent = DefaultAgent(
        model=model,
        env=env,
        **config.get("agent", {}),
    )
    return agent

@reward
def get_reward(exit_status, result, container_id, instance_id, dataset_name, split, eval_server_url):
    logger.info("get_reward func is called")
    if exit_status != "Submitted" :
        logger.info(f"not submitted, give 0 as reward")
        return 0.0
    payload = {
        "instance_id":instance_id, 
        "patch":"",
        "dataset_name":dataset_name,
        "split": split,
        "container_id": container_id
    }
    try:
        resp = requests.post(f"{eval_server_url}/evaluate", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        reward = 1.0 if data["success"] else 0.0
        return reward

    except requests.RequestException as e:
        print("HTTP error:", e)
        return 0.0


class SweAgent(LitAgent):

    async def training_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:  # type: ignore
        llm: LLM = cast(LLM, resources.get("main_llm"))

        instance = task 
        instance_id = instance["instance_id"]

        model_config = {
            "model_name":llm.model,
            "model_kwargs":{
                "endpoint": llm.endpoint
            }
        }
        config_spec = Path(builtin_config_dir / "extra" / "swebench_remote.yaml")
        config_path = get_config_path(config_spec)
        config = yaml.safe_load(config_path.read_text())
        print(config)

        server_url = "http://localhost:9527"
        agent = get_agent(instance, config, server_url, model_config)

        problem = instance["problem_statement"]
        exit_status, result = agent.run(problem)

        container_id = agent.env.container_id

        dataset_name = task.get("dataset_name", "SWE-bench/SWE-bench_Lite")
        split = task.get("split", "test")
        eval_server_url = "http://localhost:9528"

        logger.info("agent rollout successfully, start to get reward")

        reward = get_reward(exit_status, result, container_id, instance_id, dataset_name, split, eval_server_url)
        
        logger.info(f"computed reward is {reward}")



        # return reward
    
    async def validation_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:  # type: ignore
        llm: LLM = cast(LLM, resources.get("main_llm"))
        task["dataset_name"] = "SWE-bench/SWE-bench_Lite"
        task["split"] = "dev"

        resources = {
            "main_llm": LLM(
                endpoint=llm.endpoint,
                model=llm.model,
                sampling_parameters={"temperature": 0},
            )
        }

        return await self.training_rollout_async(task, rollout_id, resources)

if __name__ == "__main__":
    Trainer(n_workers=10).fit(SweAgent(), "http://localhost:9999/")