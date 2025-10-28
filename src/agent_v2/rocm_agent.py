# Copyright (c) Microsoft. All rights reserved.

"""This sample code demonstrates how to define a Calc-X agent trainable with Agent-lightning
with latest Agent-lightning API (v0.2+)."""

import asyncio
import os
import re
from typing import TypedDict, cast

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from eval_utils import evaluate

import agentlightning as agl


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

from typing import Any, cast

from minisweagent.models.qwen import QwenModel

import requests 


_OUTPUT_FILE_LOCK = threading.Lock()

class RocmProblem(TypedDict):

    repo: str 
    instance_id: str 
    path_in_repo: str 
    commit_id: str 
    problem_statement: str 
    created_at: str 
    version: str 
    test_file_path: str 
    benchmark_file_path: str 
    docker_env_path: str 
    image_name: str 
    test_file: str 
    benchmark_file: str 





def get_rocm_bench_docker_image_name(instance: dict) -> str:
    image_name = instance.get("image_name", "rocm-lib")
    return image_name

def get_rocm_environment(config: dict, instance: dict, server_url: str | None = None) -> Environment:
    """
    Get the environment for a SWE-bench instance.
    If environment_class is 'docker_remote', it creates a RemoteDockerEnvironment.
    Otherwise, it falls back to the default local environment creation.
    """
    env_config = config.setdefault("environment", {})
    env_class = env_config.get("environment_class", "docker")
    image_name = get_rocm_bench_docker_image_name(instance)
    assert env_class == "docker_remote" "we only support docker_remote"

    if RemoteDockerEnvironment is None:
        raise RuntimeError("docker_remote environment requested, but 'RemoteDockerEnvironment' could not be imported.")
    if not server_url:
        raise ValueError("The 'docker_remote' environment class requires a --server-url or a 'server_url' in the config.")
    
        # All original env_config keys will be passed to RemoteDockerEnvironment
    env_config["image"] = image_name
    print(f"env_config{env_config}")
    env = RemoteDockerEnvironment(server_url=server_url, **env_config)

    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env



def get_agent(instance: dict,
    config: dict,
    server_url: str | None,
    model_config):


    # model = get_model(config=config.get("model", {}))
    model = QwenModel(**model_config)
    agent = None

    env = get_rocm_environment(config, instance, server_url=server_url)
    agent = DefaultAgent(
        model=model,
        env=env,
        **config.get("agent", {}),
    )
    return agent


@agl.rollout
async def rocm_agent(task: RocmProblem, llm: agl.LLM) -> None:
    """Calc-X agent rollout function.

    It would accept a math problem and a LLM endpoint resource.
    It's expected to return None, and emit reward via `agl.emit_reward`.
    It can also return the reward directly without `agl.emit_reward`.
    You can choose either way, but not both.
    """

    instance = task 
    instance_id = instance["instance_id"]

    model_config = {
        "model_name":llm.model,
        "model_kwargs":{
            "endpoint": llm.endpoint
        }
    }
    config_spec = Path(builtin_config_dir / "rocm" / "config.yaml")
    config_path = get_config_path(config_spec)
    config = yaml.safe_load(config_path.read_text())
    print(config)

    server_ip = "10.67.77.184"

    docker_server_url = f"http://{server_ip}:9527"
    agent = get_agent(instance, config, docker_server_url, model_config)

    problem = instance["problem_statement"]
    exit_status, result = agent.run(problem)

    container_id = agent.env.container_id

    dataset_name = task.get("dataset_name", "SWE-bench/SWE-bench_Lite")
    split = task.get("split", "test")
    eval_server_url = f"http://{server_ip}:9528"

    logger.info("agent rollout successfully, start to get reward")

    reward = evaluate(exit_status, result, container_id, instance_id, dataset_name, split, eval_server_url)
    
    logger.info(f"computed reward is {reward}")

    agl.emit_reward(reward)



async def debug():
    """Here we show a more manual way for debugging, without Trainer.

    We get the data samples on our own, and run the agent with LitAgentRunner.
    You will need an `OPENAI_API_KEY` and `OPENAI_BASE_URL` environment variable set
    to run this function.
    """
    # Manually create a tracer as Runner will need it.
    # Use a dummy OtelTracer if you don't need to trace anything other than reward.
    tracer = agl.OtelTracer()
    # The runner processes MathProblem, which matches the agent's task type.
    runner = agl.LitAgentRunner[RocmProblem](tracer)

    # A store is required here to store the data collected.
    store = agl.InMemoryLightningStore()

    # This is what needs to be tuned (i.e., LLM)
    resource = agl.LLM(
        endpoint=os.environ["OPENAI_BASE_URL"], model="gpt-5", sampling_parameters={"temperature": 1.0}
    )

    made_up_task: RocmProblem = {
        "id": "debug-1",
        "question": "What is 12 multiplied by 15?",
        "chain": "",
        "result": "180",
        "source": "debug",
    }

    another_made_up_task: RocmProblem = {
        "id": "debug-2",
        "question": "What is the square root of 256?",
        "chain": "",
        "result": "16",
        "source": "debug",
    }

    # The agent here must be the same agent that will be used in the real run.
    with runner.run_context(agent=rocm_agent, store=store):
        await runner.step(
            made_up_task,
            resources={
                # The key "main_llm" here can be arbitrary
                "main_llm": resource
            },
        )

        # Run another task
        await runner.step(
            another_made_up_task,
            resources={"main_llm": resource},
        )


if __name__ == "__main__":
    asyncio.run(debug())