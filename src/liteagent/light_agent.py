


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


def get_hip_bench_docker_image_name(instance: dict) -> str:
    image_name = instance.get("image_name", "rocm-lib")
    return image_name

def get_hip_environment(config: dict, instance: dict, server_url: str | None = None) -> Environment:
    """
    Get the environment for a SWE-bench instance.
    If environment_class is 'docker_remote', it creates a RemoteDockerEnvironment.
    Otherwise, it falls back to the default local environment creation.
    """
    env_config = config.setdefault("environment", {})
    env_class = env_config.get("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)
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

    env = get_sb_environment(config, instance, server_url=server_url)
    agent = DefaultAgent(
        model=model,
        env=env,
        **config.get("agent", {}),
    )
    return agent

@reward
def get_reward(exit_status, result, container_id, instance_id, dataset_name, split, eval_server_url):
    """
    获取奖励分数
    
    Args:
        exit_status: 提交状态
        result: 结果对象（可能包含额外信息）
        container_id: 容器ID
        instance_id: 实例ID
        dataset_name: 数据集名称
        split: 数据集分割
        eval_server_url: 评估服务器URL
        
    Returns:
        float: 奖励分数 (0.0 或 1.0)
    """
    logger.info("="*80)
    logger.info("get_reward function called")
    logger.info(f"Exit status: {exit_status}")
    logger.info(f"Instance ID: {instance_id}")
    logger.info(f"Container ID: {container_id}")
    logger.info(f"Dataset: {dataset_name}, Split: {split}")
    logger.info(f"Eval server URL: {eval_server_url}")
    logger.info("="*80)
    
    # 检查提交状态
    if exit_status != "Submitted":
        logger.warning(f"Exit status is '{exit_status}', not 'Submitted'. Returning reward 0.0")
        return 0.0
    
    # 构建评估请求
    payload = {
        "instance_id": instance_id,
        "container_id": container_id,
        "mode": "benchmark",  # 默认使用 benchmark，可以根据需要调整
        # 可选参数
        # "args": [],
        # "timeout": 1800,
        # "gpu_acquire_timeout": 300
    }
    
    logger.info(f"Sending evaluation request to {eval_server_url}/evaluate")
    logger.debug(f"Payload: {payload}")
    
    try:
        # 发送评估请求
        resp = requests.post(
            f"{eval_server_url}/evaluate", 
            json=payload, 
            timeout=3600  # 增加超时时间，因为包含编译和执行
        )
        
        logger.info(f"Received response with status code: {resp.status_code}")
        
        # 检查 HTTP 状态码
        resp.raise_for_status()
        
        # 解析响应
        data = resp.json()
        logger.info(f"Response data: {json.dumps(data, indent=2)}")
        
        # 检查评估是否成功
        if not data.get("success", False):
            error_msg = data.get("error", "Unknown error")
            logger.error(f"Evaluation failed: {error_msg}")
            
            # 记录详细错误信息
            if "error_detail" in data:
                logger.error(f"Error detail: {data['error_detail']}")
            if "build_output" in data:
                logger.error(f"Build output: {data['build_output'][:500]}...")  # 只记录前500字符
            
            return 0.0
        
        # 获取 reward
        reward = float(data.get("reward", 0.0))
        
        # 记录详细信息
        exit_code = data.get("exit_code", -1)
        timed_out = data.get("timed_out", False)
        gpu_id = data.get("gpu_id", "N/A")
        
        logger.info(f"Evaluation completed successfully")
        logger.info(f"Reward: {reward}")
        logger.info(f"Exit code: {exit_code}")
        logger.info(f"Timed out: {timed_out}")
        logger.info(f"GPU ID: {gpu_id}")
        
        # 可选：记录输出的前几行
        if "stdout" in data and data["stdout"]:
            stdout_preview = data["stdout"][:200]
            logger.debug(f"Stdout preview: {stdout_preview}...")
        
        logger.info("="*80)
        return reward
    
    except requests.Timeout as e:
        logger.error(f"Request timeout after 3600 seconds: {str(e)}")
        logger.error("="*80)
        return 0.0
    
    except requests.HTTPError as e:
        logger.error(f"HTTP error occurred: {str(e)}")
        logger.error(f"Response status code: {resp.status_code}")
        try:
            error_data = resp.json()
            logger.error(f"Error response: {json.dumps(error_data, indent=2)}")
        except:
            logger.error(f"Response text: {resp.text[:500]}")
        logger.error("="*80)
        return 0.0
    
    except requests.ConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        logger.error(f"Could not connect to evaluation server at {eval_server_url}")
        logger.error("="*80)
        return 0.0
    
    except requests.RequestException as e:
        logger.error(f"Request exception occurred: {str(e)}")
        logger.error("="*80)
        return 0.0
    
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Error parsing response: {str(e)}")
        logger.error(f"Response data: {resp.text[:500]}")
        logger.error("="*80)
        return 0.0
    
    except Exception as e:
        logger.error(f"Unexpected error in get_reward: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error("="*80)
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
    Trainer(n_workers=8).fit(SweAgent(), "http://localhost:9999/")