# Copyright (c) Microsoft. All rights reserved.

# Copied and adapted from https://github.com/prompteus/calc-x/blob/master/gadgets/metrics.py

import asyncio
import math
import re
import string
import json

import sympy
import aiohttp

from agentlightning.reward import reward
from minisweagent.utils.log import add_file_handler, logger
from typing import Any, Dict, Tuple


async def evaluate(exit_status, result, container_id, instance_id, dataset_name, split, eval_server_url):
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
        return 0.0, -1.0
        
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
    
    logger.info(f"Sending evaluation request to {eval_server_url}/evaluate_v3")
    logger.debug(f"Payload: {payload}")
    
    timeout = aiohttp.ClientTimeout(total=3600)
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
    
    try:
        # 使用 aiohttp 进行异步 HTTP 请求，支持高并发
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async with session.post(
                f"{eval_server_url}/evaluate_v3",
                json=payload
            ) as resp:
                logger.info(f"Received response with status code: {resp.status}")
                
                # 读取响应文本
                resp_text = await resp.text()
                
                # 检查 HTTP 状态码
                if resp.status >= 400:
                    logger.error(f"HTTP error {resp.status}: {resp_text[:500]}")
                    logger.error("="*80)
                    return 0.0, -1.0
                
                # 解析响应
                data = json.loads(resp_text)
                logger.info(f"Response data: {json.dumps(data, indent=2)}")
                
                # 检查评估是否成功
                if not data.get("success", False):
                    error_msg = data.get("error", "Unknown error")
                    logger.error(f"Evaluation failed: {error_msg}")
                    
                    # 记录详细错误信息
                    if "error_detail" in data:
                        logger.error(f"Error detail: {data['error_detail']}")
                    if "build_output" in data:
                        logger.error(f"Build output: {data['build_output'][:500]}...")
                    
                    return 0.0, -1.0
                
                # 获取 reward
                reward = float(data.get("reward", 0.0))
                speedup = float(data.get("speedup", -1.0))
                
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
                return reward, speedup
    
    except asyncio.TimeoutError as e:
        logger.error(f"Request timeout after 3600 seconds: {str(e)}")
        logger.error("="*80)
        return 0.0, -1.0
    
    except aiohttp.ClientError as e:
        logger.error(f"HTTP client error occurred: {str(e)}")
        logger.error("="*80)
        return 0.0, -1.0
    
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Error parsing response: {str(e)}")
        logger.error("="*80)
        return 0.0, -1.0
    
    except Exception as e:
        logger.error(f"Unexpected error in evaluate: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error("="*80)
        return 0.0, -1.0




async def evaluate_info(
    exit_status: str,
    result: Any,
    container_id: str,
    instance_id: str,
    dataset_name: str,
    split: str,
    eval_server_url: str,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    带有详细信息的评估函数:
    - 仍返回 reward, speedup（与 evaluate 行为兼容）
    - 额外返回 evaluation_info (dict)，包含请求/响应的详细信息以及日志辅助信息

    Returns:
        (reward, speedup, evaluation_info)
    """
    logger.info("="*80)
    logger.info("evaluate_info function called")
    logger.info(f"Exit status: {exit_status}")
    logger.info(f"Instance ID: {instance_id}")
    logger.info(f"Container ID: {container_id}")
    logger.info(f"Dataset: {dataset_name}, Split: {split}")
    logger.info(f"Eval server URL: {eval_server_url}")
    logger.info("="*80)

    evaluation_info: Dict[str, Any] = {
        "request": {
            "url": f"{eval_server_url}/evaluate_v3",
            "payload": None,
            "timeout_sec": 3600,
        },
        "response": {
            "status_code": None,
            "raw": None,           # 原始文本
            "json": None,          # JSON 解析结果
        },
        "meta": {
            "reason": None,        # 非 Submitted 或异常时的原因
            "success": False,
            "error": None,
        },
        "extracted": {
            "reward": 0.0,
            "speedup": -1.0,
            "exit_code": None,
            "timed_out": None,
            "gpu_id": None,
            "stdout_preview": None,
        },
    }

    # 如果未提交，直接返回
    if exit_status != "Submitted":
        msg = f"Exit status is '{exit_status}', not 'Submitted'. Returning reward 0.0"
        logger.warning(msg)
        evaluation_info["meta"]["reason"] = msg
        return 0.0, -1.0, evaluation_info

    # 构建请求负载
    payload = {
        "instance_id": instance_id,
        "container_id": container_id,
        "mode": "benchmark",
        # 可选参数:
        # "args": [],
        # "timeout": 1800,
        # "gpu_acquire_timeout": 300
    }
    evaluation_info["request"]["payload"] = payload
    logger.info(f"Sending evaluation request to {eval_server_url}/evaluate_v3")
    logger.debug(f"Payload: {payload}")

    timeout = aiohttp.ClientTimeout(total=evaluation_info["request"]["timeout_sec"])
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
    
    try:
        # 使用 aiohttp 进行异步 HTTP 请求，支持高并发
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async with session.post(
                f"{eval_server_url}/evaluate_v3",
                json=payload
            ) as resp:
                evaluation_info["response"]["status_code"] = resp.status
                resp_text = await resp.text()
                evaluation_info["response"]["raw"] = resp_text

                logger.info(f"Received response with status code: {resp.status}")
                
                # 检查 HTTP 状态码
                if resp.status >= 400:
                    logger.error(f"HTTP error {resp.status}: {resp_text[:500]}")
                    evaluation_info["meta"]["error"] = f"http_status_{resp.status}"
                    logger.error("="*80)
                    return 0.0, -1.0, evaluation_info

                data = json.loads(resp_text)
                evaluation_info["response"]["json"] = data
                logger.info(f"Response data: {json.dumps(data, indent=2)}")

                # success 检查
                if not data.get("success", False):
                    error_msg = data.get("error", "Unknown error")
                    evaluation_info["meta"]["success"] = False
                    evaluation_info["meta"]["error"] = error_msg

                    logger.error(f"Evaluation failed: {error_msg}")
                    # 记录详细错误信息
                    if "error_detail" in data:
                        logger.error(f"Error detail: {data['error_detail']}")
                    if "build_output" in data:
                        logger.error(f"Build output: {data['build_output'][:500]}...")
                    # 将这些信息也放进 evaluation_info
                    for key in ["error_detail", "build_output", "stdout", "stderr"]:
                        if key in data:
                            evaluation_info["extracted"][key] = data[key]
                    return 0.0, -1.0, evaluation_info

                # 提取 reward/speedup
                reward_val = float(data.get("reward", 0.0))
                speedup_val = float(data.get("speedup", -1.0))

                # 额外字段
                exit_code = data.get("exit_code", -1)
                timed_out = data.get("timed_out", False)
                gpu_id = data.get("gpu_id", "N/A")

                evaluation_info["meta"]["success"] = True
                evaluation_info["extracted"].update({
                    "reward": reward_val,
                    "speedup": speedup_val,
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                    "gpu_id": gpu_id,
                })

                # 可选：记录输出预览
                if "stdout" in data and data["stdout"]:
                    stdout_preview = data["stdout"][:200]
                    logger.debug(f"Stdout preview: {stdout_preview}...")
                    evaluation_info["extracted"]["stdout_preview"] = stdout_preview

                logger.info("Evaluation completed successfully")
                logger.info(f"Reward: {reward_val}")
                logger.info(f"Exit code: {exit_code}")
                logger.info(f"Timed out: {timed_out}")
                logger.info(f"GPU ID: {gpu_id}")
                logger.info("="*80)

                return reward_val, speedup_val, evaluation_info

    except asyncio.TimeoutError as e:
        logger.error(f"Request timeout after {evaluation_info['request']['timeout_sec']} seconds: {str(e)}")
        logger.error("="*80)
        evaluation_info["meta"]["error"] = f"timeout: {str(e)}"
        return 0.0, -1.0, evaluation_info

    except aiohttp.ClientError as e:
        logger.error(f"HTTP client error occurred: {str(e)}")
        logger.error("="*80)
        evaluation_info["meta"]["error"] = f"client_error: {str(e)}"
        return 0.0, -1.0, evaluation_info

    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Error parsing response: {str(e)}")
        if evaluation_info["response"]["raw"]:
            logger.error(f"Response data: {evaluation_info['response']['raw'][:500]}")
        logger.error("="*80)
        evaluation_info["meta"]["error"] = f"parse_error: {str(e)}"
        return 0.0, -1.0, evaluation_info

    except Exception as e:
        logger.error(f"Unexpected error in evaluate_info: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Traceback: {tb}")
        logger.error("="*80)
        evaluation_info["meta"]["error"] = f"unexpected: {str(e)}"
        evaluation_info["meta"]["traceback"] = tb
        return 0.0, -1.0, evaluation_info

