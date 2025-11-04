# Copyright (c) Microsoft. All rights reserved.

# Copied and adapted from https://github.com/prompteus/calc-x/blob/master/gadgets/metrics.py

import math
import re
import string

import sympy

from agentlightning.reward import reward
from minisweagent.utils.log import add_file_handler, logger
import requests, json 



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
    
    logger.info(f"Sending evaluation request to {eval_server_url}/evaluate")
    logger.debug(f"Payload: {payload}")
    
    try:
        # 发送评估请求
        resp = requests.post(
            f"{eval_server_url}/evaluate_v2", 
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
    
    except requests.Timeout as e:
        logger.error(f"Request timeout after 3600 seconds: {str(e)}")
        logger.error("="*80)
        return 0.0, -1.0
    
    except requests.HTTPError as e:
        logger.error(f"HTTP error occurred: {str(e)}")
        logger.error(f"Response status code: {resp.status_code}")
        try:
            error_data = resp.json()
            logger.error(f"Error response: {json.dumps(error_data, indent=2)}")
        except:
            logger.error(f"Response text: {resp.text[:500]}")
        logger.error("="*80)
        return 0.0, -1.0
    
    except requests.ConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        logger.error(f"Could not connect to evaluation server at {eval_server_url}")
        logger.error("="*80)
        return 0.0, -1.0
    
    except requests.RequestException as e:
        logger.error(f"Request exception occurred: {str(e)}")
        logger.error("="*80)
        return 0.0, -1.0
    
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Error parsing response: {str(e)}")
        logger.error(f"Response data: {resp.text[:500]}")
        logger.error("="*80)
        return 0.0, -1.0
    
    except Exception as e:
        logger.error(f"Unexpected error in get_reward: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error("="*80)
        return 0.0, -1.0

