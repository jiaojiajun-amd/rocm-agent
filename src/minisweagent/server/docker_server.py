import logging
import os
import shlex
import subprocess
import uuid
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- 复用并稍作修改的原有逻辑 ---

# 直接使用原始的 DockerEnvironmentConfig dataclass，但用 Pydantic BaseModel 替代以便于API验证
class DockerEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    env: dict[str, str] = Field(default_factory=dict)
    forward_env: list[str] = Field(default_factory=list)
    timeout: int = 30
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    run_args: list[str] = Field(default_factory=lambda: ["--rm"])
    container_timeout: str = "2h"
    pull_timeout: int = 400

# --- FastAPI 应用 ---

app = FastAPI(
    title="Remote Docker Environment Server",
    description="A server to manage Docker containers remotely for distributed training.",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docker_server")

# --- API 请求体模型 ---

class StartRequest(BaseModel):
    config: DockerEnvironmentConfig

class ExecuteRequest(BaseModel):
    container_id: str
    command: str
    cwd: str = ""
    timeout: int | None = None
    env: dict[str, str] = Field(default_factory=dict)
    forward_env: list[str] = Field(default_factory=list)
    executable: str = "docker"

class CleanupRequest(BaseModel):
    container_id: str
    executable: str = "docker"


@app.post("/start")
async def start_container(request: StartRequest) -> dict[str, Any]:
    """
    Starts a new Docker container based on the provided configuration.
    """
    config = request.config
    container_name = f"minisweagent-remote-{uuid.uuid4().hex[:8]}"
    cmd = [
        config.executable,
        "run",
        "-d",
        "--name",
        container_name,
        "-w",
        config.cwd,
        *config.run_args,
        config.image,
        "sleep",
        config.container_timeout,
    ]
    logger.info(f"Starting container with command: {shlex.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.pull_timeout,
            check=True,
        )
        container_id = result.stdout.strip()
        logger.info(f"Started container {container_name} with ID {container_id}")
        return {"container_id": container_id, "status": "started"}
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start container: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Failed to start container: {e.stderr}")
    except subprocess.TimeoutExpired as e:
        logger.error(f"Timeout while starting/pulling container: {e}")
        raise HTTPException(status_code=500, detail=f"Timeout while starting/pulling container")


@app.post("/execute")
async def execute_command(request: ExecuteRequest) -> dict[str, Any]:
    """
    Executes a command inside a specific container.
    """
    cmd = [request.executable, "exec", "-w", request.cwd or "/"]
    
    # 环境变量处理
    for key in request.forward_env:
        if (value := os.getenv(key)) is not None:
            cmd.extend(["-e", f"{key}={value}"])
    for key, value in request.env.items():
        cmd.extend(["-e", f"{key}={value}"])
            
    cmd.extend([request.container_id, "bash", "-lc", request.command])

    logger.info(f"Executing in {request.container_id[:12]}: {request.command}")
    
    try:
        result = subprocess.run(
            cmd,
            text=True,
            timeout=request.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"output": "Command timed out.", "returncode": 124}


@app.post("/cleanup")
async def cleanup_container(request: CleanupRequest) -> dict[str, str]:
    """
    Stops and removes a container.
    """
    container_id = request.container_id
    executable = request.executable
    logger.info(f"Cleaning up container {container_id[:12]}")
    # 使用非阻塞方式在后台清理
    cmd = f"(timeout 60 {executable} stop {container_id} || {executable} rm -f {container_id}) >/dev/null 2>&1 &"
    subprocess.Popen(cmd, shell=True)
    return {"status": "cleanup process started", "container_id": container_id}

if __name__ == "__main__":
    import uvicorn
    # 监听所有网络接口，以便从其他机器访问
    uvicorn.run(app, host="0.0.0.0", port=9527)
