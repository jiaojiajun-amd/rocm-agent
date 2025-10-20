import logging
import os
import requests
from dataclasses import asdict, dataclass, field
from typing import Any

# 复用原来的配置类，它在这里只用于存储配置，不直接参与docker命令
@dataclass
class RemoteDockerEnvironmentConfig:
    image: str
    cwd: str = "/"
    env: dict[str, str] = field(default_factory=dict)
    forward_env: list[str] = field(default_factory=list)
    timeout: int = 30
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    run_args: list[str] = field(default_factory=lambda: ["--rm"])
    container_timeout: str = "2h"
    pull_timeout: int = 400
    environment_class: str = "docker_remote"
    server_url : str = "http://localhost:9527"


class RemoteDockerEnvironment:
    def __init__(
        self,
        *,
        server_url: str,
        config_class: type = RemoteDockerEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """
        This class executes commands in a Docker container managed by a remote server.
        
        :param server_url: The URL of the remote docker server (e.g., "http://localhost:9527").
        """
        self.logger = logger or logging.getLogger("minisweagent.environment.remote")
        self.server_url = server_url.rstrip("/")
        self.container_id: str | None = None
        self.config = config_class(**kwargs)
        
        try:
            self._start_remote_container()
        except Exception as e:
            self.logger.error(f"Failed to initialize remote container: {e}")
            # 如果启动失败，确保在对象销毁时不会尝试清理一个不存在的容器
            self.container_id = None
            raise

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def _start_remote_container(self):
        """Sends a request to the remote server to start a container."""
        self.logger.info("Requesting remote server to start a new container...")
        endpoint = f"{self.server_url}/start"
        payload = {"config": asdict(self.config)}
        
        response = requests.post(endpoint, json=payload, timeout=self.config.pull_timeout + 10)
        response.raise_for_status()  # Will raise an HTTPError for bad responses (4xx or 5xx)
        
        data = response.json()
        self.container_id = data.get("container_id")

        if not self.container_id:
            raise RuntimeError("Server did not return a container ID.")
            
        self.logger.info(f"Remote container started with ID: {self.container_id}")

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Executes a command in the remote Docker container."""
        if not self.container_id:
            raise RuntimeError("Remote container is not running or was not initialized properly.")
        
        endpoint = f"{self.server_url}/execute"
        payload = {
            "container_id": self.container_id,
            "command": command,
            "cwd": cwd or self.config.cwd,
            "timeout": timeout or self.config.timeout,
            "env": self.config.env,
            "forward_env": self.config.forward_env,
            "executable": self.config.executable
        }
        
        try:
            response = requests.post(endpoint, json=payload, timeout=(timeout or self.config.timeout) + 10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to execute command remotely: {e}")
            return {"output": f"Error communicating with server: {e}", "returncode": -1}

    def cleanup(self):
        """Sends a request to the remote server to clean up the container."""
        if self.container_id:
            self.logger.info(f"Requesting cleanup for remote container {self.container_id}")
            endpoint = f"{self.server_url}/cleanup"
            payload = {"container_id": self.container_id, "executable": self.config.executable}
            try:
                # Use a short timeout for cleanup, as it's a "fire and forget" action
                requests.post(endpoint, json=payload, timeout=10)
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Failed to send cleanup request to server (container might be orphaned): {e}")
            finally:
                self.container_id = None # Prevent multiple cleanup attempts

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()

# --- 使用示例 ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # 假设你的docker_server.py正运行在 IP 为 192.168.1.100 的机器上
    REMOTE_SERVER_URL = "http://127.0.0.1:9527" # 如果在同一台机器上测试，使用localhost

    print("--- Initializing Remote Docker Environment ---")
    try:
        remote_env = RemoteDockerEnvironment(
            server_url=REMOTE_SERVER_URL,
            image="python:3.11-slim",
            timeout=15,
        )

        print("\n--- Testing command execution: 'ls -l /' ---")
        result = remote_env.execute("ls -l /")
        print(f"Return Code: {result['returncode']}")
        print(f"Output:\n{result['output']}")

        print("\n--- Testing command execution in a different CWD: 'pwd' ---")
        result_pwd = remote_env.execute("pwd", cwd="/usr/local")
        print(f"Return Code: {result_pwd['returncode']}")
        print(f"Output:\n{result_pwd['output']}")

        print("\n--- Testing timeout ---")
        result_timeout = remote_env.execute("sleep 20", timeout=5)
        print(f"Return Code: {result_timeout['returncode']}")
        print(f"Output:\n{result_timeout['output']}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    
    finally:
        # remote_env的__del__方法会在程序结束时自动调用cleanup
        # 如果你想立即清理，可以手动调用 remote_env.cleanup()
        print("\n--- End of script. Cleanup will be triggered automatically. ---")

