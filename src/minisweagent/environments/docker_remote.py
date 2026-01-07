import logging
import os
import platform
import time
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
    timeout: int = 1800  # 默认 1800 秒（30 分钟）用于长时间运行的命令
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    run_args: list[str] = field(default_factory=lambda: ["--rm"])
    container_timeout: str = "6h"
    pull_timeout: int = 400
    max_retries: int = 3  # 仅用于启动容器
    retry_delay: int = 5
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
        
        # 配置 Session 以处理连接问题
        self.session = requests.Session()
        
        # 配置适配器：连接池和重试策略
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        # 配置连接池
        adapter = HTTPAdapter(
            pool_connections=5,
            pool_maxsize=10,
            max_retries=0,  # 我们自己处理重试
            pool_block=False
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # 设置 keep-alive 和连接超时
        self.session.headers.update({
            'Connection': 'keep-alive',
            'Keep-Alive': 'timeout=300, max=100'
        })
        
        try:
            self._start_remote_container()
        except Exception as e:
            self.logger.error(f"Failed to initialize remote container: {e}")
            self.container_id = None
            self.session.close()
            raise

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | platform.uname()._asdict()

    def _start_remote_container(self):
        """Sends a request to the remote server to start a container with retry logic."""
        self.logger.info("Requesting remote server to start a new container...")
        endpoint = f"{self.server_url}/start"
        payload = {"config": asdict(self.config)}
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(endpoint, json=payload, timeout=self.config.pull_timeout + 10)
                response.raise_for_status()
                
                data = response.json()
                self.container_id = data.get("container_id")

                if not self.container_id:
                    raise RuntimeError("Server did not return a container ID.")
                    
                self.logger.info(f"Remote container started with ID: {self.container_id}")
                return
                
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < self.config.max_retries - 1:
                    self.logger.warning(f"Attempt {attempt + 1}/{self.config.max_retries} failed: {e}. Retrying in {self.config.retry_delay}s...")
                    time.sleep(self.config.retry_delay)
                else:
                    self.logger.error(f"All {self.config.max_retries} attempts failed to start container.")
                    raise

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Executes a command in the remote Docker container.
        
        Note: No retry logic to avoid executing commands multiple times.
        """
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
            # 超时时间：1800 秒（30 分钟）用于长时间运行的命令
            # 如果命令指定了更长的超时，使用命令的超时 + 余量
            default_timeout = 1800
            command_timeout = timeout or self.config.timeout
            request_timeout = max(default_timeout, command_timeout + 30)
            
            response = self.session.post(
                endpoint, 
                json=payload, 
                timeout=request_timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to execute command remotely: {e}")
            return {"output": f"Error communicating with server: {e}", "returncode": -1}

    def cleanup(self):
        """Sends a request to the remote server to clean up the container with retry logic."""
        if self.container_id:
            self.logger.info(f"Requesting cleanup for remote container {self.container_id}")
            endpoint = f"{self.server_url}/cleanup"
            payload = {"container_id": self.container_id, "executable": self.config.executable}
            
            for attempt in range(self.config.max_retries):
                try:
                    self.session.post(endpoint, json=payload, timeout=10)
                    self.logger.info("Cleanup request sent successfully.")
                    break
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    if attempt < self.config.max_retries - 1:
                        self.logger.warning(f"Cleanup attempt {attempt + 1}/{self.config.max_retries} failed: {e}. Retrying in {self.config.retry_delay}s...")
                        time.sleep(self.config.retry_delay)
                    else:
                        self.logger.warning(f"Failed to send cleanup request after {self.config.max_retries} attempts (container might be orphaned): {e}")
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"Failed to send cleanup request to server (container might be orphaned): {e}")
                    break
            
            self.container_id = None
        
        self.session.close()

    def __del__(self):
        """Cleanup container when object is destroyed."""
        # Disabled: In multi-threaded environments, __del__ may trigger prematurely
        # causing "No such container" errors. Containers will be cleaned up by timeout.
        pass

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

