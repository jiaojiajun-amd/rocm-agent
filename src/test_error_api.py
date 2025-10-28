import json
import logging
from pathlib import Path
from typing import Optional
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("error_replay_test")


class ErrorMessageReplay:
    """读取保存的错误消息并重新调用模型进行测试"""
    
    def __init__(self, endpoint: str = "http://localhost:8000/v1", api_key: str = "EMPTY"):
        """
        初始化重放测试器
        
        Args:
            endpoint: vLLM 服务端点
            api_key: API key（本地部署通常使用 "EMPTY"）
        """
        self.endpoint = endpoint
        self.client = OpenAI(
            api_key=api_key,
            base_url=endpoint,
            timeout=None,  # 设置为 None 表示无超时限制
            max_retries=0,
        )
        logger.info(f"Initialized client with endpoint: {endpoint}")
    
    def load_error_file(self, filepath: str) -> dict:
        """
        加载错误日志文件
        
        Args:
            filepath: 错误日志文件路径
            
        Returns:
            错误日志数据
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded error file: {filepath}")
            logger.info(f"Error type: {data.get('error_type')}")
            logger.info(f"Original error: {data.get('error_message')}")
            return data
        except Exception as e:
            logger.error(f"Failed to load error file {filepath}: {e}")
            raise
    
    def replay_messages(self, messages: list[dict[str, str]], model: str = "Qwen/Qwen3-8B") -> dict:
        """
        重新发送消息到模型
        
        Args:
            messages: 消息列表
            model: 模型名称
            
        Returns:
            模型响应
        """
        try:
            logger.info(f"Replaying {len(messages)} messages to model: {model}")
            
            # 打印消息内容
            for i, msg in enumerate(messages):
                logger.info(f"Message {i+1} - Role: {msg.get('role')}, Content preview: {msg.get('content', '')[:100]}...")
            
            # 调用模型
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
            )
            
            # 格式化响应（与原代码保持一致）
            formatted_response = {
                "choices": [
                    {
                        "message": {
                            "role": response.choices[0].message.role,
                            "content": response.choices[0].message.content,
                        },
                        "finish_reason": response.choices[0].finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "model": response.model,
                "id": response.id,
            }
            
            logger.info("✅ Successfully received response")
            logger.info(f"Response content: {formatted_response['choices'][0]['message']['content'][:200]}...")
            logger.info(f"Usage: {formatted_response['usage']}")
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"❌ Failed to replay messages: {type(e).__name__}: {e}")
            raise
    
    def test_single_file(self, filepath: str, model: str = "Qwen/Qwen3-8B") -> bool:
        """
        测试单个错误日志文件
        
        Args:
            filepath: 错误日志文件路径
            model: 模型名称
            
        Returns:
            测试是否成功
        """
        try:
            # 加载错误文件
            error_data = self.load_error_file(filepath)
            
            # 提取消息
            messages = error_data.get('messages', [])
            if not messages:
                logger.warning("No messages found in error file")
                return False
            
            # 重新发送消息
            response = self.replay_messages(messages, model)
            
            # 保存测试结果
            self._save_test_result(filepath, error_data, response)
            
            return True
            
        except Exception as e:
            logger.error(f"Test failed for {filepath}: {e}")
            return False
    
    def test_all_errors(self, error_log_dir: str = "./error_logs", model: str = "Qwen/Qwen3-8B"):
        """
        测试所有错误日志文件
        
        Args:
            error_log_dir: 错误日志目录
            model: 模型名称
        """
        error_dir = Path(error_log_dir)
        if not error_dir.exists():
            logger.error(f"Error log directory not found: {error_log_dir}")
            return
        
        # 查找所有 JSON 文件
        error_files = list(error_dir.glob("**/*.json"))
        logger.info(f"Found {len(error_files)} error log files")
        
        success_count = 0
        fail_count = 0
        
        for error_file in error_files:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {error_file.name}")
            logger.info(f"{'='*60}")
            
            if self.test_single_file(str(error_file), model):
                success_count += 1
            else:
                fail_count += 1
        
        # 打印总结
        logger.info(f"\n{'='*60}")
        logger.info(f"Test Summary:")
        logger.info(f"Total: {len(error_files)}, Success: {success_count}, Failed: {fail_count}")
        logger.info(f"{'='*60}")
    
    def _save_test_result(self, original_filepath: str, error_data: dict, response: dict):
        """
        保存测试结果
        
        Args:
            original_filepath: 原始错误文件路径
            error_data: 原始错误数据
            response: 模型响应
        """
        result_dir = Path("./test_results")
        result_dir.mkdir(parents=True, exist_ok=True)
        
        original_path = Path(original_filepath)
        result_filename = f"result_{original_path.stem}.json"
        result_filepath = result_dir / result_filename
        
        result_data = {
            "original_error": {
                "timestamp": error_data.get("timestamp"),
                "error_type": error_data.get("error_type"),
                "error_message": error_data.get("error_message"),
            },
            "test_info": {
                "test_timestamp": json.dumps({"timestamp": str(Path(__file__).stat().st_mtime)}),
                "endpoint": self.endpoint,
            },
            "messages": error_data.get("messages"),
            "response": response,
        }
        
        try:
            with open(result_filepath, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Test result saved to: {result_filepath}")
        except Exception as e:
            logger.error(f"Failed to save test result: {e}")


def main():
    """主测试函数"""
    
    # 配置参数
    ENDPOINT = "http://localhost:8000/v1"  # 修改为你的 vLLM 端点
    MODEL_NAME = "Qwen/Qwen3-8B"  # 修改为你部署的模型名称
    ERROR_LOG_DIR = "./error_logs"
    
    print("="*60)
    print("Error Message Replay Test")
    print("="*60)
    print(f"Endpoint: {ENDPOINT}")
    print(f"Model: {MODEL_NAME}")
    print(f"Error log directory: {ERROR_LOG_DIR}")
    print("="*60)
    
    # 初始化测试器
    replayer = ErrorMessageReplay(endpoint=ENDPOINT, api_key="EMPTY")
    
    # 测试连接
    print("\n[1] Testing connection to vLLM service...")
    try:
        models = replayer.client.models.list()
        print(f"✅ Connection successful! Available models:")
        for model in models.data:
            print(f"   - {model.id}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("Please ensure vLLM service is running.")
        return
    
    # 选择测试模式
    print("\n[2] Select test mode:")
    print("   1. Test all error files")
    print("   2. Test specific error file")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # 测试所有错误文件
        print(f"\n[3] Testing all error files in {ERROR_LOG_DIR}...")
        replayer.test_all_errors(ERROR_LOG_DIR, MODEL_NAME)
        
    elif choice == "2":
        # 测试指定文件
        filepath = input("Enter error file path: ").strip()
        print(f"\n[3] Testing single file: {filepath}...")
        success = replayer.test_single_file(filepath, MODEL_NAME)
        if success:
            print("✅ Test completed successfully")
        else:
            print("❌ Test failed")
    else:
        print("Invalid choice")
        return
    
    print("\n" + "="*60)
    print("Test completed! Check ./test_results/ for detailed results.")
    print("="*60)


def quick_test():
    """快速测试：直接测试一个简单的消息"""
    
    ENDPOINT = "http://localhost:8000/v1"
    MODEL_NAME = "Qwen/Qwen3-8B"
    
    print("="*60)
    print("Quick Test - Direct Message Test")
    print("="*60)
    
    replayer = ErrorMessageReplay(endpoint=ENDPOINT, api_key="EMPTY")
    
    # 测试消息
    test_messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": "Hello! Can you introduce yourself?"
        }
    ]
    
    try:
        print("\n[Testing with simple messages...]")
        response = replayer.replay_messages(test_messages, MODEL_NAME)
        
        print("\n" + "="*60)
        print("Response:")
        print("="*60)
        print(response['choices'][0]['message']['content'])
        print("\n" + "="*60)
        print(f"Tokens used: {response['usage']['total_tokens']}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")


if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            # 快速测试模式
            quick_test()
        elif sys.argv[1] == "--file":
            # 测试指定文件
            if len(sys.argv) < 3:
                print("Usage: python test_error_replay.py --file <error_file_path>")
                sys.exit(1)
            
            ENDPOINT = "http://localhost:8000/v1"
            MODEL_NAME = "Qwen/Qwen3-8B"
            
            replayer = ErrorMessageReplay(endpoint=ENDPOINT, api_key="EMPTY")
            replayer.test_single_file(sys.argv[2], MODEL_NAME)
        elif sys.argv[1] == "--all":
            # 测试所有文件
            ENDPOINT = "http://localhost:8000/v1"
            MODEL_NAME = "Qwen/Qwen3-8B"
            ERROR_LOG_DIR = sys.argv[2] if len(sys.argv) > 2 else "./error_logs"
            
            replayer = ErrorMessageReplay(endpoint=ENDPOINT, api_key="EMPTY")
            replayer.test_all_errors(ERROR_LOG_DIR, MODEL_NAME)
        else:
            print("Unknown option. Available options:")
            print("  --quick          : Run quick test with simple message")
            print("  --file <path>    : Test specific error file")
            print("  --all [dir]      : Test all error files in directory")
    else:
        # 交互式模式
        main()

