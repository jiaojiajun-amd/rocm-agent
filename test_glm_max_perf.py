import asyncio
import time
import argparse
from typing import List, Dict, Optional
import aiohttp
import numpy as np
from dataclasses import dataclass
from enum import Enum

class RequestStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRY_EXHAUSTED = "retry_exhausted"

@dataclass
class TestResult:
    total_tokens: int
    duration: float
    tokens_per_second: float
    request_id: int
    retry_count: int
    status: RequestStatus

class SGLangLoadTester:
    def __init__(self, 
                 base_url: str = "http://localhost:30000",
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 timeout: int = 300):
        self.base_url = base_url
        self.endpoint = f"{base_url}/v1/chat/completions"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.failed_requests = []
    
    async def send_request_with_retry(self, 
                                      session: aiohttp.ClientSession, 
                                      prompt: str, 
                                      max_tokens: int,
                                      request_id: int) -> TestResult:
        """发送请求，带指数退避重试机制"""
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                result = await self.send_request(
                    session, prompt, max_tokens, request_id, retry_count
                )
                if result:
                    return result
            except asyncio.TimeoutError:
                last_error = "Timeout"
                print(f"Request {request_id} timeout (attempt {retry_count + 1}/{self.max_retries + 1})")
            except aiohttp.ClientError as e:
                last_error = str(e)
                print(f"Request {request_id} client error: {e} (attempt {retry_count + 1}/{self.max_retries + 1})")
            except Exception as e:
                last_error = str(e)
                print(f"Request {request_id} unexpected error: {e} (attempt {retry_count + 1}/{self.max_retries + 1})")
            
            retry_count += 1
            
            if retry_count <= self.max_retries:
                # 指数退避策略
                delay = self.retry_delay * (2 ** (retry_count - 1))
                print(f"Retrying request {request_id} after {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        # 所有重试都失败
        print(f"Request {request_id} failed after {self.max_retries + 1} attempts. Last error: {last_error}")
        self.failed_requests.append({
            'request_id': request_id,
            'error': last_error,
            'retry_count': retry_count
        })
        
        return TestResult(
            total_tokens=0,
            duration=0,
            tokens_per_second=0,
            request_id=request_id,
            retry_count=retry_count,
            status=RequestStatus.RETRY_EXHAUSTED
        )
    
    async def send_request(self, 
                          session: aiohttp.ClientSession, 
                          prompt: str, 
                          max_tokens: int,
                          request_id: int,
                          retry_count: int) -> Optional[TestResult]:
        """发送单个请求并测量性能"""
        payload = {
            "model": "glm-4",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False
        }
        
        start_time = time.time()
        
        try:
            async with session.post(
                self.endpoint, 
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise aiohttp.ClientError(
                        f"HTTP {response.status}: {error_text}"
                    )
                
                result = await response.json()
                end_time = time.time()
                
                duration = end_time - start_time
                tokens = result['usage']['completion_tokens']
                tps = tokens / duration if duration > 0 else 0
                
                print(f"✓ Request {request_id} completed: {tokens} tokens in {duration:.2f}s ({tps:.2f} TPS)")
                
                return TestResult(
                    total_tokens=tokens,
                    duration=duration,
                    tokens_per_second=tps,
                    request_id=request_id,
                    retry_count=retry_count,
                    status=RequestStatus.SUCCESS
                )
        except asyncio.TimeoutError:
            raise
        except aiohttp.ClientError:
            raise
        except Exception as e:
            raise
    
    async def run_parallel_test(self, 
                               num_requests: int,
                               prompt: str,
                               max_tokens: int = 512,
                               batch_size: Optional[int] = None) -> List[TestResult]:
        """并行运行多个请求，支持分批执行"""
        
        # 创建连接池配置
        connector = aiohttp.TCPConnector(
            limit=100,  # 最大并发连接数
            limit_per_host=50,  # 每个主机的最大连接数
            ttl_dns_cache=300  # DNS缓存时间
        )
        
        timeout = aiohttp.ClientTimeout(
            total=self.timeout,
            connect=30,
            sock_read=self.timeout
        )
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        ) as session:
            
            if batch_size and batch_size < num_requests:
                # 分批执行
                print(f"使用分批模式: 每批 {batch_size} 个请求")
                all_results = []
                
                for batch_start in range(0, num_requests, batch_size):
                    batch_end = min(batch_start + batch_size, num_requests)
                    batch_num = batch_start // batch_size + 1
                    total_batches = (num_requests + batch_size - 1) // batch_size
                    
                    print(f"\n--- 执行批次 {batch_num}/{total_batches} (请求 {batch_start}-{batch_end-1}) ---")
                    
                    tasks = [
                        self.send_request_with_retry(session, prompt, max_tokens, i)
                        for i in range(batch_start, batch_end)
                    ]
                    
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # 处理异常
                    for i, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            print(f"Batch request {batch_start + i} raised exception: {result}")
                            all_results.append(TestResult(
                                total_tokens=0,
                                duration=0,
                                tokens_per_second=0,
                                request_id=batch_start + i,
                                retry_count=self.max_retries + 1,
                                status=RequestStatus.FAILED
                            ))
                        else:
                            all_results.append(result)
                    
                    # 批次间短暂延迟，避免服务器过载
                    if batch_end < num_requests:
                        await asyncio.sleep(0.5)
                
                return all_results
            else:
                # 一次性并发执行
                print(f"使用全并发模式: {num_requests} 个请求")
                tasks = [
                    self.send_request_with_retry(session, prompt, max_tokens, i)
                    for i in range(num_requests)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 处理异常
                processed_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"Request {i} raised exception: {result}")
                        processed_results.append(TestResult(
                            total_tokens=0,
                            duration=0,
                            tokens_per_second=0,
                            request_id=i,
                            retry_count=self.max_retries + 1,
                            status=RequestStatus.FAILED
                        ))
                    else:
                        processed_results.append(result)
                
                return processed_results
    
    def analyze_results(self, results: List[TestResult]):
        """分析测试结果"""
        if not results:
            print("No requests completed!")
            return
        
        # 分离成功和失败的请求
        successful_results = [r for r in results if r.status == RequestStatus.SUCCESS]
        failed_results = [r for r in results if r.status != RequestStatus.SUCCESS]
        
        print("\n" + "="*70)
        print("GLM-4 SGLang 性能测试结果")
        print("="*70)
        
        # 总体统计
        print(f"\n--- 总体统计 ---")
        print(f"总请求数: {len(results)}")
        print(f"成功请求: {len(successful_results)} ({len(successful_results)/len(results)*100:.1f}%)")
        print(f"失败请求: {len(failed_results)} ({len(failed_results)/len(results)*100:.1f}%)")
        
        # 重试统计
        retry_counts = [r.retry_count for r in results]
        total_retries = sum(retry_counts)
        print(f"\n--- 重试统计 ---")
        print(f"总重试次数: {total_retries}")
        print(f"平均重试次数: {np.mean(retry_counts):.2f}")
        print(f"最大重试次数: {max(retry_counts)}")
        
        if not successful_results:
            print("\n⚠️  没有成功的请求！")
            if self.failed_requests:
                print("\n失败请求详情:")
                for req in self.failed_requests[:10]:  # 只显示前10个
                    print(f"  Request {req['request_id']}: {req['error']}")
            return
        
        # 成功请求的性能统计
        tps_values = [r.tokens_per_second for r in successful_results]
        durations = [r.duration for r in successful_results]
        total_tokens = sum(r.total_tokens for r in successful_results)
        
        print(f"\n--- Token 生成统计 ---")
        print(f"总生成 tokens: {total_tokens}")
        print(f"平均每请求 tokens: {total_tokens/len(successful_results):.1f}")
        
        print(f"\n--- Token/秒 统计 (仅成功请求) ---")
        print(f"平均 TPS: {np.mean(tps_values):.2f}")
        print(f"中位数 TPS: {np.median(tps_values):.2f}")
        print(f"最大 TPS: {np.max(tps_values):.2f}")
        print(f"最小 TPS: {np.min(tps_values):.2f}")
        print(f"标准差: {np.std(tps_values):.2f}")
        
        print(f"\n--- 延迟统计 (秒) ---")
        print(f"平均延迟: {np.mean(durations):.2f}")
        print(f"中位数延迟: {np.median(durations):.2f}")
        print(f"P50 延迟: {np.percentile(durations, 50):.2f}")
        print(f"P90 延迟: {np.percentile(durations, 90):.2f}")
        print(f"P95 延迟: {np.percentile(durations, 95):.2f}")
        print(f"P99 延迟: {np.percentile(durations, 99):.2f}")
        print(f"最大延迟: {np.max(durations):.2f}")
        
        # 计算总吞吐量
        total_time = max(durations) if durations else 0
        throughput = total_tokens / total_time if total_time > 0 else 0
        
        print(f"\n--- 整体吞吐量 ---")
        print(f"峰值吞吐量: {throughput:.2f} tokens/秒")
        print(f"平均吞吐量: {total_tokens / sum(durations):.2f} tokens/秒" if sum(durations) > 0 else "N/A")
        
        # 失败请求详情
        if failed_results:
            print(f"\n--- 失败请求详情 ---")
            for result in failed_results[:5]:  # 只显示前5个
                print(f"Request {result.request_id}: {result.status.value} (重试 {result.retry_count} 次)")
            if len(failed_results) > 5:
                print(f"... 还有 {len(failed_results) - 5} 个失败请求")
        
        print("="*70 + "\n")
        
        # 性能建议
        self.print_recommendations(successful_results, failed_results)
    
    def print_recommendations(self, successful_results: List[TestResult], failed_results: List[TestResult]):
        """根据测试结果提供优化建议"""
        print("--- 优化建议 ---")
        
        failure_rate = len(failed_results) / (len(successful_results) + len(failed_results))
        
        if failure_rate > 0.1:
            print("⚠️  失败率较高 (>10%)，建议:")
            print("   • 减少并发请求数 (--num-requests)")
            print("   • 使用分批模式 (--batch-size)")
            print("   • 增加超时时间 (--timeout)")
            print("   • 检查 SGLang 服务器资源使用情况")
        
        if successful_results:
            avg_tps = np.mean([r.tokens_per_second for r in successful_results])
            if avg_tps < 50:
                print("⚠️  平均 TPS 较低 (<50)，建议:")
                print("   • 检查 GPU 利用率")
                print("   • 调整 SGLang 的 tensor_parallel_size")
                print("   • 优化 max_tokens 参数")
                print("   • 检查是否有其他进程占用 GPU")
        
        if successful_results:
            durations = [r.duration for r in successful_results]
            p99 = np.percentile(durations, 99)
            median = np.median(durations)
            
            if p99 > median * 3:
                print("⚠️  延迟波动较大 (P99 > 3x median)，建议:")
                print("   • 使用分批模式减少瞬时负载")
                print("   • 检查网络稳定性")
                print("   • 增加重试延迟时间")
        
        print()

async def main():
    parser = argparse.ArgumentParser(
        description='SGLang GLM-4 并行性能测试 (带重试机制)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础测试
  python test_sglang_glm4.py --num-requests 10
  
  # 高并发测试（分批执行）
  python test_sglang_glm4.py --num-requests 100 --batch-size 20
  
  # 自定义重试策略
  python test_sglang_glm4.py --num-requests 50 --max-retries 5 --retry-delay 2.0
  
  # 长文本生成测试
  python test_sglang_glm4.py --max-tokens 2048 --timeout 600
        """
    )
    
    # 服务配置
    parser.add_argument('--url', type=str, default='http://localhost:30000',
                       help='SGLang 服务地址 (默认: http://localhost:30000)')
    
    # 测试参数
    parser.add_argument('--num-requests', type=int, default=10,
                       help='并行请求总数 (默认: 10)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='分批执行的批次大小，None表示不分批 (默认: None)')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='每个请求最大生成 tokens (默认: 512)')
    parser.add_argument('--prompt', type=str, 
                       default='请详细介绍一下人工智能的发展历史和未来趋势。',
                       help='测试提示词')
    
    # 重试配置
    parser.add_argument('--max-retries', type=int, default=3,
                       help='最大重试次数 (默认: 3)')
    parser.add_argument('--retry-delay', type=float, default=1.0,
                       help='初始重试延迟(秒)，使用指数退避 (默认: 1.0)')
    parser.add_argument('--timeout', type=int, default=8000,
                       help='单个请求超时时间(秒) (默认: 300)')
    
    # 输出配置
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细日志')
    parser.add_argument('--save-results', type=str, default=None,
                       help='保存结果到 JSON 文件')
    
    args = parser.parse_args()
    
    # 打印测试配置
    print("="*70)
    print("SGLang GLM-4 性能测试配置")
    print("="*70)
    print(f"服务地址: {args.url}")
    print(f"并行请求数: {args.num_requests}")
    if args.batch_size:
        print(f"分批大小: {args.batch_size}")
    print(f"最大 tokens: {args.max_tokens}")
    print(f"最大重试次数: {args.max_retries}")
    print(f"重试延迟: {args.retry_delay}s (指数退避)")
    print(f"请求超时: {args.timeout}s")
    print(f"提示词: {args.prompt[:50]}..." if len(args.prompt) > 50 else f"提示词: {args.prompt}")
    print("="*70 + "\n")
    
    # 创建测试器
    tester = SGLangLoadTester(
        base_url=args.url,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        timeout=args.timeout
    )
    
    # 运行测试
    print("开始测试...\n")
    overall_start_time = time.time()
    
    results = await tester.run_parallel_test(
        num_requests=args.num_requests,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size
    )
    
    overall_end_time = time.time()
    total_test_time = overall_end_time - overall_start_time
    
    print(f"\n总测试时间: {total_test_time:.2f} 秒")
    
    # 分析结果
    tester.analyze_results(results)
    
    # 保存结果到文件
    if args.save_results:
        import json
        
        results_data = {
            'config': {
                'url': args.url,
                'num_requests': args.num_requests,
                'batch_size': args.batch_size,
                'max_tokens': args.max_tokens,
                'max_retries': args.max_retries,
                'retry_delay': args.retry_delay,
                'timeout': args.timeout,
                'prompt': args.prompt
            },
            'summary': {
                'total_test_time': total_test_time,
                'total_requests': len(results),
                'successful_requests': len([r for r in results if r.status == RequestStatus.SUCCESS]),
                'failed_requests': len([r for r in results if r.status != RequestStatus.SUCCESS])
            },
            'results': [
                {
                    'request_id': r.request_id,
                    'total_tokens': r.total_tokens,
                    'duration': r.duration,
                    'tokens_per_second': r.tokens_per_second,
                    'retry_count': r.retry_count,
                    'status': r.status.value
                }
                for r in results
            ]
        }
        
        with open(args.save_results, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 结果已保存到: {args.save_results}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n\n测试出错: {e}")
        import traceback
        traceback.print_exc()


