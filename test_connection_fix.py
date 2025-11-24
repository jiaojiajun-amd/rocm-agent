#!/usr/bin/env python3
"""
测试 HTTP 连接管理修复是否有效
"""

import requests
import time
import psutil
import os

def count_open_connections(port=None):
    """统计当前进程的打开连接数"""
    pid = os.getpid()
    process = psutil.Process(pid)
    connections = process.connections()
    
    if port:
        connections = [c for c in connections if c.raddr and c.raddr.port == port]
    
    return len(connections)

def test_without_session(url, n_requests=10):
    """测试不使用 session 的情况（旧方式）"""
    print(f"\n=== 测试不使用 Session（旧方式）===")
    print(f"初始连接数: {count_open_connections()}")
    
    for i in range(n_requests):
        try:
            requests.get(url, timeout=1)
        except:
            pass
    
    time.sleep(1)  # 等待连接关闭
    print(f"请求后连接数: {count_open_connections()}")

def test_with_session(url, n_requests=10):
    """测试使用 session 的情况（新方式）"""
    print(f"\n=== 测试使用 Session（新方式）===")
    print(f"初始连接数: {count_open_connections()}")
    
    session = requests.Session()
    for i in range(n_requests):
        try:
            session.get(url, timeout=1)
        except:
            pass
    session.close()
    
    time.sleep(1)  # 等待连接关闭
    print(f"请求后连接数: {count_open_connections()}")

def test_with_context_manager(url, n_requests=10):
    """测试使用上下文管理器的情况（新方式）"""
    print(f"\n=== 测试使用 Context Manager（新方式）===")
    print(f"初始连接数: {count_open_connections()}")
    
    with requests.Session() as session:
        for i in range(n_requests):
            try:
                session.get(url, timeout=1)
            except:
                pass
    
    time.sleep(1)  # 等待连接关闭
    print(f"请求后连接数: {count_open_connections()}")

if __name__ == "__main__":
    # 使用一个公开的测试 URL
    test_url = "http://httpbin.org/status/200"
    
    print("=" * 60)
    print("HTTP 连接管理测试")
    print("=" * 60)
    
    # 测试三种方式
    test_without_session(test_url, 5)
    test_with_session(test_url, 5)
    test_with_context_manager(test_url, 5)
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n说明：")
    print("- 使用 Session 可以复用 TCP 连接，减少连接数")
    print("- 使用 Context Manager 确保连接被正确关闭")
    print("- 修复后的代码使用这些最佳实践，避免连接泄漏")

