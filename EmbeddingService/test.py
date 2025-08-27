import statistics
import threading
import time
from collections import deque

import requests

# 配置参数
API_URL = "http://localhost:7100/v1/embeddings"
MODEL_NAME = "jina-embeddings-v4"
TEST_DURATION = 60  # 测试时长(秒)
CONCURRENT_WORKERS = 8  # 并发线程数
TEXTS = ["今天" * 100, "大模型" * 200] * 20  # 测试文本池(重复扩充)

# 全局统计变量
success_count = 0
error_count = 0
latencies = deque(maxlen=1000)  # 保留最近1000次请求的延迟


def send_request():
    global success_count, error_count
    while time.time() - start_time < TEST_DURATION:
        try:
            payload = {
                "input": TEXTS[:2],  # 每次发送2个文本
                "model": MODEL_NAME,
            }
            start_req = time.time()
            resp = requests.post(API_URL, json=payload, timeout=10)
            emebddings = resp.json().get("data", [])
            if resp.status_code == 200:
                success_count += 1
                latencies.append(time.time() - start_req)
            else:
                error_count += 1
        except Exception as e:
            error_count += 1
            print(f"Request failed: {str(e)}")


# 启动测试
start_time = time.time()
threads = []
for _ in range(CONCURRENT_WORKERS):
    t = threading.Thread(target=send_request)
    t.start()
    threads.append(t)

# 实时打印进度
print(f"🚀 开始压测(持续时间: {TEST_DURATION}秒)...")
while time.time() - start_time < TEST_DURATION:
    time.sleep(5)
    elapsed = time.time() - start_time
    print(
        f"进度: {elapsed:.1f}s | "
        f"成功: {success_count} | "
        f"失败: {error_count} | "
        f"当前RPS: {success_count/elapsed:.1f}/s"
    )

# 等待所有线程结束
for t in threads:
    t.join()

# 计算结果
total_requests = success_count + error_count
throughput = success_count / TEST_DURATION * 60  # 转换为每分钟
avg_latency = statistics.mean(latencies) * 1000 if latencies else 0

# 输出报告
print("\n📊 测试结果:")
print(f"总请求数: {total_requests}")
print(
    f"成功请求: {success_count} (成功率: {success_count/total_requests:.1%})"
)
print(f"吞吐量: {throughput:.1f} 请求/分钟")
print(f"平均延迟: {avg_latency:.1f} ms")
print(
    f"P95延迟: {statistics.quantiles(latencies, n=20)[18]*1000:.1f} ms"
)  # 95分位
