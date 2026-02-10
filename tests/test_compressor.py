"""
测试压缩器功能
"""
import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aworld.config.conf import AgentConfig
from agents.compressor_agent import CompressorAgent
from memory.memory_manager import MemoryManager
from memory.memory_item import RawContextItem, AgentType
from datetime import datetime
from dotenv import load_dotenv


# 加载环境变量
load_dotenv()


async def simple_compress(text: str, 
                         threshold: int = 500,
                         api_key: str = None,
                         model_name: str = None,
                         base_url: str = None) -> str:
    """
    简化的压缩接口
    
    Args:
        text: 要压缩的文本
        threshold: 压缩阈值（字符数）
        api_key: API密钥（默认使用main_aiopslab.py中的配置）
        model_name: 模型名称（默认使用main_aiopslab.py中的配置）
        base_url: API基础URL（默认使用main_aiopslab.py中的配置）
        
    Returns:
        压缩后的文本（如果低于阈值则直接返回原文）
        
    Raises:
        Exception: 压缩失败时抛出异常
    """
    # 1. 如果文本长度低于阈值，直接输出
    if len(text) < threshold:
        print(f"✅ Text length ({len(text)}) < threshold ({threshold}), returning original text")
        return text
    
    # 2. 超过阈值，使用LLM压缩
    print(f"🗜️  Text length ({len(text)}) >= threshold ({threshold}), compressing...")
    
    try:
        # 从main_aiopslab导入配置（参考main_aiopslab.py的用法）
        from main_aiopslab import DEV_API_KEY, DEV_API_BASE, DEV_MODEL, DEV_API_SOURCE
        
        # 使用main_aiopslab.py中的配置
        if api_key is None:
            api_key = DEV_API_KEY
        if model_name is None:
            model_name = DEV_MODEL
        if base_url is None:
            # 根据API源判断是否使用base_url（与main_aiopslab.py保持一致）
            base_url = DEV_API_BASE if DEV_API_SOURCE == "openrouter" else None
        
        # 创建 AgentConfig（直接传参数，与main_aiopslab.py保持一致）
        llm_config_params = {
            "llm_provider": "openai",  # OpenRouter也兼容OpenAI API格式
            "llm_model_name": model_name,
            "llm_api_key": api_key,
            "llm_temperature": 0.1
        }
        
        if base_url:
            llm_config_params["llm_base_url"] = base_url
        
        agent_config = AgentConfig(**llm_config_params)
        
        # 创建内存管理器
        memory_manager = MemoryManager()
        
        # 创建压缩器
        compressor = CompressorAgent(
            llm_config=agent_config,
            memory_manager=memory_manager,
            min_compress_length=threshold
        )
        
        # 创建一个临时的RawContextItem用于压缩
        raw_item = RawContextItem(
            source_agent=AgentType.PROBE,
            round_number=1,
            command="test_command",
            raw_output=text,
            success=True,
            metadata={}
        )
        
        # 执行压缩
        compressed_text = await compressor._intelligent_compress_single(
            output_text=text,
            item=raw_item,
            target_tokens=2000
        )
        
        print(f"✅ Compression successful: {len(text)} → {len(compressed_text)} chars")
        print(f"   Compression ratio: {(1 - len(compressed_text)/len(text))*100:.1f}%")
        
        return compressed_text
        
    except Exception as e:
        # 3. 出错直接报错
        print(f"❌ Compression failed: {str(e)}")
        raise


async def test_compression_with_sample_text():
    """测试用户提供的示例文本"""
    
    sample_text = """**Command**: exec_shell("kubectl describe services -n test-social-network")
**Result**:
Name:                     compose-post-service
Namespace:                test-social-network
Labels:                   app.kubernetes.io/managed-by=Helm
Annotations:              meta.helm.sh/release-name: social-network
                          meta.helm.sh/release-namespace: test-social-network
Selector:                 service=compose-post-service
Type:                     ClusterIP
IP Family Policy:         SingleStack
IP Families:              IPv4
IP:                       10.96.99.98
Port:                     9090  9090/TCP
TargetPort:               9090/TCP
Endpoints:                10.244.1.161:9090
Session Affinity:         None
Internal Traffic Policy:  Cluster
Events:                   <none>


Name:                     home-timeline-redis
Selector:                 service=home-timeline-redis
IP:                       10.96.227.172
Port:                     6379  6379/TCP
TargetPort:               6379/TCP
Endpoints:                10.244.1.148:6379


Name:                     home-timeline-service
Selector:                 service=home-timeline-service
IP:                       10.96.45.32


Name:                     jaeger
Selector:                 service=jaeger
IPs:                      10.96.91.58
Port:                     5775  5775/UDP
TargetPort:               5775/UDP
Endpoints:                10.244.1.152:5775
Port:                     6831  6831/UDP
TargetPort:               6831/UDP
Endpoints:                10.244.1.152:6831
Port:                     5778  5778/TCP
TargetPort:               5778/TCP
Port:                     16686  16686/TCP
TargetPort:               16686/TCP
Port:                     14268  14268/TCP
Port:                     9411  9411/TCP
TargetPort:               9411/TCP


Name:                     media-frontend
Selector:                 service=media-frontend
IPs:                      10.96.15.23
Port:                     8081  8081/TCP


Name:                     media-memcached
Selector:                 service=media-memcached
IP:                       10.96.211.177
Port:                     11211  11211/TCP
TargetPort:               11211/TCP
Endpoints:                10.244.1.164:11211


Name:                     media-mongodb
Selector:                 service=media-mongodb
IP:                       10.96.142.175
Port:                     27017  27017/TCP
TargetPort:               27017/TCP
Endpoints:                10.244.1.167:27017


Name:                     media-service
Selector:                 service=media-service
IP:                       10.96.147.164


Name:                     nginx-thrift
Selector:                 service=nginx-thrift


Name:                     post-storage-memcached
Selector:                 service=post-storage-memcached
IP:                       10.96.56.116


Name:                     post-storage-mongodb
Selector:                 service=post-storage-mongodb
IPs:                      10.96.224.124


Name:                     post-storage-service
Selector:                 service=post-storage-service
IPs:                      10.96.112.137


Name:                     social-graph-mongodb
Selector:                 service=social-graph-mongodb


Name:                     social-graph-redis
Selector:                 service=social-graph-redis


Name:                     social-graph-service
Selector:                 service=social-graph-service


Name:                     text-service
Selector:                 service=text-service
IP:                       10.96.131.191


Name:                     unique-id-service
Selector:                 service=unique-id-service
IPs:                      10.96.5.190


Name:                     url-shorten-memcached
Selector:                 service=url-shorten-memcached
IP:                       10.96.77.222


Name:                     url-shorten-mongodb
Selector:                 service=url-shorten-mongodb
IPs:                      10.96.40.25


Name:                     url-shorten-service
Selector:                 service=url-shorten-service
IP:                       10.96.30.240


Name:                     user-memcached
Selector:                 service=user-memcached


Name:                     user-mention-service
Selector:                 service=user-mention-service
IP:                       10.96.201.232


Name:                     user-mongodb
Selector:                 service=user-mongodb
IP:                       10.96.98.182


Name:                     user-service
Endpoints:                10.244.1.157:9999


Name:                     user-timeline-mongodb
Selector:                 service=user-timeline-mongodb
IP:                       10.96.238.61
"""
    
    print("=" * 80)
    print("测试压缩器功能")
    print("=" * 80)
    print(f"\n📝 原始文本长度: {len(sample_text)} 字符")
    print(f"📝 原始文本前200字符:\n{sample_text[:200]}...\n")
    
    # 测试1: 使用默认阈值500
    print("\n" + "=" * 80)
    print("测试 1: 使用默认阈值 (500 字符)")
    print("=" * 80)
    try:
        compressed = await simple_compress(sample_text, threshold=500)
        print(f"\n✅ 压缩成功!")
        print(f"📊 压缩后长度: {len(compressed)} 字符")
        print(f"📊 压缩比: {(1 - len(compressed)/len(sample_text))*100:.1f}%")
        print(f"\n📄 压缩后文本:\n{compressed}\n")
    except Exception as e:
        print(f"\n❌ 压缩失败: {str(e)}")
    
    # 测试2: 使用高阈值，文本不应被压缩
    print("\n" + "=" * 80)
    print("测试 2: 使用高阈值 (10000 字符) - 应直接返回原文")
    print("=" * 80)
    try:
        result = await simple_compress(sample_text, threshold=10000)
        if result == sample_text:
            print("✅ 测试通过: 文本未被压缩，直接返回原文")
        else:
            print("❌ 测试失败: 文本被压缩了（不应该）")
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
    
    # 测试3: 测试短文本
    print("\n" + "=" * 80)
    print("测试 3: 短文本 (低于阈值)")
    print("=" * 80)
    short_text = "This is a short text."
    try:
        result = await simple_compress(short_text, threshold=500)
        if result == short_text:
            print("✅ 测试通过: 短文本直接返回，未压缩")
        else:
            print("❌ 测试失败: 短文本不应被压缩")
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")


async def test_error_handling():
    """测试错误处理"""
    print("\n" + "=" * 80)
    print("测试 4: 错误处理（使用无效的API密钥）")
    print("=" * 80)
    
    try:
        # 使用无效的API密钥触发错误
        await simple_compress(
            "This is a test text that is long enough to trigger compression. " * 20,
            threshold=500,
            api_key="invalid_key"
        )
        print("❌ 测试失败: 应该抛出异常但没有")
    except Exception as e:
        print(f"✅ 测试通过: 成功捕获异常 - {type(e).__name__}")


if __name__ == "__main__":
    # 运行所有测试
    asyncio.run(test_compression_with_sample_text())
    asyncio.run(test_error_handling())
    
    print("\n" + "=" * 80)
    print("✅ 所有测试完成!")
    print("=" * 80)

