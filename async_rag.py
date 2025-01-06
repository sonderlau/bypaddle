from typing import Dict, Any, Optional
import asyncio
import logging
from openai import OpenAI
# 12-llm-workflow.py
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime
from openai import OpenAI
import importlib.util
from event_bus import EventBus  # Import EventBus
import asyncio
from datetime import datetime  # 修改这里
import time

# 使用已经创建的调试日志记录器
debug_logger = logging.getLogger('debug')

# 动态导入 LLMRAG
spec = importlib.util.spec_from_file_location("llm_rag", "11-llm-rag.py")
rag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_module)
LLMRAG = rag_module.LLMRAG

class AsyncLLMRAG:
    """异步RAG系统封装器"""
    
    def __init__(self, rag_system):
        """
        初始化异步RAG封装器
        
        Args:
            rag_system: 同步RAG系统实例
        """
        self.rag = rag_system
        # 添加信号量来控制并发
        self.semaphore = asyncio.Semaphore(1)  # 限制同时只能有一个搜索请求
        
    async def answer_question(self, query: str, return_context: bool = False, user_id: str = None) -> Dict[str, Any]:
        """异步回答问题"""
        start_time = time.time()
        request_id = f"{user_id}-{int(start_time)}"  # 创建唯一的请求ID
        
        try:
            debug_logger.info(f"[{request_id}] AsyncLLMRAG 开始处理请求")
            debug_logger.info(f"[{request_id}] 当前信号量状态: {self.semaphore._value}")
            
            # 使用信号量控制并发
            debug_logger.info(f"[{request_id}] 等待获取 AsyncLLMRAG 信号量")
            async with self.semaphore:
                debug_logger.info(f"[{request_id}] 获得 AsyncLLMRAG 信号量")
                
                # 记录开始执行同步操作
                to_thread_start = time.time()
                debug_logger.info(f"[{request_id}] 开始执行同步 RAG 操作")
                
                try:
                    # 使用同步 RAG 系统回答问题，但在异步上下文中执行
                    result = await asyncio.to_thread(
                        self.rag.answer_question,
                        query=query,
                        return_context=return_context,
                        user_id=user_id
                    )
                    
                    to_thread_time = time.time() - to_thread_start
                    debug_logger.info(f"[{request_id}] 同步 RAG 操作完成，耗时: {to_thread_time:.2f}秒")
                    return result
                    
                except Exception as e:
                    debug_logger.error(f"[{request_id}] 同步 RAG 操作出错: {str(e)}")
                    raise
                finally:
                    debug_logger.info(f"[{request_id}] 释放 AsyncLLMRAG 信号量")
                
        except Exception as e:
            debug_logger.error(f"[{request_id}] AsyncLLMRAG 处理请求时出错: {str(e)}")
            raise
        finally:
            total_time = time.time() - start_time
            debug_logger.info(f"[{request_id}] AsyncLLMRAG 请求处理完成，总耗时: {total_time:.2f}秒")
            debug_logger.info(f"[{request_id}] 最终信号量状态: {self.semaphore._value}")

async def main():
    """测试同步和异步RAG系统的性能对比"""
    from time import time
    import os
    
    # 配置
    DATA_PATH = "output/data_with_abstracts.json"
    API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-c3b22834c96a4f368657ad8eafa1999f")
    
    # 初始化同步RAG系统
    sync_rag = LLMRAG(
        data_path=DATA_PATH,
        api_key=API_KEY,
        initial_top_k=15,
        final_top_k=5
    )
    
    # 初始化异步封装器
    async_rag = AsyncLLMRAG(sync_rag)
    
    # 测试问题
    test_questions = [
        "学生申请休学的流程是什么？",
        "国家奖学金的评选条件有哪些？",
        "学生考试作弊会受到什么处分？"
    ]
    
    # 同步测试
    print("\n=== 开始同步测试 ===")
    sync_start = time()
    for question in test_questions:
        print(f"\n问题: {question}")
        result = sync_rag.answer_question(question)
        print(f"答案: {result['answer']}")
    sync_duration = time() - sync_start
    print(f"\n同步测试总耗时: {sync_duration:.2f}秒")
    
    # 异步测试
    print("\n=== 开始异步测试 ===")
    async_start = time()
    tasks = [async_rag.answer_question(q) for q in test_questions]
    results = await asyncio.gather(*tasks)
    
    for q, r in zip(test_questions, results):
        print(f"\n问题: {q}")
        print(f"答案: {r['answer']}")
    async_duration = time() - async_start
    print(f"\n异步测试总耗时: {async_duration:.2f}秒")
    
    # 性能对比
    speedup = (sync_duration - async_duration) / sync_duration * 100
    print(f"\n=== 性能对比 ===")
    print(f"同步执行时间: {sync_duration:.2f}秒")
    print(f"异步执行时间: {async_duration:.2f}秒")
    print(f"性能提升: {speedup:.1f}%")

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    asyncio.run(main()) 