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
logger = logging.getLogger(__name__)

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
        # 只保留 RAG 搜索的信号量
        self.semaphore = asyncio.Semaphore(1)  # 限制同时只能有一个搜索请求
        
    async def answer_question(self, query: str, return_context: bool = False, user_id: str = None) -> Dict[str, Any]:
        """异步回答问题"""
        start_time = time.time()
        request_id = f"{user_id}-{int(start_time)}"
        logger_adapter = logging.LoggerAdapter(
                logger,
                {'user_id': user_id}
            )
            
        try:
            debug_logger.info(f"[{request_id}] AsyncLLMRAG 开始处理请求")

            # 使用信号量控制 RAG 搜索
            debug_logger.info(f"[{request_id}] 等待获取 RAG 搜索信号量")
            async with self.semaphore:
                debug_logger.info(f"[{request_id}] 获得 RAG 搜索信号量")
                logger_adapter.info("获得 RAG 搜索信号量")

                result = await asyncio.to_thread(
                    self.rag.answer_question,
                    query=query,
                    return_context=return_context,
                    user_id=user_id
                )
                debug_logger.info(f"[{request_id}] RAG 搜索完成")
                
            return result
                
        except Exception as e:
            debug_logger.error(f"[{request_id}] AsyncLLMRAG 处理请求时出错: {str(e)}")
            raise

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