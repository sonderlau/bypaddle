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
logger = logging.getLogger(__name__)

# 动态导入 LLMRAG
spec = importlib.util.spec_from_file_location("llm_rag", "11-llm-rag.py")
rag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_module)
LLMRAG = rag_module.LLMRAG

class AsyncLLMRAG:
    """异步RAG系统封装器"""
    
    def __init__(self, sync_rag):
        """
        初始化异步RAG封装器
        
        Args:
            sync_rag: 同步RAG系统实例
        """
        self.sync_rag = sync_rag
        
    async def answer_question(self, query: str, return_context: bool = False) -> Dict[str, Any]:
        """
        异步方式回答问题
        
        Args:
            query: 用户问题
            return_context: 是否返回上下文
            
        Returns:
            包含答案和可选上下文的字典
        """
        try:
            # 使用线程池执行同步操作
            result = await asyncio.to_thread(
                self.sync_rag.answer_question,
                query=query,
                return_context=return_context
            )
            return result
            
        except Exception as e:
            logger.error(f"异步回答问题时出错: {str(e)}")
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