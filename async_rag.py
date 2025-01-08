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

# 在文件顶部的导入部分后添加
logger = logging.getLogger(__name__)

# 使用已经创建的调试日志记录器
debug_logger = logging.getLogger('debug')

# 动态导入 LLMRAG
spec = importlib.util.spec_from_file_location("llm_rag", "11-llm-rag.py")
rag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_module)
LLMRAG = rag_module.LLMRAG

# 添加动态导入:
spec_reranker = importlib.util.spec_from_file_location("rag_reranker", "10-2.rag-reranker.py")
reranker_module = importlib.util.module_from_spec(spec_reranker)
spec_reranker.loader.exec_module(reranker_module)
search_with_rerank = reranker_module.search_with_rerank

class AsyncLLMRAG:
    """异步RAG系统封装器"""
    
    def __init__(self, rag_system):
        """
        初始化异步RAG封装器
        
        Args:
            rag_system: 同步RAG系统实例
        """
        self.rag = rag_system
        # 只针对RAG搜索的信号量
        self.search_semaphore = asyncio.Semaphore(1)
        
    async def answer_question(self, query: str, return_context: bool = False, user_id: str = None) -> Dict[str, Any]:
        """异步回答问题"""
        try:
            # 创建日志适配器
            logger_adapter = logging.LoggerAdapter(
                logger,
                {'user_id': user_id}
            )
            
            start_total = time.time()
            request_id = f"{user_id}-{int(start_total)}"
            
            debug_logger.info(f"[{request_id}] AsyncLLMRAG 开始处理请求")

            # 1. RAG搜索阶段 - 使用信号量
            logger_adapter.info("开始检索相关文档。这一步需要等待约25秒⚠️⚠️⚠️。 "
                              f"搜索参数: 粗召回top_k={self.rag.initial_top_k}, 精排final_top_k={self.rag.final_top_k}")
            
            search_start = time.time()
            debug_logger.info(f"[{request_id}] 等待获取 RAG 搜索信号量")
            
            async with self.search_semaphore:
                debug_logger.info(f"[{request_id}] 获得 RAG 搜索信号量")
                
                # 执行混合搜索和重排序
                search_results = await asyncio.to_thread(
                    search_with_rerank,
                    query=query,
                    hybrid_searcher=self.rag.hybrid_searcher,
                    reranker=self.rag.reranker,
                    initial_top_k=self.rag.initial_top_k,
                    final_top_k=self.rag.final_top_k
                )
                
                search_time = time.time() - search_start
                logger_adapter.info(f"检索完成，耗时 {search_time:.2f}秒。精排序后 {len(search_results)} 条结果")
                for i, result in enumerate(search_results):
                    content = result['chunk']['current_content']
                    logger_adapter.info(f"搜索结果 #{i+1} 预览:\n"
                                    f"开头:\n {content[:100]}...\n"
                                    f"结尾:\n ...{content[-100:]}")
                
                logger_adapter.info(f"检索完成，耗时 {search_time:.2f}秒。精排序后 {len(search_results)} 条结果")
                    
                # 2. 格式化上下文
                logger_adapter.info("开始格式化上下文...")
                format_start = time.time()
                context = await asyncio.to_thread(
                    self.rag._format_context,
                    search_results
                )
                format_time = time.time() - format_start
                logger_adapter.info(f"格式化完成，耗时 {format_time:.2f}秒，上下文长度: {len(context)} 字符")
                
                debug_logger.info(f"[{request_id}] RAG 搜索和上下文准备完成")

            # 3. LLM生成阶段 - 不使用信号量
            logger_adapter.info("开始生成答案...")
            generate_start = time.time()
            answer = await asyncio.to_thread(
                self.rag._generate_answer,
                query=query,
                context=context,
                user_id=user_id
            )
            generate_time = time.time() - generate_start
            
            total_time = time.time() - start_total
            logger_adapter.info(f"答案生成完成，耗时 {generate_time:.2f}秒")
            logger_adapter.info(f"总耗时: {total_time:.2f}秒 (检索: {search_time:.2f}秒, "
                              f"格式化: {format_time:.2f}秒, 生成: {generate_time:.2f}秒)")
            
            debug_logger.info(f"[{request_id}] LLM 生成完成")
            
            result = {"answer": answer}
            if return_context:
                result["context"] = context
            return result
                
        except Exception as e:
            logger_adapter.error(f"生成回答时出错: {str(e)}")
            raise

async def main():
    """测试同步和异步RAG系统的性能对比"""
    from time import time
    import os
    
    # 配置
    DATA_PATH = "output/data_with_abstracts.json"
    API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    
    # 初始化同步RAG系统
    sync_rag = LLMRAG(
        data_path=DATA_PATH,
        api_key=API_KEY,
        initial_top_k=10,
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