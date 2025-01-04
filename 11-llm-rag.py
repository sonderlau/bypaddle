# 11-llm-rag.py
from typing import List, Dict, Any
import logging
import json
from openai import OpenAI
from pathlib import Path
import os
import importlib.util
import httpx
import time
from datetime import datetime  # 修改这里

# 动态导入向量处理器
spec = importlib.util.spec_from_file_location("vector_processor", "7-1.vector-with-abstract.py")
vector_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vector_module)
VectorProcessor = vector_module.VectorProcessor

# 动态导入RAG重排序组件
spec2 = importlib.util.spec_from_file_location("rag_reranker", "10-2.rag-reranker.py")
rag_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(rag_module)
HybridSearcher = rag_module.HybridSearcher
Reranker = rag_module.Reranker
search_with_rerank = rag_module.search_with_rerank

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMRAG:
    def __init__(
        self, 
        data_path: str,
        api_key: str,
        model_name: str = "qwen-long",
        initial_top_k: int = 10,
        final_top_k: int = 3
    ):
        """初始化LLM-RAG系统
        
        Args:
            data_path: 包含文档内容的JSON文件路径
            api_key: DashScope API密钥
            model_name: 使用的模型名称
            initial_top_k: 混合检索的初始召回数量
            final_top_k: 重排序后保留的文档数量
        """
        self.model_name = model_name
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k
        
        # 初始化OpenAI客户端
        http_client = httpx.Client()
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            http_client=http_client
        )
        
        # 初始化向量处理器
        self.vector_processor = VectorProcessor()
        
        # 加载数据
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在：{data_path}")
        self.data_path = data_path
        
        # 初始化检索组件
        self._initialize_search_components()
        
    def _initialize_search_components(self):
        """初始化检索相关的组件"""
        logger.info("初始化检索组件...")
        
        # 准备文档块
        chunks = self.vector_processor.prepare_chunks(self.data_path)
        
        # 初始化混合搜索器
        self.hybrid_searcher = HybridSearcher(self.vector_processor)
        self.hybrid_searcher.prepare_bm25(chunks)
        
        # 初始化重排序器
        self.reranker = Reranker()
        
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """将检索结果格式化为上下文"""
        context_parts = []
        
        for result in results:
            chunk = result['chunk']
            # 添加章节信息和两种页码
            section_info = f"【{chunk['section']}】"
            page_info = f"(PDF页码第{chunk['page_number']}页"
            if chunk['document_page'] != chunk['page_number']:
                page_info += f", 文档第{chunk['document_page']}页)"
            else:
                page_info += ")"
            
            section_info += page_info
            
            if chunk['section_abstract']:
                section_info += f"\n章节概述：{chunk['section_abstract']}"
            
            # 添加具体内容
            content = f"\n具体内容：{chunk['current_content']}"
            
            # 合并当前文档块的信息
            context_parts.append(f"{section_info}{content}")
            
        return "\n\n".join(context_parts)
        
    def _generate_answer(self, query: str, context: str) -> str:
        """使用LLM生成答案
        
        Args:
            query: 用户问题
            context: 检索到的相关上下文
            
        Returns:
            生成的答案
        """
        prompt = f"""请基于以下参考信息回答用户的问题。要求：
1. 答案必须准确，与参考信息保持一致
2. 如果参考信息不足以完整回答问题，请明确指出
3. 合理组织答案结构，适当分点说明
4. 可以直接引用原文内容，注意语言流畅
5. 如果有页码，请在答案中说明可以查阅手册的页码


参考信息：
==========
{context}
==========

用户问题：{query}


请生成解答："""

        logger.info(f"生成答案，这一步比较耗时⚠️。使用模型：{self.model_name}")
        response = self.client.chat.completions.create(
            model=self.model_name,  
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.2
        )
        
        return response.choices[0].message.content.strip()
        
    def answer_question(self, query: str, return_context: bool = False) -> Dict[str, Any]:
        """回答用户问题"""
        try:
            import time
            start_total = time.time()
            
            # 1. 检索相关文档
            logger.info(f"开始检索相关文档。这一步比较耗时⚠️。 搜索参数: 粗召回top_k={self.initial_top_k}, 精排final_top_k={self.final_top_k}")
            search_start = time.time()
            search_results = search_with_rerank(
                query=query,
                hybrid_searcher=self.hybrid_searcher,
                reranker=self.reranker,
                initial_top_k=self.initial_top_k,
                final_top_k=self.final_top_k
            )
            search_time = time.time() - search_start
            
            logger.info(f"检索完成，耗时 {search_time:.2f}秒。精排序后 {len(search_results)} 条结果")

            # 2. 格式化上下文
            logger.info("开始格式化上下文...")
            format_start = time.time()
            context = self._format_context(search_results)
            format_time = time.time() - format_start
            logger.info(f"格式化完成，耗时 {format_time:.2f}秒，上下文长度: {len(context)} 字符")
            logger.info(f"格式化后的上下文预览:\n"
                       f"开头部分:\n{context[:200]}...\n"
                       f"结尾部分:\n...{context[-200:]}")
            
            # 3. 生成答案
            logger.info("开始生成答案...")
            generate_start = time.time()
            answer = self._generate_answer(query, context)
            
            # 如果答案中没有包含页码，从检索结果中提取页码并添加
            if not any(f"第{i}页" in answer for i in range(1000)):
                # 收集两种页码
                logical_pages = sorted(set(result['chunk']['page_number'] for result in search_results))
                doc_pages = sorted(set(result['chunk']['document_page'] for result in search_results))
                
                # 构建页码信息
                if set(doc_pages) == set(logical_pages):
                    # 如果两种页码相同，只显示一种
                    page_info = f"\n\n(参考自第{', '.join(map(str, logical_pages))}页)"
                else:
                    # 如果不同，同时显示两种页码，PDF
                    # 页码在前
                    page_info = f"\n\n(参考自PDF页码第{', '.join(map(str, logical_pages))}页"
                    page_info += f"，对应文档第{', '.join(map(str, doc_pages))}页)"
                
                answer += page_info
                
            generate_time = time.time() - generate_start
            
            total_time = time.time() - start_total
            logger.info(f"答案生成完成，耗时 {generate_time:.2f}秒")
            logger.info(f"总耗时: {total_time:.2f}秒 (检索: {search_time:.2f}秒, 格式化: {format_time:.2f}秒, 生成: {generate_time:.2f}秒)")
            
            result = {"answer": answer}
            if return_context:
                result["context"] = context
            return result
            
        except Exception as e:
            logger.error(f"生成回答时出错: {str(e)}")
            raise

    def search_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """搜索相似内容"""
        try:
            query_embedding = self._create_embedding(query)
            collection = self.chroma_client.get_collection("document_embeddings")
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            similar_chunks = []
            for i in range(len(results['documents'][0])):
                similar_chunks.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': float(results['distances'][0][i])
                })
            
            return similar_chunks
            
        except Exception as e:
            logger.error(f"搜索过程中出错: {str(e)}")
            raise

    def generate_answer(self, query: str) -> str:
        """生成回答"""
        try:
            # 搜索相关内容
            similar_chunks = self.search_similar(query, top_k=3)
            
            # 构建提示
            context_texts = []
            page_references = []  # 新增：用于存储页码引用
            
            for chunk in similar_chunks:
                content = chunk['content']
                page_number = chunk['metadata']['page_number']
                context_texts.append(content)
                page_references.append(f"第{page_number}页")  # 新增：收集页码信息
            
            context = "\n\n".join(context_texts)
            pages_info = "、".join(page_references)  # 新增：组合页码信息
            
            prompt = f"""请基于以下内容回答问题。如果无法从内容中找到答案，请明确说明。

内容：
{context}

问题：{query}

请提供准确、简洁的回答。在回答末尾注明参考页码。"""

            # 调用 LLM 生成回答
            response = self.llm(prompt)
            
            # 在回答末尾添加页码引用
            final_answer = f"{response}\n\n参考来源：{pages_info}"
            
            return final_answer
            
        except Exception as e:
            logger.error(f"生成回答时出错: {str(e)}")
            return f"抱歉，生成回答时出现错误: {str(e)}"

def main():
    # 配置
    DATA_PATH = "output/data_with_abstracts.json"
    API_KEY="sk-c3b22834c96a4f368657ad8eafa1999f"
  # 替换为实际的API密钥
    
    # 初始化RAG系统
    rag = LLMRAG(
        data_path=DATA_PATH,
        api_key=API_KEY
    )
    
    # 测试问题
    test_queries = [
        "学籍异动包括哪些情况？",
        "学生申请休学的流程是什么？",
    ]
    
    # 测试回答
    for query in test_queries:
        print(f"\n问题：{query}")
        try:
            result = rag.answer_question(query, return_context=True)
            print("\n答案：")
            print(result["answer"])
            print("\n参考上下文：")
            print(result["context"])
        except Exception as e:
            logger.error(f"处理问题时出错：{str(e)}")
            
if __name__ == "__main__":
    main()