import json
from typing import List, Dict, Any
import logging
import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from chromadb import Client
import chromadb
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorProcessor:
    def __init__(self, model_name: str = 'BAAI/bge-large-zh-v1.5', persist_directory: str = "./chroma_db"):
        """初始化向量处理器
        
        Args:
            model_name: HuggingFace模型名称
            persist_directory: ChromaDB持久化目录
        """
        logger.info(f"初始化向量处理器，使用模型: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            
            # 设备选择逻辑
            if torch.backends.mps.is_available():
                self.device = 'mps'
                logger.info("使用 MPS (Metal Performance Shaders) 加速")
            elif torch.cuda.is_available():
                self.device = 'cuda'
                logger.info("使用 CUDA 加速")
            else:
                self.device = 'cpu'
                logger.info("使用 CPU 处理")
                
            self.model = self.model.to(self.device)
            
            # 初始化ChromaDB，使用持久化存储
            os.makedirs(persist_directory, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
            logger.info(f"ChromaDB初始化完成，使用目录: {persist_directory}")
            
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            raise
            
    def prepare_chunks(self, data_path: str, window_size: int = 200) -> List[Dict[str, Any]]:
        """准备文档块，包含上下文窗口
        
        Args:
            data_path: 带章节信息的JSON文件路径
            window_size: 上下文窗口大小
            
        Returns:
            处理后的文档块列表
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        chunks = []
        pages = data['pages']
        
        for i, page in enumerate(pages):
            # 获取前文上下文
            prev_content = ""
            if i > 0:
                prev_content = pages[i-1]['content'][-window_size:] if pages[i-1]['content'] else ""
                
            # 获取后文上下文    
            next_content = ""
            if i < len(pages) - 1:
                next_content = pages[i+1]['content'][:window_size] if pages[i+1]['content'] else ""
            
            # 构建完整的上下文
            context = {
                'prev_content': prev_content,
                'current_content': page['content'],
                'next_content': next_content,
                'section': page['section'],
                'page_number': page['page_number'],
                'document_page': page.get('document_page'),
                'metadata': {
                    'section': page['section'],
                    'page_number': page['page_number'],
                    'document_page': page.get('document_page')
                }
            }
            
            # 处理页面内容，包含表格及其上下文
            if page.get('tables'):
                combined_content = context['current_content']
                for idx, table in enumerate(page['tables'], 1):
                    if table.get('llm_analysis'):
                        combined_content += f"\n【表格{idx}】\n{table['llm_analysis']}\n"
                
                context['metadata']['has_tables'] = True
                context['metadata']['table_count'] = len(page['tables'])
                # 将列表转换为字符串
                context['metadata']['table_indices'] = ','.join(str(i) for i in range(1, len(page['tables']) + 1))
                context['current_content'] = combined_content
            
            chunks.append(context)
            
        return chunks
        
    def _create_embedding(self, text: str) -> List[float]:
        """使用HuggingFace模型创建文本嵌入向量"""
        try:
            # 对超长文本进行截断
            max_length = 512  # 根据模型调整
            encoded_input = self.tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                sentence_embedding = model_output[0][:, 0]
                sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)
                
            return sentence_embedding.cpu().numpy()[0].tolist()
            
        except Exception as e:
            logger.error(f"创建嵌入向量时出错: {str(e)}")
            raise
            
    def create_embeddings(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> None:
        """为所有文档块创建嵌入向量并存入ChromaDB
        
        Args:
            chunks: 文档块列表
            batch_size: 批处理大小
        """
        try:
            collection = self.chroma_client.get_or_create_collection(
                name="document_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            
            total_chunks = len(chunks)
            logger.info(f"开始处理 {total_chunks} 个文档块")
            
            # 批量处理
            for i in tqdm(range(0, total_chunks, batch_size), desc="处理文档块"):
                batch_chunks = chunks[i:i + batch_size]
                batch_texts = []
                batch_metadatas = []
                batch_ids = []
                
                for chunk in batch_chunks:
                    text_to_embed = f"""
                    章节：{chunk['section']}
                    页码：{chunk['page_number']}
                    位置信息：这是第{chunk['page_number']}页，属于{chunk['section']}章节。
                    上文：{chunk['prev_content']}
                    正文内容：{chunk['current_content']}
                    下文：{chunk['next_content']}
                    """.strip()
                    
                    batch_texts.append(text_to_embed)
                    batch_metadatas.append(chunk['metadata'])
                    batch_ids.append(f"chunk_{chunk['page_number']}_{hash(chunk['current_content'])}")
                
                # 批量创建嵌入向量
                batch_embeddings = [self._create_embedding(text) for text in batch_texts]
                
                # 批量存储
                collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                if (i + batch_size) % 100 == 0:
                    logger.info(f"已处理 {min(i + batch_size, total_chunks)}/{total_chunks} 个文档块")
            
            logger.info("所有文档块处理完成")
            
        except Exception as e:
            logger.error(f"创建嵌入向量过程中出错: {str(e)}")
            raise
            
    def search_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """搜索相似内容
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            相似度最高的文档块列表
        """
        try:
            # 获取查询向量
            query_embedding = self._create_embedding(query)
            
            # 获取集合
            collection = self.chroma_client.get_collection("document_embeddings")
            
            # 使用ChromaDB的查询功能
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # 组织返回结果
            similar_chunks = []
            for i in range(len(results['documents'][0])):
                similar_chunks.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': float(results['distances'][0][i])  # Convert to float for JSON serialization
                })
            
            return similar_chunks
            
        except Exception as e:
            logger.error(f"搜索过程中出错: {str(e)}")
            raise

def main():
    try:
        # 初始化处理器
        processor = VectorProcessor(persist_directory="chroma_db")
        
        # 调试模式：只处理前20条数据
        DEBUG_LIMIT = -1
        
        # 1. 准备文档块
        logger.info("开始准备文档块...")
        all_chunks = processor.prepare_chunks("output/data_with_sections.json")
        chunks = all_chunks[:DEBUG_LIMIT]  # 只取前20条
        logger.info(f"调试模式：使用前 {DEBUG_LIMIT} 条数据（共 {len(all_chunks)} 条）")
        
        # 2. 创建嵌入向量并存储
        logger.info("开始创建嵌入向量...")
        processor.create_embeddings(chunks, batch_size=5)  # 减小batch size便于调试
        logger.info("完成向量化和存储")
        
        # 3. 测试搜索
        logger.info("执行测试搜索...")
        results = processor.search_similar("省政府奖学金", top_k=3)
        
        print("\n搜索结果:")
        for i, result in enumerate(results, 1):
            print(f"\n结果 {i}:")
            print(f"相似度: {result['similarity_score']:.3f}")
            print(f"章节: {result['metadata']['section']}")
            print(f"页码: {result['metadata']['page_number']}")
            print(f"内容片段: {result['content'][:200]}...")
            
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()