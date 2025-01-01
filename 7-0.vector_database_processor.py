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
            self.collection = self.chroma_client.get_or_create_collection("document_chunks")
            logger.info(f"ChromaDB初始化完成，使用目录: {persist_directory}")
            
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            raise
            
    def prepare_chunks(self, json_file: str) -> List[Dict]:
        """准备文档块，确保所有必要的元数据都被正确保存"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取章节摘要信息
            section_abstracts = data.get('section_abstracts', {})
            chunks = []
            
            # 处理每个页面
            for page in data.get('pages', []):
                section = page.get('section', '')
                section_abstract = section_abstracts.get(section, '')  # 从section_abstracts中获取对应章节的摘要
                
                chunk = {
                    'section': section,
                    'section_abstract': section_abstract,
                    'page_number': page.get('page_number'),
                    'current_content': page.get('content', ''),
                    'metadata': {
                        'section': section,
                        'section_abstract': section_abstract,
                        'page_number': page.get('page_number')
                    }
                }
                chunks.append(chunk)
            
            logger.info(f"成功准备 {len(chunks)} 个文档块")
            return chunks
            
        except Exception as e:
            logger.error(f"准备文档块时出错: {str(e)}")
            raise
            
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
            
    def create_embeddings(self, chunks: List[Dict]):
        """创建并存储嵌入向量"""
        try:
            # 获取现有的所有文档ID
            existing_ids = self.collection.get()['ids']
            if existing_ids:
                # 如果有现有文档，则删除
                self.collection.delete(ids=existing_ids)
            
            texts = [chunk['current_content'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            # 批量添加文档
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=[str(i) for i in range(len(texts))]
            )
            logger.info(f"成功添加 {len(texts)} 个文档到向量数据库")
        except Exception as e:
            logger.error(f"创建嵌入向量时出错: {str(e)}")
            raise
            
    def search_similar(self, query: str, top_k: int = 3) -> List[Dict]:
        """搜索相似内容，确保返回完整的元数据"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            similar_chunks = []
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i]
                similar_chunks.append({
                    'chunk': {
                        'section': metadata['section'],
                        'section_abstract': metadata['section_abstract'],  # 从metadata中获取
                        'page_number': metadata['page_number'],
                        'current_content': results['documents'][0][i]
                    },
                    'similarity_score': float(results['distances'][0][i])
                })
            
            return similar_chunks
        except Exception as e:
            logger.error(f"搜索相似内容时出错: {str(e)}")
            raise

def main():
    try:
        # 初始化处理器
        processor = VectorProcessor(persist_directory="chroma_db")
        
        # 调试模式：只处理前20条数据
        DEBUG_LIMIT = -1
        
        # 1. 准备文档块
        logger.info("开始准备文档块...")
        all_chunks = processor.prepare_chunks("output/data_with_abstracts.json")
        chunks = all_chunks[:DEBUG_LIMIT]
        logger.info(f"调试模式：使用前 {DEBUG_LIMIT} 条数据（共 {len(all_chunks)} 条）")
        
        # 2. 创建嵌入向量并存储
        logger.info("开始创建嵌入向量...")
        processor.create_embeddings(chunks)  # 移除 batch_size 参数
        logger.info("完成向量化和存储")
        
        # 3. 测试搜索
        logger.info("执行测试搜索...")
        results = processor.search_similar("省政府奖学金", top_k=3)
        
        print("\n搜索结果:")
        for i, result in enumerate(results, 1):
            print(f"\n结果 {i}:")
            print(f"相似度: {result['similarity_score']:.3f}")
            print(f"章节: {result['chunk']['section']}")
            print(f"页码: {result['chunk']['page_number']}")
            print(f"内容片段: {result['chunk']['current_content'][:200]}...")
            
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()