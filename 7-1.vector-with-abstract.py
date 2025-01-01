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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorProcessor:
    def __init__(self, model_name: str = 'BAAI/bge-large-zh-v1.5', persist_directory: str = "./chroma_db"):
        """初始化向量处理器"""
        logger.info(f"初始化向量处理器，使用模型: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            def search_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
                try:
                    # ... 现有代码 ...
                    similar_chunks = []
                    for i in range(len(results['documents'][0])):
                        similar_chunks.append({
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'section_abstract': section_abstracts.get(results['metadatas'][0][i]['section'], ""),  # 添加这行
                            'similarity_score': float(results['distances'][0][i])
                        })
                except Exception as e:
                    logger.error(f"搜索相似内容时出错: {str(e)}")
                    raise
            # 设备选择逻辑
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
            logger.info(f"使用设备: {self.device}")
                
            self.model = self.model.to(self.device)
            
            # 初始化ChromaDB
            os.makedirs(persist_directory, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
            
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            raise
            
    def prepare_chunks(self, data_path: str, window_size: int = 200) -> List[Dict[str, Any]]:
        """准备文档块，包含章节摘要和上下文窗口"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 获取章节摘要
        section_abstracts = data.get('section_abstracts', {})
        chunks = []
        pages = data['pages']
        
        for i, page in enumerate(pages):
            # 获取上下文
            prev_content = pages[i-1]['content'][-window_size:] if i > 0 and pages[i-1]['content'] else ""
            next_content = pages[i+1]['content'][:window_size] if i < len(pages) - 1 and pages[i+1]['content'] else ""
            
            section = page.get('section', '未知章节')
            section_abstract = section_abstracts.get(section, "")
            
            # 构建完整的上下文
            context = {
                'prev_content': prev_content,
                'current_content': page['content'],
                'next_content': next_content,
                'section': section,
                'section_abstract': section_abstract,
                'page_number': page.get('page_number', 0),  # 使用默认值
                'document_page': page.get('document_page', ""),  # 使用默认值
                'metadata': {
                    'section': section,
                    'page_number': page.get('page_number', 0),  # 使用默认值
                    'document_page': page.get('document_page', ""),  # 使用默认值
                    'has_abstract': bool(section_abstract)
                }
            }
            
            # 处理表格内容
            if page.get('tables'):
                combined_content = context['current_content']
                for idx, table in enumerate(page['tables'], 1):
                    if table.get('llm_analysis'):
                        combined_content += f"\n【表格{idx}】\n{table['llm_analysis']}\n"
                
                context['metadata']['has_tables'] = True
                context['metadata']['table_count'] = len(page['tables'])
                context['metadata']['table_indices'] = ','.join(str(i) for i in range(1, len(page['tables']) + 1))
                context['current_content'] = combined_content
            else:
                context['metadata']['has_tables'] = False
                context['metadata']['table_count'] = 0
                context['metadata']['table_indices'] = ""
            
            chunks.append(context)
            
        return chunks
        
    def _create_embedding(self, text: str) -> List[float]:
        """使用模型创建文本嵌入向量"""
        try:
            encoded_input = self.tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=512,
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

    def _format_text_for_embedding(self, chunk: Dict[str, Any]) -> str:
        """格式化文本用于生成embedding
        
        使用特殊标记来分隔和强调不同的文本部分，帮助模型更好地理解文本结构
        """
        # 构建结构化文本
        formatted_text = []
        
        # 1. 章节信息（最重要的元信息）
        formatted_text.append(f"<章节标题>{chunk['section']}</章节标题>")
        
        # 2. 章节摘要（如果存在）
        if chunk['section_abstract']:
            formatted_text.append(f"<章节摘要>\n{chunk['section_abstract']}\n</章节摘要>")
        
        # 3. 位置信息
        formatted_text.append(f"<位置信息>这是第{chunk['page_number']}页，属于{chunk['section']}章节。</位置信息>")
        
        # 4. 上文（如果存在）
        if chunk['prev_content']:
            formatted_text.append(f"<上文>{chunk['prev_content']}</上文>")
        
        # 5. 主要内容
        formatted_text.append(f"<正文内容>\n{chunk['current_content']}\n</正文内容>")
        
        # 6. 下文（如果存在）
        if chunk['next_content']:
            formatted_text.append(f"<下文>{chunk['next_content']}</下文>")
            
        return "\n".join(formatted_text)

    def create_embeddings(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> None:
        """为所有文档块创建嵌入向量并存入ChromaDB"""
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
                
                for chunk_idx, chunk in enumerate(batch_chunks):
                    # 使用新的格式化函数
                    text_to_embed = self._format_text_for_embedding(chunk)
                    
                    # 检查metadata中的None值
                    for key, value in chunk['metadata'].items():
                        if value is None:
                            logger.error(f"发现None值！批次 {i}, 块 {chunk_idx}, 键 {key}")
                            logger.error(f"完整metadata: {json.dumps(chunk['metadata'], ensure_ascii=False, indent=2)}")
                            chunk['metadata'][key] = ""  # 将None替换为空字符串
                    
                    batch_texts.append(text_to_embed)
                    batch_metadatas.append(chunk['metadata'])
                    batch_ids.append(f"chunk_{chunk['page_number']}_{hash(chunk['current_content'])}")
                
                try:
                    # 批量创建嵌入向量
                    batch_embeddings = [self._create_embedding(text) for text in batch_texts]
                    
                    # 在添加之前打印metadata
                    logger.info(f"正在添加批次 {i}, metadata示例: {json.dumps(batch_metadatas[0], ensure_ascii=False)}")
                    
                    # 批量存储
                    collection.add(
                        embeddings=batch_embeddings,
                        documents=batch_texts,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                except Exception as e:
                    logger.error(f"处理批次 {i} 时出错")
                    logger.error(f"错误信息: {str(e)}")
                    logger.error(f"问题批次的metadata: {json.dumps(batch_metadatas, ensure_ascii=False, indent=2)}")
                    raise
                
            logger.info("所有文档块处理完成")
            
        except Exception as e:
            logger.error(f"创建嵌入向量过程中出错: {str(e)}")
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

def main():
    try:
        processor = VectorProcessor()
        
        # 调试模式
        DEBUG_LIMIT = -1  # 设置为-1处理所有数据
        
        logger.info("开始准备文档块...")
        all_chunks = processor.prepare_chunks("output/data_with_abstracts.json")
        chunks = all_chunks[:DEBUG_LIMIT] if DEBUG_LIMIT > 0 else all_chunks
        
        logger.info("开始创建嵌入向量...")
        processor.create_embeddings(chunks)
        
        # 测试搜索
        logger.info("执行测试搜索...")
        test_queries = [
            "省政府奖学金的评选条件是什么？",
            "学位论文答辩的流程是怎样的？",
            "研究生培养方案的主要内容有哪些？"
        ]
        
        for query in test_queries:
            print(f"\n搜索查询: {query}")
            results = processor.search_similar(query, top_k=5)
            
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