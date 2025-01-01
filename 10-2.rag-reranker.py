from rank_bm25 import BM25Okapi
import numpy as np
import json
import logging
from pathlib import Path
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import importlib.util

# 动态导入以数字开头的模块
spec = importlib.util.spec_from_file_location("vector_processor", "7-1.vector-with-abstract.py")
vector_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vector_module)
VectorProcessor = vector_module.VectorProcessor

# 设置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridSearcher:
    def __init__(self, vector_processor):
        self.vector_processor = vector_processor
        self.bm25 = None
        self.corpus = []
        self.alpha = 0.7  # 调节向量检索和BM25的权重
        
    def prepare_bm25(self, chunks):
        """准备BM25索引"""
        # 对文档分词
        self.corpus = [chunk['current_content'] for chunk in chunks]
        tokenized_corpus = [list(doc) for doc in self.corpus]  # 简单按字符分词,实际中可以用jieba
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.chunks = chunks
    
    def hybrid_search(self, query: str, top_k: int = 5):
        """混合检索方法"""
        # 1. 向量检索
        vector_results = self.vector_processor.search_similar(query, top_k=top_k*2)
        vector_scores = {res['metadata']['page_number']: res['similarity_score'] 
                        for res in vector_results}
        
        # 2. BM25检索
        tokenized_query = list(query)  # 简单按字符分词
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # 归一化BM25分数
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
        
        # 3. 融合得分
        final_scores = []
        for idx, chunk in enumerate(self.chunks):
            page_num = chunk['page_number']
            vector_score = vector_scores.get(page_num, 0)
            bm25_score = bm25_scores[idx]
            
            # 加权融合
            final_score = self.alpha * vector_score + (1 - self.alpha) * bm25_score
            final_scores.append({
                'chunk': chunk,
                'score': final_score,
                'vector_score': vector_score,
                'bm25_score': bm25_score
            })
        
        # 4. 排序并返回top-k结果
        final_scores.sort(key=lambda x: x['score'], reverse=True)
        return final_scores[:top_k]

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
        # 设备选择
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        logger.info(f"Reranker使用设备: {self.device}")
        
        self.model = self.model.to(self.device)
        
    def rerank(self, query: str, candidates: list, batch_size: int = 8):
        """对候选文档进行重排序"""
        all_scores = []
        
        # 批量处理
        for i in range(0, len(candidates), batch_size):
            batch_candidates = candidates[i:i + batch_size]
            
            # 准备输入
            pairs = []
            for candidate in batch_candidates:
                pairs.append([query, candidate['chunk']['current_content']])
                
            # 编码
            features = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.device)
            
            # 计算分数
            with torch.no_grad():
                scores = self.model(**features).logits.squeeze(-1)
                scores = torch.sigmoid(scores).cpu().numpy()
                
            all_scores.extend(scores)
        
        # 更新分数并重新排序
        for candidate, rerank_score in zip(candidates, all_scores):
            candidate['rerank_score'] = float(rerank_score)
            
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        return candidates

def search_with_rerank(query: str, hybrid_searcher: HybridSearcher, reranker: Reranker, 
                      initial_top_k: int = 30, final_top_k: int = 5):
    # 1. 混合检索召回候选
    candidates = hybrid_searcher.hybrid_search(query, top_k=initial_top_k)
    
    # 2. Rerank重排序
    reranked_results = reranker.rerank(query, candidates)
    
    return reranked_results[:final_top_k]

def format_search_results(results, show_scores: bool = False):
    """格式化搜索结果"""
    formatted_results = []
    for idx, result in enumerate(results, 1):
        chunk = result['chunk']
        text = f"\n=== 结果 {idx} ===\n"
        text += f"章节: {chunk['section']}\n"
        if chunk['section_abstract']:
            text += f"章节摘要: {chunk['section_abstract']}\n"
        text += f"页码: {chunk['page_number']}\n"
        text += f"内容: {chunk['current_content']}\n"
        
        if show_scores:
            text += f"重排序分数: {result.get('rerank_score', 0):.4f}\n"
            text += f"混合检索分数: {result.get('score', 0):.4f}\n"
            text += f"向量检索分数: {result.get('vector_score', 0):.4f}\n"
            text += f"BM25分数: {result.get('bm25_score', 0):.4f}\n"
            
        formatted_results.append(text)
    return "\n".join(formatted_results)

def main():
    # 加载向量处理器
    processor = VectorProcessor()
    
    # 加载数据
    logger.info("加载数据...")
    with open("output/data_with_abstracts.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 准备chunks
    logger.info("准备文档块...")
    chunks = processor.prepare_chunks("output/data_with_abstracts.json")
    
    # 初始化搜索器和重排序器
    logger.info("初始化搜索器和重排序器...")
    hybrid_searcher = HybridSearcher(processor)
    hybrid_searcher.prepare_bm25(chunks)
    reranker = Reranker()
    
    # 测试查询
    test_queries = [
        "学籍异动包括哪些情况？",
        "学生申请休学的流程是什么？",
        "如何处理学生考试作弊？",
        "学生证补办流程",
        "奖学金评定标准"
    ]
    
    logger.info("开始搜索...")
    for query in test_queries:
        print(f"\n\n查询: {query}")
        results = search_with_rerank(
            query=query,
            hybrid_searcher=hybrid_searcher,
            reranker=reranker,
            initial_top_k=10,
            final_top_k=3
        )
        print(format_search_results(results, show_scores=True))

if __name__ == "__main__":
    main()