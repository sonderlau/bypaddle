import json
import logging
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def add_section_info(data_path: str, index_path: str, output_path: str) -> None:
    """给文档内容添加章节信息
    
    Args:
        data_path: 原始数据JSON路径
        index_path: 目录索引JSON路径
        output_path: 输出JSON路径
    """
    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(index_path, 'r', encoding='utf-8') as f:
        index = json.load(f)
        
    # 建立页码到章节的映射
    page_to_section = {}
    for section in index:
        for page_num in range(section['start_page'], section['end_page'] + 1):
            page_to_section[page_num] = section['section']
    
    # 为每个页面添加章节信息
    pages_processed = 0
    for page in data['pages']:
        # 优先使用document_page，如果没有则使用page_number
        page_num = page.get('document_page')
        if page_num is None:
            page_num = page['page_number']
            
        # 添加章节信息
        section = page_to_section.get(page_num)
        if section:
            page['section'] = section
            pages_processed += 1
        else:
            logging.warning(f"未找到页面 {page_num} 对应的章节信息")
            page['section'] = "未知章节"
    
    # 添加额外的元信息
    data['metadata'] = {
        'total_pages': len(data['pages']),
        'sections': list(set(page_to_section.values())),
        'preprocessing_info': {
            'pages_processed': pages_processed,
            'sections_count': len(set(page_to_section.values()))
        }
    }
    
    # 保存处理后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"处理完成！共处理 {pages_processed} 页")
    logging.info(f"包含 {len(data['metadata']['sections'])} 个章节")

def prepare_for_vector_search(data_path: str) -> List[Dict[str, Any]]:
    """准备用于向量搜索的文档片段
    
    Args:
        data_path: 已添加章节信息的JSON路径
        
    Returns:
        文档片段列表，每个片段包含内容、章节、页码等信息
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = []
    for page in data['pages']:
        # 处理正文内容
        if page['content'].strip():
            chunks.append({
                'content': page['content'],
                'section': page['section'],
                'page_number': page['page_number'],
                'document_page': page.get('document_page'),
                'type': 'text'
            })
        
        # 处理表格内容
        for table in page.get('tables', []):
            if table.get('llm_analysis'):
                chunks.append({
                    'content': table['llm_analysis'],
                    'section': page['section'],
                    'page_number': page['page_number'],
                    'document_page': page.get('document_page'),
                    'type': 'table',
                    'base64_image': table.get('base64_image')
                })
    
    return chunks

if __name__ == "__main__":
    # 添加章节信息
    add_section_info(
        data_path="output/raw.json",
        index_path="output/index.json",
        output_path="output/data_with_sections.json"
    )
    
    # 准备向量搜索的文档片段
    chunks = prepare_for_vector_search("output/data_with_sections.json")
    print(f"生成了 {len(chunks)} 个文档片段")