import json
import logging
from typing import Dict, Any, List
from openai import OpenAI
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SectionAbstractGenerator:
    def __init__(self, api_key: str):
        """初始化摘要生成器
        
        Args:
            api_key: DashScope API密钥
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
    def generate_abstract(self, section_content: str, section_name: str) -> str:
        """为章节生成摘要
        
        Args:
            section_content: 章节完整内容
            section_name: 章节名称
            
        Returns:
            生成的摘要
        """
        prompt = f"""请为以下手册的章节内容生成一个详细的摘要。读者是该学院的学生，对于手册有基本的了解。摘要应该：
1. 概括章节的主要内容和要点
2. 保留重要的规定、数字和具体要求
3. 长度控制在300-500字
4. 使用简洁清晰的语言
5. 如“确保评审过程公开、公平、公正。遵守学校规章制度”这种常识性文字可以忽略。

章节名称：{section_name}
章节内容：
==========
{section_content}
==========
"""
        try:
            response = self.client.chat.completions.create(
                model="qwen-long",
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"生成摘要时出错: {str(e)}")
            return ""

def add_section_abstracts(data_path: str, index_path: str, output_path: str, api_key: str, debug_limit: int = -1) -> None:
    """给数据添加章节摘要
    
    Args:
        data_path: 包含章节信息的数据JSON路径
        index_path: 目录索引JSON路径
        output_path: 输出JSON路径
        api_key: DashScope API密钥
        debug_limit: 调试模式，限制处理的章节数量，-1表示处理所有章节
    """
    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(index_path, 'r', encoding='utf-8') as f:
        index = json.load(f)
        
    # 检查是否存在临时文件
    temp_file = "output/temp_abstracts.json"
    if os.path.exists(temp_file):
        with open(temp_file, 'r', encoding='utf-8') as f:
            section_abstracts = json.load(f)
            logging.info(f"从临时文件加载了 {len(section_abstracts)} 个已生成的摘要")
    else:
        section_abstracts = {}
        
    generator = SectionAbstractGenerator(api_key)
    
    # 收集每个章节的完整内容
    section_contents: Dict[str, List[str]] = {}
    for page in data['pages']:
        section = page.get('section', '未知章节')
        if section not in section_contents:
            section_contents[section] = []
            
        # 添加页面文本内容
        if page['content'].strip():
            section_contents[section].append(page['content'])
            
        # 添加表格分析内容
        for table in page.get('tables', []):
            if table.get('llm_analysis'):
                section_contents[section].append(table['llm_analysis'])
    
    # 为每个章节生成摘要
    total_sections = len(section_contents)
    
    # 如果是调试模式，只处理指定数量的章节
    if debug_limit > 0:
        sections_to_process = list(section_contents.items())[:debug_limit]
        logging.info(f"调试模式：处理前 {debug_limit} 个章节（共 {total_sections} 个）")
    else:
        sections_to_process = section_contents.items()
        logging.info(f"处理所有 {total_sections} 个章节")
    
    try:
        for section, contents in sections_to_process:
            # 如果该章节已经有摘要，跳过
            if section in section_abstracts:
                logging.info(f"章节 '{section}' 已有摘要，跳过")
                continue
                
            full_content = "\n".join(contents)
            logging.info(f"正在为章节 '{section}' 生成摘要...")
            abstract = generator.generate_abstract(full_content, section)
            
            if abstract:  # 只在成功生成摘要时保存
                section_abstracts[section] = abstract
                logging.info(f"章节 '{section}' 摘要生成完成，长度：{len(abstract)} 字符")
                
                # 保存中间结果
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(section_abstracts, f, ensure_ascii=False, indent=2)
                logging.info(f"已保存中间结果到 {temp_file}")
            else:
                logging.error(f"章节 '{section}' 摘要生成失败")
    
    except KeyboardInterrupt:
        logging.info("收到中断信号，保存当前进度...")
    except Exception as e:
        logging.error(f"处理过程中出错: {str(e)}")
    finally:
        # 更新数据结构
        data['section_abstracts'] = section_abstracts
        
        # 保存最终结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"最终结果已保存到 {output_path}")
        
        # 如果所有章节都处理完了，删除临时文件
        if len(section_abstracts) == total_sections:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logging.info("所有章节处理完成，已删除临时文件")

if __name__ == "__main__":
    # 从环境变量获取API密钥
    api_key = "sk-c3b22834c96a4f368657ad8eafa1999f"

        
    add_section_abstracts(
        data_path="output/data_with_sections.json",
        index_path="output/index.json",
        output_path="output/data_with_abstracts.json",
        api_key=api_key,
        debug_limit=-1  # 只处理前5个章节
    )
