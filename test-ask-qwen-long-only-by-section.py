from typing import List, Dict, Any
import json
import logging
from openai import OpenAI
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DocumentQA:
    def __init__(self, index_path: str, data_path: str, api_key: str):
        """初始化文档问答系统
        
        Args:
            index_path: 目录索引文件路径（index.json）
            data_path: 文档内容文件路径（data.json）
            api_key: DashScope API密钥
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 加载索引和数据
        # 检查文件存在
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引文件不存在：{index_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在：{data_path}")
        with open(index_path, 'r', encoding='utf-8') as f:
            self.index = json.load(f)
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def _find_relevant_section(self, query: str) -> Dict[str, Any]:
        """使用LLM找到最相关的章节
        
        Args:
            query: 用户问题
            
        Returns:
            包含section和页码范围的字典
        """
        # 构建提示词，列出所有章节供LLM选择
        sections_prompt = "文档包含以下章节：\n"
        for item in self.index:
            sections_prompt += f"- {item['section']}\n"
            
        prompt = f"""基于用户的问题，从上述章节中选择最相关的一个章节。
考虑章节的主题和内容范围，选择最可能包含答案的章节。
只需返回章节名称，无需其他解释。
===========
用户问题：{query}
==========
章节目录：{sections_prompt}
=========
例如

普通高等学校学生管理规定

"""
        logging.info(f"使用LLM找到最相关的章节: {prompt}")

        response = self.client.chat.completions.create(
            model="qwen-long",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0
        )
        
        selected_section = response.choices[0].message.content.strip()
        selected_section = selected_section.replace("\n", "").replace("```", "")
        
        def get_matching_ratio(s1: str, s2: str) -> float:
            """计算两个字符串的匹配率（相对于较长字符串的长度）"""
            common_len = longest_common_substring(s1, s2)
            max_len = max(len(s1), len(s2))
            return common_len / max_len if max_len > 0 else 0


        # 1. 先尝试精确匹配
        for section in self.index:
            if section['section'] == selected_section:
                logging.info(f"找到精确匹配章节: {section['section']}")
                return section
        
        # 2. 如果精确匹配失败，使用相对匹配率
        logging.warning(f"未找到精确匹配章节: {selected_section}，尝试模糊匹配")
        best_match = None
        best_ratio = 0
        
        for section in self.index:
            ratio = get_matching_ratio(section['section'], selected_section)
            logging.debug(f"章节'{section['section']}'的匹配率: {ratio:.2%}")
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = section
        
        if best_match:
            logging.info(f"找到最佳模糊匹配章节: {best_match['section']}, 匹配率: {best_ratio:.2%}")
            return best_match
        
        # 3. 如果都失败了，返回None
        logging.warning(f"未找到任何匹配章节，使用默认章节: {self.index[0]['section']}")
        return None
        
    def _get_section_content(self, section: Dict[str, Any]) -> str:
        """获取指定章节范围内的所有内容
        
        Args:
            section: 包含start_page和end_page的章节信息
            
        Returns:
            合并后的文本内容
        """
        content = []
        for page in self.data['pages']:
            page_num = page.get('document_page')
            if page_num is None:
                page_num = page['page_number']
                
            if (section['start_page'] <= page_num <= section['end_page']):
                # 添加页面文本内容
                if page['content'].strip():
                    content.append(page['content'])
                    
                # 如果页面包含表格，添加表格分析结果
                for table in page['tables']:
                    if 'llm_analysis' in table:
                        content.append(table['llm_analysis'])
                        
        return "\n".join(content)
        
    def answer_question(self, query: str) -> str:
        """回答用户问题（非流式）
        
        Args:
            query: 用户问题
            
        Returns:
            生成的答案
        """
        # 1. 找到相关章节
        relevant_section = self._find_relevant_section(query)
        # 2. 获取章节内容
        context = self._get_section_content(relevant_section)
        
        # 3. 构建最终提示词
        prompt = f"""请基于以下背景信息回答用户的问题。如果无法从背景信息中找到答案，请明确说明。
回答要准确、完整，并尽可能直接引用原文的具体内容。

背景信息：
=======
{context}
=======
用户问题：{query}
"""
        logging.info(f"根据学生手册进行回答: {prompt}")

        # 4. 调用LLM生成答案
        response = self.client.chat.completions.create(
            model="qwen-long",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.2
        )
        
        ret = response.choices[0].message.content.strip()
        ret = ret.replace("\n", "").replace("```", "")
        return ret

    async def answer_question_stream(self, query: str):
        """流式回答用户问题
        
        Args:
            query: 用户问题
            
        Yields:
            生成的答案片段
        """
        # 1. 找到相关章节
        relevant_section = self._find_relevant_section(query)
        # 2. 获取章节内容
        context = self._get_section_content(relevant_section)
        
        # 3. 构建最终提示词
        prompt = f"""请基于以下背景信息回答用户的问题。如果无法从背景信息中找到答案，请明确说明。
回答要准确、完整，并尽可能直接引用原文的具体内容。

背景信息：
=======
{context}
=======
用户问题：{query}
"""
        logging.info(f"根据学生手册进行流式回答: {prompt}")

        # 4. 调用LLM生成流式答案
        response = self.client.chat.completions.create(
            model="qwen-long",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.2,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

# 使用示例更新
if __name__ == "__main__":
    import os
    import asyncio
    
    async def main():
        # 从环境变量获取API密钥
        dashscope_api_key = os.getenv("DASH_SCOPE_API_KEY","")
        qa = DocumentQA(
            index_path="output/index.json",
            data_path="output/data.json",
            api_key=dashscope_api_key
        )
        
        # 测试问题
        question = "毕业设计有几个学分？"
        
        # 测试流式回答
        print(f"问题：{question}")
        print("流式答案：", end="", flush=True)
        async for chunk in qa.answer_question_stream(question):
            print(chunk, end="", flush=True)
        print()  # 打印换行

    # 运行异步主函数
    asyncio.run(main())