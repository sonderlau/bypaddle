from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self, max_history: int = 5):
        self.conversations = {}
        self.max_history = max_history
        
    def add_message(self, user_id: str, role: str, content: str, intent: str = None):
        """添加消息到历史，增加意图标记"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
            
        self.conversations[user_id].append({
            'role': role,
            'content': content,
            'intent': intent,
            'timestamp': datetime.now().isoformat()
        })
        
        if len(self.conversations[user_id]) > self.max_history * 2:
            self.conversations[user_id] = self.conversations[user_id][-self.max_history * 2:]
            
    def get_history(self, user_id: str) -> List[Dict]:
        return self.conversations.get(user_id, [])

class HandbookQueryProcessor:
    """学生手册查询处理器"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    async def is_handbook_related(self, query: str) -> bool:
        """判断是否是学生手册相关查询"""
        prompt = f"""请判断用户的问题是否与学生手册相关。只返回"yes"或"no"。

问题：{query}

判断标准：
1. 学校规章制度相关的问题返回"yes"，
具体包括：普通高等学校学生管理规定,高等学校学生行为准则,学生伤害事故处理办法,学生奖学金评审办法,国家奖学金评比办法,国家励志奖学金评比办法,国家助学金评比办法,省政府奖学金评比办法,三好学生、优秀学生干部评比办法,学生综合素质测评实施办法(试行),学生资助对象认定办法,学生校内勤工助学管理办法,学生纪律处分实施细则(试行),学生解除处分管理办法（试行）,学生校内申诉管理规定,学生考试违规、作弊的认定办法,学生学籍管理实施细则（2023年修订）,学生转学管理办法（试行）,关于学生参加毕业设计（论文）的若干规定,普通全日制本科生提前毕业管理办法,本科毕业生学士学位授予细则（2023年修订）,学生证和学生校徽管理办法,关于学生参加学科竞赛活动有关学业奖励的规定(试行),关于学生参加学科竞赛活动有关学业奖励的补充规定,学生体育、艺术类竞赛成果奖励办法（试行）,大学生创新创业训练计划实施与管理办法,“互联网+”大学生创新创业大赛奖励办法（试行）,本科生学分制收费管理办法,学生学杂费收缴实施办法（试行）,大学生医药费管理办法,学生公寓管理办法（暂行）,课外教育选修
2. 日常问候、感谢等社交用语返回"no"
3. 与学校规章制度无关的问题返回"no"，例如今天天气、政治人物、新闻。

回答："""

        try:
            response = await self.llm_client.chat.completions.create(
                model="qwen-long",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content.strip().lower().contains("yes")
        except Exception as e:
            logger.error(f"判断问题相关性时出错: {str(e)}")
            return False
            
    async def rewrite_query(self, query: str, history: List[Dict]) -> str:
        """根据上下文改写查询"""
        if not history:
            return query
            
        # 筛选最近的对话历史
        filtered_history = []
        user_count = 0
        assistant_count = 0
        
        # 从最近的消息开始向前遍历
        for msg in reversed(history):
            if msg['role'] == 'user' and user_count < 3:
                filtered_history.append(msg)
                user_count += 1
            elif msg['role'] == 'assistant' and assistant_count < 1:
                filtered_history.append(msg)
                assistant_count += 1
                
            # 如果都达到限制，就停止
            if user_count >= 3 and assistant_count >= 1:
                break
                
        # 将筛选后的历史反转回正确的时间顺序
        filtered_history.reverse()
        
        # 构建对话历史文本
        context = "\n".join([
            f"{'用户' if msg['role'] == 'user' else '助手'}: {msg['content']}"
            for msg in filtered_history
        ])
        
        prompt = f"""基于以下对话历史和当前问题，生成一个完整的查询语句。

对话历史：
{context}

当前问题：{query}

要求：
1. 结合上下文理解用户真实意图
2. 补充必要的上下文信息
3. 生成完整的查询语句
4. 只返回改写后的查询语句，不要其他解释

改写后的查询："""

        try:
            response = await self.llm_client.chat.completions.create(
                model="qwen-long",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"改写查询时出错: {str(e)}")
            return query

    def format_context_with_pages(self, context: str) -> str:
        """格式化上下文，突出显示页码信息"""
        prompt = f"""请基于以下参考信息回答用户的问题。要求：
1. 答案必须准确，与参考信息保持一致
2. 如果参考信息不足以完整回答问题，请明确指出
3. 合理组织答案结构，适当分点说明
4. 在回答的最后，用"参考来源："说明信息来自哪些页码
5. 如果有多个页码，请全部列出，格式为：参考来源：第X页、第Y页

参考信息：
{context}

请生成解答："""
        return prompt

class ChatManager:
    """聊天管理器"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    async def handle_general_chat(self, message: str) -> str:
        """处理一般对话"""
        prompt = f"""你是一个学生手册问答助手。请对用户的输入生成合适的回复。

用户输入：{message}

要求：
1. 对问候语回应友好
2. 对感谢语回应得体
3. 对于非学生手册相关的问题，礼貌地表示只能回答学生手册相关的问题
4. 保持回复简短自然

回复："""

        try:
            response = await self.llm_client.chat.completions.create(
                model="qwen-long",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"处理一般对话时出错: {str(e)}")
            return "抱歉，我只能回答与学生手册相关的问题。"

class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self, rag_system, llm_client):
        self.conversation_manager = ConversationManager()
        self.handbook_processor = HandbookQueryProcessor(llm_client)
        self.chat_manager = ChatManager(llm_client)
        self.rag_system = rag_system
        
    async def process_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """处理用户消息"""
        try:
            # 1. 添加用户消息到历史
            self.conversation_manager.add_message(user_id, "user", message)
            
            # 2. 判断是否是学生手册相关查询
            is_handbook_query = await self.handbook_processor.is_handbook_related(message)
            
            if is_handbook_query:
                # 3a. 处理学生手册相关查询
                # 获取历史记录并改写查询
                history = self.conversation_manager.get_history(user_id)
                rewritten_query = await self.handbook_processor.rewrite_query(message, history)
                
                # 使用RAG系统回答
                result = await self.rag_system.answer_question(
                    query=rewritten_query,
                    return_context=True,
                    conversation_history=history,
                    context_prompt=self.handbook_processor.format_context_with_pages
                )
                
                # 记录回复
                self.conversation_manager.add_message(
                    user_id, "assistant", result["answer"], intent="handbook"
                )
                
                return {
                    "answer": result["answer"],
                    "context": result.get("context"),
                    "intent": "handbook",
                    "rewritten_query": rewritten_query
                }
            else:
                # 3b. 处理一般对话
                response = await self.chat_manager.handle_general_chat(message)
                self.conversation_manager.add_message(
                    user_id, "assistant", response, intent="chat"
                )
                
                return {
                    "answer": response,
                    "context": None,
                    "intent": "chat"
                }
                
        except Exception as e:
            logger.error(f"处理消息时出错: {str(e)}")
            raise
