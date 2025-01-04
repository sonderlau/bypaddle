from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime
from openai import OpenAI
import importlib.util
from event_bus import EventBus  # Import EventBus
import asyncio

# 动态导入 LLMRAG
spec = importlib.util.spec_from_file_location("llm_rag", "11-llm-rag.py")
rag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_module)
LLMRAG = rag_module.LLMRAG

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

    def build_recent_history(self, user_id: str, max_user_messages: int = 3, max_assistant_messages: int = 1) -> str:
        """构建最近的对话历史"""
        history = self.get_history(user_id)
        if not history:
            return ""

        filtered_history = []
        user_count = 0
        assistant_count = 0

        # 从后往前遍历历史
        for msg in reversed(history):
            if msg['role'] == 'assistant' and assistant_count < max_assistant_messages:
                filtered_history.append(msg)
                assistant_count += 1
            elif msg['role'] == 'user' and user_count < max_user_messages:
                filtered_history.append(msg)
                user_count += 1
            if user_count >= max_user_messages and assistant_count >= max_assistant_messages:
                break

        # 反转回来以保持时间顺序
        filtered_history.reverse()

        # 构建对话历史文本
        history_text = "\n".join([
            f"{'用户' if msg['role'] == 'user' else '助手'}: {msg['content']}"
            for msg in filtered_history
        ])

        return history_text

class HandbookQueryProcessor:
    """学生手册查询处理器"""
    
    def __init__(self, llm_client, conversation_manager):
        self.llm_client = llm_client
        self.conversation_manager = conversation_manager
        
    async def is_handbook_related(self, user_id: str, query: str) -> bool:
        """判断是否是学生手册相关查询"""
        recent_history = self.conversation_manager.build_recent_history(user_id)
        
        prompt = f"""请判断用户的问题是否与学生手册相关。只返回"yes"或"no"。

对话历史：
{recent_history}

问题：{query}

判断标准：
1. 学校制度相关的问题返回"yes"，具体包括：
普通高等学校学生管理规定,高等学校学生行为准则,学生伤害事故处理办法,学生奖学金评审办法,国家奖学金评比办法,国家励志奖学金评比办法,国家助学金评比办法,省政府奖学金评比办法,三好学生、优秀学生干部评比办法,学生综合素质测评实施办法(试行),学生资助对象认定办法,学生校内勤工助学管理办法,学生纪律处分实施细则(试行),学生解除处分管理办法（试行）,学生校内申诉管理规定,学生考试违规、作弊的认定办法,学生学籍管理实施细则（2023年修订）,学生转学管理办法（试行）,关于学生参加毕业设计（论文）的若干规定,普通全日制本科生提前毕业管理办法,本科毕业生学士学位授予细则（2023年修订）,学生证和学生校徽管理办法,关于学生参加学科竞赛活动有关学业奖励的规定(试行),关于学生参加学科竞赛活动有关学业奖励的补充规定,学生体育、艺术类竞赛成果奖励办法（试行）,大学生创新创业训练计划实施与管理办法,“互联网+”大学生创新创业大赛奖励办法（试行）,本科生学分制收费管理办法,学生学杂费收缴实施办法（试行）,大学生医药费管理办法,学生公寓管理办法（暂行）,课外教育选修
2. 以下情况返回"no"：
- 日常问候（你好、早上好等）
- 感谢用语（谢谢、感激等）
- 天气、新闻、娱乐等话题
- 与学校制度无关的学术问题

请只返回"yes"或"no"："""

        try:
            await asyncio.sleep(0)  # 让出控制权
            response = self.llm_client.chat.completions.create(
                model="qwen-long",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            await asyncio.sleep(0)  # 让出控制权
            return "yes" in response.choices[0].message.content.strip().lower()
        except Exception as e:
            logger.error(f"判断问题相关性时出错: {str(e)}")
            return False
            
    async def rewrite_query(self, user_id: str, query: str, history: List[Dict]) -> str:
        """根据上下文改写查询"""
        recent_history = self.conversation_manager.build_recent_history(user_id)
        
        prompt = f"""基于对话历史和当前问题，生成完整的查询语句。

对话历史：
{recent_history}

当前学生的问题：{query}

要求：
1. 理解用户真实意图，补充必要的上下文信息
2. 处理代词指代（比如"它"、"这个"等）
3. 保持查询的完整性和准确性
4. 如果当前问题是对上文的追问，需要将相关上下文合并
5. 如果是全新的问题，直接使用原问题

只返回改写后的查询语句："""

        try:
            await asyncio.sleep(0)  # 让出控制权
            response = self.llm_client.chat.completions.create(
                model="qwen-long",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            await asyncio.sleep(0)  # 让出控制权
            rewritten = response.choices[0].message.content.strip()
            logger.info(f"原始查询: {query}")
            logger.info(f"改写后查询: {rewritten}")
            return rewritten
        except Exception as e:
            logger.error(f"改写查询时出错: {str(e)}")
            return query

    def format_conversation_history(self, history: List[Dict]) -> str:
        """格式化对话历史"""
        if not history:
            return ""
            
        # 筛选最近的对话历史
        filtered_history = []
        user_count = 0
        assistant_count = 0
        
        # 从最近的消息开始向前遍历
        for msg in reversed(history):
            if msg['role'] == 'user' and user_count < 3:  # 保留最近3个用户消息
                filtered_history.append(msg)
                user_count += 1
            elif msg['role'] == 'assistant' and assistant_count < 1:  # 保留最近1个助手回复
                filtered_history.append(msg)
                assistant_count += 1
                
            if user_count >= 3 and assistant_count >= 1:
                break
                
        # 将筛选后的历史反转回正确的时间顺序
        filtered_history.reverse()
        
        # 构建对话历史文本
        history_text = "\n".join([
            f"{'用户' if msg['role'] == 'user' else '助手'}: {msg['content']}"
            for msg in filtered_history
        ])
        
        self.current_history = history_text  # 保存当前处理的历史，供 format_context_with_pages 使用
        return history_text

class ChatManager:
    """聊天管理器"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    async def handle_general_chat(self, message: str) -> str:
        """处理一般对话"""
        prompt = f"""你是一个专注于学生手册问答的AI助手。请对用户的输入生成合适的回复。

学生手册的内容的大纲为：
普通高等学校学生管理规定,高等学校学生行为准则,学生伤害事故处理办法,学生奖学金评审办法,国家奖学金评比办法,国家励志奖学金评比办法,国家助学金评比办法,省政府奖学金评比办法,三好学生、优秀学生干部评比办法,学生综合素质测评实施办法(试行),学生资助对象认定办法,学生校内勤工助学管理办法,学生纪律处分实施细则(试行),学生解除处分管理办法（试行）,学生校内申诉管理规定,学生考试违规、作弊的认定办法,学生学籍管理实施细则（2023年修订）,学生转学管理办法（试行）,关于学生参加毕业设计（论文）的若干规定,普通全日制本科生提前毕业管理办法,本科毕业生学士学位授予细则（2023年修订）,学生证和学生校徽管理办法,关于学生参加学科竞赛活动有关学业奖励的规定(试行),关于学生参加学科竞赛活动有关学业奖励的补充规定,学生体育、艺术类竞赛成果奖励办法（试行）,大学生创新创业训练计划实施与管理办法,“互联网+”大学生创新创业大赛奖励办法（试行）,本科生学分制收费管理办法,学生学杂费收缴实施办法（试行）,大学生医药费管理办法,学生公寓管理办法（暂行）,课外教育选修


用户输入：{message}

要求：
1. 保持友好专业的语气
2. 对问候和感谢要简短得体回应
3. 提醒用户你主要回答学生手册相关的问题
4. 可以举例说明你可以回答哪些类型的问题

回复："""

        try:
            response = self.llm_client.chat.completions.create(
                model="qwen-long",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"处理一般对话时出错: {str(e)}")
            return "抱歉，我暂时无法回应。我是学生手册问答助手，主要回答关于学籍、奖学金、校规校纪等问题。"

class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self, rag_system, llm_client):
        self.conversation_manager = ConversationManager()
        self.handbook_processor = HandbookQueryProcessor(llm_client, self.conversation_manager)
        self.chat_manager = ChatManager(llm_client)
        self.rag_system = rag_system
        self.event_bus = EventBus()  # 使用单例模式的 EventBus
        
    async def process_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """处理用户消息"""
        try:
            logger.info(f"[状态] [用户ID:{user_id}] 开始处理用户消息")
            
            # 记录用户消息
            self.conversation_manager.add_message(user_id, "user", message)
            
            # 获取历史记录
            history = self.conversation_manager.get_history(user_id)
            logger.info(f"[信息] [用户ID:{user_id}] 当前对话历史数量: {len(history)}")
            
            # 始终调用 LLM 判断当前意图
            is_handbook_query = await self.handbook_processor.is_handbook_related(user_id, message)
            logger.info(f"[信息] [用户ID:{user_id}] 问题意图判断: {'手册相关' if is_handbook_query else '一般对话'}")
            
            if is_handbook_query:
                logger.info(f"[状态] [用户ID:{user_id}] 正在处理手册相关查询")
                await asyncio.sleep(0)
                
                rewritten_query = await self.handbook_processor.rewrite_query(user_id, message, history)
                logger.info(f"[信息] [用户ID:{user_id}] 原始问题: {message}")
                logger.info(f"[信息] [用户ID:{user_id}] 改写后的问题: {rewritten_query}")
                await asyncio.sleep(0)
                
                logger.info(f"[状态] [用户ID:{user_id}] 正在搜索相关内容")
                result = await asyncio.to_thread(
                    self.rag_system.answer_question,
                    query=rewritten_query,
                    return_context=True
                )
                await asyncio.sleep(0)
                
                logger.info(f"[状态] [用户ID:{user_id}] 正在生成回答")
                self.conversation_manager.add_message(
                    user_id, "assistant", result["answer"], intent="handbook"
                )
                
                logger.info(f"[状态] [用户ID:{user_id}] 处理完成")
                return {
                    "intent": "handbook",
                    "answer": result["answer"],
                    "context": result.get("context"),
                    "rewritten_query": rewritten_query
                }
            else:
                logger.info(f"[状态] [用户ID:{user_id}] 正在处理一般对话")
                response = await self.chat_manager.handle_general_chat(message)
                self.conversation_manager.add_message(
                    user_id, "assistant", response, intent="chat"
                )
                
                logger.info(f"[状态] [用户ID:{user_id}] 处理完成")
                return {
                    "intent": "chat",
                    "answer": response,
                    "context": None
                }
            
        except Exception as e:
            logger.error(f"[错误] [用户ID:{user_id}] 处理消息时出错: {str(e)}")
            raise

async def main():
    """测试工作流"""
    try:
        # 配置
        DATA_PATH = "output/data_with_abstracts.json"
        API_KEY = "sk-c3b22834c96a4f368657ad8eafa1999f"
        
        # 初始化 RAG 系统
        rag_system = LLMRAG(
            data_path=DATA_PATH,
            api_key=API_KEY,
            initial_top_k=15,
            final_top_k=5
        )
        
        # 初始化 OpenAI 客户端
        llm_client = OpenAI(
            api_key=API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 初始化工作流管理器
        workflow = WorkflowManager(rag_system, llm_client)
        
        # 测试用例
        test_cases = [
            "你好",
            # 连续对话测试
            "奖学金有哪些类型？",
            "国家奖学金的具体要求是什么？",
            "如何申请？",
            
            # 非手册问题

            "谢谢你的帮助"
        ]
        
        # 模拟两个用户的对话
        user_ids = ["user1"]
        
        for user_id in user_ids:
            print(f"\n=== 测试用户 {user_id} ===")
            
            for query in test_cases:
                print(f"\n用户问题: {query}")
                
                # 处理消息
                result = await workflow.process_message(user_id, query)
                
                # 打印结果
                print(f"答案: {result['answer']}")
                if result.get('context'):
                    print(f"参考上下文: {result['context']}")
                    
                print("-" * 50)
                
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
