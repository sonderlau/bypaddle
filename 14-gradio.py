import gradio as gr
import logging
import importlib.util
from typing import Dict, Any
import os
from openai import OpenAI
from datetime import datetime
import queue
import asyncio

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 动态导入 LLMRAG 和 WorkflowManager
spec = importlib.util.spec_from_file_location("llm_rag", "11-llm-rag.py")
rag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_module)
LLMRAG = rag_module.LLMRAG

spec_workflow = importlib.util.spec_from_file_location("workflow", "12-llm-workflow.py")
workflow_module = importlib.util.module_from_spec(spec_workflow)
spec_workflow.loader.exec_module(workflow_module)
WorkflowManager = workflow_module.WorkflowManager

class ChatInterface:
    def __init__(self):
        self.workflow_manager = None
        self.rag_instance = None
        self.logs = []
        self.setup_logging()
        self.initialize_system()
        
    def setup_logging(self):
        """设置日志处理"""
        pass  # 使用默认的日志配置即可

    def initialize_system(self):
        """初始化系统"""
        try:
            # 配置
            DATA_PATH = "output/data_with_abstracts.json"
            API_KEY = "sk-c3b22834c96a4f368657ad8eafa1999f"
            
            # 初始化 RAG 系统
            logger.info("正在初始化 RAG 系统...")
            self.rag_instance = LLMRAG(
                data_path=DATA_PATH,
                api_key=API_KEY,
                initial_top_k=30,
                final_top_k=5
            )
            
            # 初始化 OpenAI 客户端
            llm_client = OpenAI(
                api_key=API_KEY,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            
            # 初始化工作流管理器
            self.workflow_manager = WorkflowManager(self.rag_instance, llm_client)
            logger.info("系统初始化完成")
            
        except Exception as e:
            logger.error(f"初始化系统失败: {str(e)}")
            raise

    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(title="学生手册问答系统") as interface:
            gr.Markdown("# 学生手册问答系统")
            
            with gr.Row():
                # 左侧聊天区域
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="对话历史",
                        height=500,
                        type="messages"
                    )
                    msg = gr.Textbox(
                        label="请输入您的问题",
                        placeholder="请输入您的问题...",
                        show_label=True
                    )
                    with gr.Row():
                        submit_btn = gr.Button("发送")
                        clear_btn = gr.Button("清除历史")
                
                # 右侧日志区域
                with gr.Column(scale=1):
                    logs_display = gr.TextArea(
                        label="处理日志",
                        value="",
                        interactive=False,
                        lines=25,
                        max_lines=25
                    )

            def get_current_logs():
                return "\n".join(self.logs)
                
            async def user_input(message, history):
                if not message:
                    return "", history, ""
                
                self.logs = []  # 清空之前的日志
                
                try:
                    # 记录基本信息
                    self.logs.append(f"处理查询: {message}")
                    user_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    
                    # 阶段1：开始处理
                    self.logs.append("开始处理查询...")
                    self.logs.append(f"原始查询: {message}")
                    
                    # 阶段2：处理查询
                    self.logs.append("正在分析问题...")
                    result = await self.workflow_manager.process_message(
                        user_id=user_id,
                        message=message
                    )
                    self.logs.append("问题分析完成，正在生成回答...")
                    
                    # 阶段3：完成处理
                    if isinstance(result, dict) and "answer" in result:
                        self.logs.append("已生成回答")
                        history.append({"role": "assistant", "content": result["answer"]})
                    else:
                        self.logs.append("生成回答时遇到异常格式，将直接返回结果")
                        history.append({"role": "assistant", "content": str(result)})
                    
                    return "", history, "\n".join(self.logs)
                        
                except Exception as e:
                    error_message = f"错误: {str(e)}"
                    self.logs.append(error_message)
                    history.append({"role": "assistant", "content": f"抱歉，发生了错误：{error_message}"})
                    return "", history, "\n".join(self.logs)

            def clear_history():
                self.logs = []
                return None, [], ""

            def update_logs():
                return "\n".join(self.logs)

            # 设置按钮事件
            submit_btn.click(
                user_input,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot, logs_display]
            )
            
            msg.submit(
                user_input,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot, logs_display]
            )
            
            clear_btn.click(
                clear_history,
                inputs=None,
                outputs=[chatbot, logs_display]
            )

            # 设置自动刷新日志
            gr.update(update_logs, None, logs_display, every=1)

            return interface

def main():
    # 创建接口实例
    chat_interface = ChatInterface()
    
    # 启动Gradio服务
    interface = chat_interface.create_interface()
    interface.queue()  # 启用队列以支持异步操作
    interface.launch(
        server_name="0.0.0.0",
        server_port=8011,
        share=True
    )

if __name__ == "__main__":
    main()
