from fastapi import FastAPI, HTTPException, Request, WebSocket
from pydantic import BaseModel
import uvicorn
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv
import logging
import importlib.util
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from openai import OpenAI
import asyncio
from event_bus import EventBus
import httpx

# 获取项目根目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 获取根日志记录器
root_logger = logging.getLogger()

# 添加WebSocket处理器到根日志记录器
class WebSocketHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            message = {
                "type": "log",
                "message": msg
            }
            import json
            # Get the running event loop if it exists, otherwise create a new one
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the broadcast coroutine
            if loop.is_running():
                loop.create_task(manager.broadcast(json.dumps(message)))
            else:
                loop.run_until_complete(manager.broadcast(json.dumps(message)))
        except Exception as e:
            import sys
            print(f"Error in WebSocket handler: {str(e)}", file=sys.stderr)
            self.handleError(record)

class HTTPFilter(logging.Filter):
    def filter(self, record):
        # Filter out HTTP request logs containing dashscope.aliyuncs.com
        return "dashscope.aliyuncs.com" not in record.getMessage()

websocket_handler = WebSocketHandler()
websocket_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
websocket_handler.addFilter(HTTPFilter())  # Add the filter
root_logger.addHandler(websocket_handler)

logger = logging.getLogger(__name__)

# 动态导入 LLMRAG
spec = importlib.util.spec_from_file_location("llm_rag", "11-llm-rag.py")
rag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_module)
LLMRAG = rag_module.LLMRAG

# 动态导入 WorkflowManager
spec_workflow = importlib.util.spec_from_file_location("workflow", "12-llm-workflow.py")
workflow_module = importlib.util.module_from_spec(spec_workflow)
spec_workflow.loader.exec_module(workflow_module)
WorkflowManager = workflow_module.WorkflowManager

# 创建 FastAPI 应用
app = FastAPI(
    title="学生手册问答系统",
    description="基于学生手册",
    version="0.0.1"
)

# 创建模板目录
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# 挂载静态文件
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# WebSocket连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                continue

manager = ConnectionManager()

# 请求模型
class QuestionRequest(BaseModel):
    question: str
    user_id: str
    return_context: bool = False

# 响应模型
class QuestionResponse(BaseModel):
    answer: str
    context: Optional[str] = None
    
# 错误响应模型
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# 全局 RAG 实例
rag_instance = None
workflow_manager = None
event_bus = EventBus()

async def forward_to_websocket(message: str):
    """转发消息到WebSocket"""
    try:
        await manager.broadcast(message)
    except Exception as e:
        logger.error(f"转发到WebSocket失败: {str(e)}")

# 订阅 EventBus 消息
event_bus.subscribe(forward_to_websocket)

@app.on_event("startup")
async def startup_event():
    """服务启动时初始化系统"""
    global rag_instance, workflow_manager
    
    try:
        # 配置参数
        DATA_PATH = "output/data_with_abstracts.json"
        API_KEY="sk-c3b22834c96a4f368657ad8eafa1999f"
        
        # 初始化OpenAI客户端
        http_client = httpx.Client()
        llm_client = OpenAI(
            api_key=API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            http_client=http_client
        )
        
        # 初始化RAG系统
        rag_instance = LLMRAG(
            data_path=DATA_PATH,
            api_key=API_KEY,
            initial_top_k=30,
            final_top_k=5
        )
        
        # 初始化工作流管理器
        workflow_manager = WorkflowManager(rag_instance, llm_client)
        
        logger.info("系统初始化完成")
    except Exception as e:
        logger.error(f"系统初始化失败: {str(e)}")
        raise

@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """处理问答请求"""
    try:
        # 使用工作流管理器处理问题
        result = await workflow_manager.process_message(
            user_id=request.user_id,
            message=request.question
        )
        
        # 如果结果是字符串，转换为字典格式
        if isinstance(result, str):
            result = {"answer": result, "context": None}
        
        return QuestionResponse(
            answer=result["answer"] if isinstance(result, dict) else result,
            context=result.get("context") if isinstance(result, dict) else None
        )
        
    except Exception as e:
        logger.error(f"处理问题时出错: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "rag_initialized": rag_instance is not None,
        "workflow_initialized": workflow_manager is not None
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """提供Web界面"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        logger.error("WebSocket connection closed")
    finally:
        manager.disconnect(websocket)
        logger.info("WebSocket connection cleaned up")

def main():
    """主函数"""
    # 运行服务器
    uvicorn.run(
        "13-web:app",
        host="0.0.0.0",
        port=8011,
        reload=True
    )

if __name__ == "__main__":
    main()