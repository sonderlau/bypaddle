# 13-web.py
from fastapi import FastAPI, HTTPException, Request, WebSocket, Depends
from pydantic import BaseModel
import uvicorn
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv
import logging
import importlib.util
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from openai import OpenAI
import asyncio
from event_bus import EventBus
import httpx
import json

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
        return # 禁用WebSocket日志
        try:
            # 从 extra 中获取用户ID
            user_id = getattr(record, 'user_id', None)
            if user_id is None:
                return  # 如果没有用户ID，直接忽略这条日志
                
            msg = self.format(record)
            message = {
                "type": "log",
                "message": msg,
                "status": self.get_status_type(msg),
                "user_id": user_id
            }
            
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast(json.dumps(message)),
                    loop
                )
            else:
                loop.run_until_complete(
                    manager.broadcast(json.dumps(message))
                )
                
        except Exception as e:
            import sys
            print(f"Error in WebSocket handler: {str(e)}", file=sys.stderr)
            self.handleError(record)

    def get_status_type(self, msg):
        if "[状态]" in msg:
            return "status"
        elif "[信息]" in msg:
            return "info"
        elif "ERROR" in msg or "[错误]" in msg:
            return "error"
        elif "HTTP Request:" in msg:
            return "http"
        return "default"

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
        self.lock = asyncio.Lock()  # 添加锁来保护连接列表

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.active_connections.append(websocket)
            logger.info("[状态] 新的WebSocket连接已建立")

    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                logger.info("[状态] WebSocket连接已断开")

    async def broadcast(self, message: str):
        async with self.lock:
            dead_connections = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"[错误] 发送WebSocket消息失败: {str(e)}")
                    dead_connections.append(connection)
            
            # 清理失效的连接
            for dead in dead_connections:
                self.active_connections.remove(dead)

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
        import os 

        # 配置
        DATA_PATH = "output/data_with_abstracts.json"
        load_dotenv()  
        API_KEY=os.getenv("DASH_SCOPE_API_KEY")
        if not API_KEY:
            raise ValueError("DASH_SCOPE_API_KEY 环境变量未设置")
        
        # 初始化OpenAI客户端
        http_client = httpx.Client()  # 改用同步客户端
        llm_client = OpenAI(
            api_key=API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            http_client=http_client
        )
        
        # 初始化同步RAG系统
        rag_instance = LLMRAG(
            data_path=DATA_PATH,
            api_key=API_KEY,
            initial_top_k=15,
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
        # 创建一个带有用户ID的日志适配器
        logger_adapter = logging.LoggerAdapter(
            logger,
            {'user_id': request.user_id}
        )
        
        logger_adapter.info("开始处理您的问题...")
        
        result = await workflow_manager.process_message(
            user_id=request.user_id,
            message=request.question
        )
        
        logger_adapter.info("✅ 问题处理完成")
        
        return QuestionResponse(
            answer=result["answer"] if isinstance(result, dict) else result,
            context=result.get("context") if isinstance(result, dict) else None
        )
    except Exception as e:
        logger_adapter.error(f"处理问题时出错: {str(e)}")
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
    # 获取认证头
    try:
        await websocket.accept()
        while True:
            await websocket.receive_text()
    except Exception as e:
        logger.error(f"[错误] WebSocket连接出错: {str(e)}")
    finally:
        await manager.disconnect(websocket)

@app.get("/download-manual")
async def download_manual():
    """提供学生手册PDF下载"""
    pdf_path = "data/student-manual.pdf"  # 替换为实际的PDF文件路径
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF文件不存在")
    return FileResponse(
        pdf_path, 
        filename="杭电信工学生手册(2024版).pdf",
        media_type="application/pdf"
    )

@app.websocket("/ws/ask")
async def websocket_ask_endpoint(websocket: WebSocket):
    """处理流式问答的 WebSocket 连接"""
    try:
        await websocket.accept()
        logger.info("问答 WebSocket 连接已建立")
        
        while True:
            try:
                # 接收前端发送的问题
                data = await websocket.receive_json()
                question = data.get('question')
                user_id = data.get('user_id')
                return_context = data.get('return_context', True)
                
                if not question or not user_id:
                    await websocket.send_json({
                        "type": "error",
                        "content": "无效的请求数据"
                    })
                    continue
                
                # 流式生成答案
                async for response in workflow_manager.process_message_stream(
                    user_id=user_id,
                    message=question,
                    return_context=return_context
                ):
                    await websocket.send_json(response)
                    
                # 发送完成标记
                await websocket.send_json({
                    "type": "done",
                    "content": None
                })
                
            except Exception as e:
                logger.error(f"处理问题时出错: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "content": f"处理问题时出错: {str(e)}"
                })
                
    except Exception as e:
        logger.error(f"WebSocket 连接出错: {str(e)}")
    finally:
        await websocket.close()
        logger.info("问答 WebSocket 连接已断开")

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