from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
import logging
import importlib.util
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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
templates = Jinja2Templates(directory="templates")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 请求模型
class QuestionRequest(BaseModel):
    question: str
    user_id: str  # 添加用户ID字段
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

# 全局实例
workflow_manager = None

@app.on_event("startup")
async def startup_event():
    """服务启动时初始化系统"""
    global workflow_manager, rag_instance
    try:
        # 从环境变量获取配置
        DATA_PATH = "output/data_with_abstracts.json"
        API_KEY = "sk-c3b22834c96a4f368657ad8eafa1999f"
        api_key = os.getenv("API_KEY", API_KEY)
        if not api_key:
            raise ValueError("未设置 API_KEY 环境变量")
            
        data_path = os.getenv("DATA_PATH", DATA_PATH)
        
        # 初始化 RAG 系统
        logger.info("初始化 RAG 系统...")
        rag_instance = LLMRAG(
            data_path=data_path,
            api_key=api_key,
            initial_top_k=30,
            final_top_k=5
        )
        
        # 初始化 OpenAI 客户端
        llm_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 初始化工作流管理器
        workflow_manager = WorkflowManager(rag_instance, llm_client)
        logger.info("系统初始化完成")
        
    except Exception as e:
        logger.error(f"初始化系统失败: {str(e)}")
        raise

@app.post("/api/ask", 
         response_model=QuestionResponse,
         responses={
             200: {"description": "成功获取答案"},
             500: {"model": ErrorResponse, "description": "服务器内部错误"},
             400: {"model": ErrorResponse, "description": "请求参数错误"}
         })
async def ask_question(request: QuestionRequest):
    """处理问答请求"""
    try:
        if not workflow_manager:
            raise HTTPException(
                status_code=500,
                detail="系统未正确初始化"
            )
            
        # 记录请求
        logger.info(f"用户 {request.user_id} 提问: {request.question}")
        
        # 使用工作流管理器处理消息
        result = await workflow_manager.process_message(
            user_id=request.user_id,
            message=request.question
        )
        
        # 构造响应
        response = {
            "answer": result["answer"],
            "intent": result["intent"]
        }
        if request.return_context:
            response["context"] = result.get("context")
        if "rewritten_query" in result:
            response["rewritten_query"] = result["rewritten_query"]
            
        return response
        
    except Exception as e:
        logger.error(f"处理问题时出错: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "rag_initialized": rag_instance is not None
    }

# 添加首页路由
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """提供Web界面"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

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