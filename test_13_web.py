import asyncio
import httpx
import websockets
import json
import logging
import base64
from typing import Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebAppTester:
    def __init__(self, base_url: str = "http://127.0.0.1:8011"):
        self.base_url = base_url
        self.auth_header = self._get_auth_header()
        
    def _get_auth_header(self) -> Dict[str, str]:
        """生成Basic认证头"""
        credentials = base64.b64encode(b"admin:password123").decode('utf-8')
        return {"Authorization": f"Basic {credentials}"}
        
    async def test_health(self) -> bool:
        """测试健康检查接口"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/health",
                    headers=self.auth_header
                )
                logger.info(f"健康检查响应: {response.json()}")
                return response.status_code == 200
            except Exception as e:
                logger.error(f"健康检查失败: {str(e)}")
                return False

    async def test_ask_question(self, questions: list) -> bool:
        """测试问答接口"""
        async with httpx.AsyncClient() as client:
            success = True
            for question in questions:
                try:
                    response = await client.post(
                        f"{self.base_url}/api/ask",
                        headers=self.auth_header,
                        json={
                            "question": question,
                            "user_id": "test-user-1",
                            "return_context": True
                        }
                    )
                    result = response.json()
                    logger.info(f"\n问题: {question}")
                    logger.info(f"答案: {result['answer']}")
                    if 'context' in result:
                        logger.info(f"上下文长度: {len(result['context'])} 字符")
                except Exception as e:
                    logger.error(f"问答测试失败: {str(e)}")
                    success = False
            return success

    async def test_websocket(self) -> bool:
        """测试WebSocket连接"""
        ws_url = f"ws://localhost:8011/ws"
        try:
            async with websockets.connect(
                ws_url,
                extra_headers=self.auth_header
            ) as websocket:
                logger.info("WebSocket连接成功")
                
                # 等待几条日志消息
                for _ in range(3):
                    try:
                        message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=5.0
                        )
                        data = json.loads(message)
                        logger.info(f"收到WebSocket消息: {data}")
                    except asyncio.TimeoutError:
                        break
                return True
        except Exception as e:
            logger.error(f"WebSocket测试失败: {str(e)}")
            return False

async def main():
    """运行所有测试"""
    tester = WebAppTester()
    
    # 测试用例
    test_questions = [
        "学生申请休学的流程是什么？",
        "国家奖学金的评选条件有哪些？",
        "考试作弊会受到什么处分？"
    ]
    
    # 运行测试
    logger.info("=== 开始测试 ===")
    
    # 1. 测试健康检查
    logger.info("\n1. 测试健康检查")
    health_ok = await tester.test_health()
    logger.info(f"健康检查测试{'通过' if health_ok else '失败'}")
    
    # 2. 测试问答接口
    logger.info("\n2. 测试问答接口")
    qa_ok = await tester.test_ask_question(test_questions)
    logger.info(f"问答接口测试{'通过' if qa_ok else '失败'}")
    
    # 3. 测试WebSocket
    logger.info("\n3. 测试WebSocket")
    ws_ok = await tester.test_websocket()
    logger.info(f"WebSocket测试{'通过' if ws_ok else '失败'}")
    
    # 总结
    logger.info("\n=== 测试总结 ===")
    all_passed = all([health_ok, qa_ok, ws_ok])
    logger.info(f"总体测试结果: {'全部通过' if all_passed else '存在失败'}")

if __name__ == "__main__":
    asyncio.run(main()) 