from openai import OpenAI
import asyncio
import os
import time

class StreamTester:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    async def test_stream(self):
        """测试流式输出，要求AI生成一段100字左右的文本"""
        prompt = """请生成一段大约100字的文本，描述人工智能的发展。
要求：
1. 内容要有条理
2. 每句话都要相对独立，便于观察流式输出效果
3. 语言要通俗易懂
"""
        
        response = self.client.chat.completions.create(
            model="qwen-long",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.7,
            stream=True
        )

        print("开始流式输出：")
        print("-" * 50)
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                chunk_content = chunk.choices[0].delta.content
                print(chunk_content, end="", flush=True)
                # 可以取消下面的注释来观察每个chunk的输出间隔
                # time.sleep(0.1)
        print("\n" + "-" * 50)

async def main():
    # 从环境变量获取API密钥
    api_key = os.getenv("DASH_SCOPE_API_KEY", "")
    if not api_key:
        raise ValueError("请设置环境变量 DASH_SCOPE_API_KEY")

    tester = StreamTester(api_key)
    await tester.test_stream()

if __name__ == "__main__":
    asyncio.run(main()) 