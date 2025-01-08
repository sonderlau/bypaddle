# How-to-start

## 安装依赖

```
pip install -r requirements.txt
```

目录下创建一个文件.env，内容如下：

```
ADMIN_USERNAME=admin
ADMIN_PASSWORD=password123 
DASH_SCOPE_API_KEY=sk-c3b22834c96a4f368657ad8eafa1999f
```

# 启动web服务

```
python 13-web.py
```

```
INFO:     Will watch for changes in these directories: ['G:\\bypaddle']
INFO:     Uvicorn running on http://0.0.0.0:8011 (Press CTRL+C to quit)
INFO:     Started reloader process [7400] using WatchFiles
INFO:     Started server process [7084]
INFO:     Waiting for application startup.
2025-01-08 20:16:38,324 - INFO - 5 changes detected
2025-01-08 20:16:38,420 - INFO - 初始化向量处理器，使用模型: BAAI/bge-large-zh-v1.5
2025-01-08 20:16:41,516 - INFO - 使用设备: cpu
2025-01-08 20:16:41,587 - INFO - Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
2025-01-08 20:16:41,854 - INFO - 初始化检索组件...
2025-01-08 20:16:45,380 - INFO - Reranker使用设备: cpu
2025-01-08 20:16:45,411 - INFO - 系统初始化完成
INFO:     Application startup complete.
```

# 本地调试(测试浏览器版本127.0.6533.120)

浏览器打开 http://localhost:8011/ 或者 127.0.0.1:8011/


# 关于代码解释：

1. 13-web.py 是web服务的主文件，负责启动和停止服务，以及处理请求。
2. 对应的前端文件是templates/index.html
3. 其他几个主要的文件是： 
3.1. 11-llm-rag.py 是LLM RAG的实现，负责生成回答。async_rag.py 是异步RAG的实现，负责异步生成回答。
3.2. 12-llm-workflow.py 是LLM工作流的实现，负责生成工作流。
3.3. 10-2.rag-reranker.py 是RAG重排序的实现.这就是我们希望放在GPU 4090上运行的。我已经改了代码，它会检测是否有torch.cuda.is_available()，如果没有，则使用CPU。

# 日志显示的工作流程 （为了用户不用长期等待无聊）

当Python代码中调用logger.info()或logger.error()等方法时
自定义的WebSocketHandler捕获这些日志
处理器将日志消息格式化并通过WebSocket广播（这里比较低效率）
前端WebSocket接收到消息后，根据用户ID过滤
符合条件的日志消息会被添加到页面的日志区域

# 关于Qwen的流式返回

目前代码里面没有，这个修改比较容易，我会在后面添加。

