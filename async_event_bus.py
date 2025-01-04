# async_event_bus.py

import asyncio
import logging

from typing import Callable, List
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class AsyncEventBus:
    def __init__(self):
        self.subscribers: List[Callable[[str], None]] = []
        self.queue = asyncio.Queue()
        self.dispatch_task = asyncio.create_task(self._dispatcher())

    def subscribe(self, callback: Callable[[str], None]):
        self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str], None]):
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    async def publish(self, message: str):
        await self.queue.put(message)

    async def _dispatcher(self):
        while True:
            message = await self.queue.get()
            for subscriber in self.subscribers:
                try:
                    if asyncio.iscoroutinefunction(subscriber):
                        asyncio.create_task(subscriber(message))
                    else:
                        subscriber(message)
                except Exception as e:
                    print(f"Error in subscriber: {e}")
            self.queue.task_done() 

async def async_subscriber(message: str):
    await asyncio.sleep(1)  # 模拟异步处理
    print(f"异步订阅者收到消息: {message}")

def sync_subscriber(message: str):
    print(f"同步订阅者收到消息: {message}")

async def main():
    # 创建事件总线实例
    event_bus = AsyncEventBus()
    
    # 注册订阅者
    event_bus.subscribe(async_subscriber)
    event_bus.subscribe(sync_subscriber)
    
    # 发布多条消息
    logger.info("开始发布消息...")
    await event_bus.publish("消息1")
    await event_bus.publish("消息2")
    
    # 等待所有消息处理完成
    await event_bus.queue.join()
    
    # 取消调度任务
    event_bus.dispatch_task.cancel()
    try:
        await event_bus.dispatch_task
    except asyncio.CancelledError:
        logger.info("事件总线调度任务已取消")


if __name__ == "__main__":
    asyncio.run(main()) 