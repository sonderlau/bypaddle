import queue
import threading
from typing import Callable

class EventBus:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EventBus, cls).__new__(cls)
                cls._instance.subscribers = []
                cls._instance.message_queue = queue.Queue()
                cls._instance._start_dispatcher()
            return cls._instance
    
    def _start_dispatcher(self):
        def dispatcher():
            while True:
                try:
                    message = self.message_queue.get()
                    for subscriber in self.subscribers:
                        try:
                            subscriber(message)
                        except Exception as e:
                            print(f"Error in subscriber: {e}")
                except Exception as e:
                    print(f"Error in dispatcher: {e}")
                    
        thread = threading.Thread(target=dispatcher, daemon=True)
        thread.start()
    
    def publish(self, message: str):
        self.message_queue.put(message)
    
    def subscribe(self, callback: Callable[[str], None]):
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[str], None]):
        if callback in self.subscribers:
            self.subscribers.remove(callback)
