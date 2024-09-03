from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, generate_latest, Gauge
import time
import asyncio
from collections import defaultdict

app = FastAPI()

# Prometheus metrics
REQUEST_COUNTER = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_IN_PROGRESS = Gauge('http_requests_in_progress', 'Number of HTTP requests in progress', ['endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'Duration of HTTP requests in seconds', ['method', 'endpoint'])
REQUEST_QUEUE = Gauge('http_requests_queue', 'Number of HTTP requests in queue', ['endpoint'])
TOTAL_REQUESTS = Counter('total_requests', 'Total requests received', ['endpoint'])
CURRENT_PROCESSING = Gauge('current_processing', 'Currently processing requests', ['endpoint'])

class RequestQueue:
    def __init__(self):
        self.queues = defaultdict(asyncio.Queue)
        self.processing = defaultdict(int)

    async def add(self, endpoint):
        await self.queues[endpoint].put(1)
        REQUEST_QUEUE.labels(endpoint).inc()

    async def remove(self, endpoint):
        await self.queues[endpoint].get()
        REQUEST_QUEUE.labels(endpoint).dec()

    async def start_processing(self, endpoint):
        self.processing[endpoint] += 1
        CURRENT_PROCESSING.labels(endpoint).inc()

    async def end_processing(self, endpoint):
        self.processing[endpoint] -= 1
        CURRENT_PROCESSING.labels(endpoint).dec()

    def get_queue_size(self, endpoint):
        return self.queues[endpoint].qsize()

    def get_processing_count(self, endpoint):
        return self.processing[endpoint]

request_queue = RequestQueue()

@app.middleware("http")
async def add_prometheus_metrics(request: Request, call_next):
    start_time = time.time()
    endpoint = request.url.path
    method = request.method

    await request_queue.add(endpoint)
    
    try:
        await request_queue.start_processing(endpoint)
        response = await call_next(request)
        return response
    finally:
        duration = time.time() - start_time
        REQUEST_DURATION.labels(method, endpoint).observe(duration)
        await request_queue.end_processing(endpoint)
        await request_queue.remove(endpoint)

@app.get("/metrics")
def get_metrics():
    return Response(generate_latest(), media_type='text/plain')

@app.get("/testing/queue")
async def read_root():
    TOTAL_REQUESTS.labels("/testing/queue").inc()
    try:
        await some_long_running_task()  # Simulating a long-running task
        return {"message": "Hello, World!"}
    finally:
        pass

@app.get("/queue")
async def get_queue():
    return {
        "queue_status": {endpoint: request_queue.get_queue_size(endpoint) for endpoint in request_queue.queues},
        "total_queued": sum(request_queue.get_queue_size(endpoint) for endpoint in request_queue.queues),
        "in_progress": {endpoint: request_queue.get_processing_count(endpoint) for endpoint in request_queue.queues},
        "total_requests": {endpoint: TOTAL_REQUESTS.labels(endpoint)._value.get() for endpoint in request_queue.queues}
    }

async def some_long_running_task():
    await asyncio.sleep(5)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)