from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, generate_latest, Gauge
from prometheus_client import start_http_server
import time

app = FastAPI()

# Prometheus metrics
REQUEST_COUNTER = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_IN_PROGRESS = Gauge('http_requests_in_progress', 'Number of HTTP requests in progress', ['endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'Duration of HTTP requests in seconds', ['method', 'endpoint'])

@app.middleware("http")
async def add_prometheus_metrics(request: Request, call_next):
    start_time = time.time()
    endpoint = request.url.path
    method = request.method
    REQUEST_COUNTER.labels(method, endpoint).inc()
    REQUEST_IN_PROGRESS.labels(endpoint).inc()
    
    try:
        response = await call_next(request)
        return response
    finally:
        duration = time.time() - start_time
        REQUEST_DURATION.labels(method, endpoint).observe(duration)
        REQUEST_IN_PROGRESS.labels(endpoint).dec()

@app.get("/metrics")
def get_metrics():
    return Response(generate_latest(), media_type='text/plain')

@app.get("/")
async def read_root():
    await some_long_running_task()  # Simulating a long-running task
    return {"message": "Hello, World!"}

async def some_long_running_task():
    time.sleep(20)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
