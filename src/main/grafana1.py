from fastapi import FastAPI, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

app = FastAPI()

# Define Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')

@app.get("/api/hello")
async def hello():
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        time.sleep(0.1)  # Simulate some processing time
    return {"message": "Hello, World!"}

@app.get("/api/status")
async def status():
    REQUEST_COUNT.inc()
    return {"status": "OK"}

@app.get("/metrics")
async def metrics():
    metrics_data = generate_latest()
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)