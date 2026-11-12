FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY run_inference.py model.py ./

ENTRYPOINT ["python", "/app/run_inference.py"]
