FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pipeline/product/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pipeline/product/ ./product/
COPY pipeline/fonts/ ./fonts/
COPY pipeline/gokturk_labels_v1_locked.json ./gokturk_labels_v1_locked.json
COPY model/ ./model/

WORKDIR /app/product

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]