FROM python:3.11-slim

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Klasör yolu belirtmeden doğrudan kök dizindeki requirements.txt'yi al
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Kök dizindeki app.py, fonts/, model/ ve json dahil her şeyi kopyala
COPY . /app/

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]