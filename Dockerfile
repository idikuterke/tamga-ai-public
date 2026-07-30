FROM python:3.11-slim

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Klasör yolu belirtmeden doğrudan kök dizindeki requirements.txt'yi al
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kök dizindeki app.py, fonts/, model/ ve json dahil her şeyi kopyala
COPY . /app/

EXPOSE 8000

# app.py, repo kökünde değil pipeline/product/ altında yaşıyor; kendi
# içindeki tüm dosya yolları (model/, schema, fonts) __file__ tabanlı
# çözülüyor (bkz. PRODUCT_DIR = Path(__file__).resolve().parent), cwd'ye
# bağlı değil. Bu yüzden WORKDIR'ı değiştirmeye gerek yok — modülü
# bulması için uvicorn'a sadece --app-dir ile doğru klasörü göstermek
# yeterli. Render $PORT'u runtime'da enjekte ediyor; shell formu ile
# ${PORT:-8000} olarak genişletiliyor (yoksa lokal varsayılan 8000).
CMD uvicorn app:app --app-dir pipeline/product --host 0.0.0.0 --port ${PORT:-8000}