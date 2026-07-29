# Göktürkçe Doğrulama Aracı — Public API

Göktürkçe/Orhun alfabesinin 38 sınıflı CNN sınıflandırması, kelime segmentasyonu, ünlü
uyumu kontrolü, Latin→Göktürkçe dönüşümü ve görsel/video render için FastAPI tabanlı
web servisi.

> **Not:** Bu repo sadece **inference** için gereken dosyaları içerir. Eğitim
> pipeline'ı (`pipeline/01_*`, `02_*`, `03_*`, `04_*`, `06_*`, `data/`,
> `checkpoints/`) ve `run_night_pipeline.py` bu repo'da yer almaz.

## Canlı Demo

- (Render.com'a deploy edildikten sonra URL'i buraya ekleyin)

## Mimari

```
Browser ──HTTP──> FastAPI (app.py) ──> MobileNetV2 (PyTorch CPU)
                  │                  └─> rules_engine (ünlü uyumu)
                  │                  └─> render (font/texture)
                  └─ PostgreSQL (kullanıcı, kredi, ilgi kayıtları)
```

## Lokal Geliştirme

### 1. Gereksinimler

- Python 3.11
- PostgreSQL 14+ (lokalde veya Docker)

### 2. Kurulum

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Ortam değişkenleri

```bash
cp .env.example .env
# .env içindeki DATABASE_URL'i kendi local postgres'inize göre düzenleyin
# SECRET_KEY ve API_KEY_SECRET'i openssl rand -hex 32 ile doldurun
```

### 4. Veritabanı tabloları

`app.py` ilk çalıştığında SQLAlchemy `Base.metadata.create_all` ile tabloları
otomatik oluşturur. Ek bir migration adımı gerekmez (production'da alembic
önerilir).

### 5. Çalıştır

```bash
cd pipeline/product
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Tarayıcıda: http://localhost:8000

## Render.com'a Deploy

### Yöntem A — render.yaml ile (önerilen)

1. Bu repo'yu GitHub'a push'layın.
2. Render.com → **New** → **Blueprint** → GitHub reponuzu seçin.
3. Render `render.yaml`'ı okuyup Web Service + Postgres'i otomatik oluşturur.
4. **Environment Variables** sekmesinde `SECRET_KEY` ve `API_KEY_SECRET`
   otomatik üretilir; `ALLOWED_ORIGINS` değerini kendi domain'inize güncelleyin.
5. İlk build ~10-15 dakika sürer (torch CPU binary büyük).

### Yöntem B — Manuel

1. Render.com → **New** → **Web Service** → GitHub reponuzu bağlayın.
2. **Runtime**: Python
3. **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt && git lfs pull`
4. **Start Command** (Procfile yerine bu da çalışır):
   ```
   bash -c "export PYTHONPATH=\"${PYTHONPATH}:$RENDER_PROJECT_DIR/pipeline/product:$RENDER_PROJECT_DIR/pipeline\" && exec gunicorn app:app --chdir pipeline/product --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120"
   ```
5. **Environment**:
   - `PYTHON_VERSION=3.11.11`
   - `PYTHONUNBUFFERED=1`
   - `SECRET_KEY=<openssl rand -hex 32>`
   - `API_KEY_SECRET=<openssl rand -hex 32>`
   - `DATABASE_URL=<Render Postgres internal connection string>`
   - `ALLOWED_ORIGINS=https://<your-app>.onrender.com`
6. **Health Check Path**: `/api/model_info`
7. **Plan**: Starter ($7/ay) veya Free (soğuk başlatma yavaş)

### Git LFS Notu

Model dosyaları (`*.pt`) ve fontlar (`*.ttf`) **Git LFS** ile takip edilir. İlk
push'tan önce lokalde:

```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.ttf"
git add .gitattributes
git add model/checkpoints/best.pt pipeline/fonts/*.ttf
git commit -m "Track model and fonts with LFS"
```

Render build pipeline'ı GitHub entegrasyonu üzerinden LFS objelerini otomatik
çeker. `git lfs pull` adımı buildCommand'da yedek olarak çalışır.

## API Endpoints (özet)

| Method | Path                  | Açıklama                                | Auth |
| ------ | --------------------- | --------------------------------------- | ---- |
| POST   | `/register`           | Yeni kullanıcı (davet kodu opsiyonel)   | —    |
| POST   | `/login`              | E-posta/parola → API anahtarı            | —    |
| GET    | `/me`                 | Kullanıcı bilgisi + kredi               | API  |
| POST   | `/interest`           | Paket ilgisi kaydet                     | —    |
| POST   | `/predict`            | Tek glyph tahmin                        | API  |
| POST   | `/predict_word`       | Çoklu glyph → kelime tahmini            | API  |
| POST   | `/predict_image`      | Görsel → otomatik segmentasyon + tahmin | API  |
| POST   | `/translate`          | Latin → Göktürkçe class_id dizisi       | API  |
| POST   | `/decode_text`        | Göktürkçe Unicode metin → okunuş        | API  |
| POST   | `/api/render`         | Metin → PNG (stil + texture)            | API  |
| POST   | `/api/render_video`   | Metin → MP4 (parallax/zoom/pan/fade)    | API  |
| POST   | `/api/composite`      | Metin → baz görsel üzerine kompozit     | API  |
| GET    | `/api/model_info`     | Model meta (sürüm, doğruluk eşiği)      | —    |
| GET    | `/api/config`         | FE için feature flag'ler                 | —    |
| GET    | `/terms` `/privacy` `/kvkk` | Yasal metinler                    | —    |

Tüm API anahtarı gerektiren endpoint'ler `X-API-Key` header'ı kullanır. Rate
limit: dakikada 30, günde 1000 istek (SlowAPI).

## Güvenlik

- **Veritabanı**: Render Postgres (kalıcı). Lokalde `.env`'de `DATABASE_URL` ayarlayın.
- **Secret yönetimi**: `SECRET_KEY`, `API_KEY_SECRET` env'den. `.env` repoya
  eklenmez (`.gitignore`).
- **Parola hash**: PBKDF2-SHA256, 100.000 iterasyon, 16-byte salt.
- **Rate limiting**: SlowAPI; `register`/`login` dahil tüm endpoint'lerde.
- **CORS**: Beyaz liste. `ALLOWED_ORIGINS` env ile kontrol.
- **Upload limit**: 10 MB (`.env` ile ayarlanabilir).
- **XSS**: Frontend'de `textContent` tercih edilir, `innerHTML` kullanımı sadece
  güvenilir kaynak için.

## Lisans

(TBD — proje sahibi tarafından belirlenecek)
