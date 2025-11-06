# Backend - API Super-Resolution

API REST FastAPI pour upscaler des images avec Real-ESRGAN.

## Stack Technique

- **Framework** : FastAPI
- **Model** : Real-ESRGAN (RRDBNet)
- **Storage** : Cloudflare R2 / AWS S3
- **Queue** : Redis + RQ (pour async processing)
- **Auth** : JWT (optionnel, pour version premium)

## Structure

```
backend/
├── app/
│   ├── main.py              # Point d'entrée FastAPI
│   ├── api/
│   │   ├── routes/
│   │   │   ├── upscale.py   # POST /upscale
│   │   │   └── health.py    # GET /health
│   │   └── dependencies.py   # Dépendances (DB, auth)
│   ├── core/
│   │   ├── config.py        # Configuration (env vars)
│   │   └── security.py      # JWT, API keys
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   ├── services/
│   │   ├── inference.py     # Logique Real-ESRGAN
│   │   └── storage.py       # Upload/download S3
│   └── utils/
│       └── image.py         # Preprocessing images
├── tests/
│   ├── test_api.py
│   └── test_inference.py
├── requirements.txt
└── Dockerfile
```

## Endpoints Prévus

### 1. Upscale Image
```http
POST /api/v1/upscale
Content-Type: multipart/form-data

{
  "file": <image>,
  "scale": 4,
  "model": "RealESRGAN_x4plus"
}

Response:
{
  "job_id": "uuid",
  "status": "processing",
  "estimated_time": 10
}
```

### 2. Get Result
```http
GET /api/v1/result/{job_id}

Response:
{
  "status": "completed",
  "output_url": "https://cdn.../result.png",
  "metrics": {
    "processing_time": 8.5,
    "original_size": [512, 512],
    "output_size": [2048, 2048]
  }
}
```

## TODO

- [ ] Setup FastAPI base
- [ ] Intégrer Real-ESRGAN inference
- [ ] Upload/download S3
- [ ] Queue système (Celery/RQ)
- [ ] Rate limiting
- [ ] Tests unitaires
- [ ] Dockerfile

## Démarrage (à venir)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```
