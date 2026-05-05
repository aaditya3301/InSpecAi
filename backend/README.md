# InSpec AI Backend

Unified FastAPI backend for image analysis, video analysis, batch benchmarks,
and saved image-analysis history.

## Setup

From the project root:

```bat
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

From the `backend` folder with the virtual environment activated:

```bat
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

API base URL:

```text
http://localhost:8000
```

## Saved History

Saved image, video, and benchmark history requires a database URL. To configure
one locally:

```bat
copy .env.example .env
```

Then edit `.env` and set:

```text
DATABASE_URL=postgresql://user:password@host/database?sslmode=require
```

Without `DATABASE_URL`, inference still works and history endpoints return empty
lists.

## Layout

```text
backend/
  server.py
  requirements.txt
  video/
    predictor.py
    model/
      cvit.py
      pred_func.py
    weight/
      cvit2_deepfake_detection_ep_50.pth
      deepdeepfake_cvit_gpu_ep50.pkl
```

## Endpoints

```text
GET  /api/health
GET  /api/models
POST /api/predict
POST /api/video/predict
```

Persistence endpoints are available when `DATABASE_URL` is configured in the
terminal environment:

```text
GET  /api/single-images
POST /api/single-images
GET  /api/single-images/{id}

GET  /api/videos
POST /api/videos
GET  /api/videos/{id}

GET  /api/benchmarks
POST /api/benchmarks
GET  /api/benchmarks/{id}
```

## Optional Environment Variables

Set these before starting the backend if needed:

```bat
set HF_TOKEN=hf_your_token_here
set DATABASE_URL=postgresql://user:password@host/database?sslmode=require
set INSPEC_DEVICE=0
```

- `HF_TOKEN` is used for private or gated Hugging Face models.
- `DATABASE_URL` enables saved results and benchmark history.
- `INSPEC_DEVICE=0` runs Hugging Face image models on CUDA device 0.
