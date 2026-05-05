# InSpec AI

InSpec AI is a local forensic media analysis dashboard for detecting deepfake
and AI-generated content. It uses one FastAPI backend for image and video
inference, and one Next.js frontend for the user interface.

## Project Structure

```text
InSpecAI/
  backend/
    server.py
    requirements.txt
    video/
      predictor.py
      model/
      weight/
  inspecai/
    app/
    package.json
```

## Requirements

- Python 3.10+
- Node.js 20+
- npm
- pip

## Backend Setup

Open a terminal in the project root, then go to the backend folder:

```bat
cd backend
```

Create and activate the backend virtual environment:

```bat
python -m venv venv
venv\Scripts\activate
```

Install backend dependencies:

```bat
pip install -r requirements.txt
```

Run the backend:

```bat
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Backend URL:

```text
http://localhost:8000
```

Optional saved-history database:

```bat
copy .env.example .env
```

Then edit `backend\.env` and set a real `DATABASE_URL`. Without a database URL,
the app still runs inference, but saved image/video/benchmark history will be
empty.

## Frontend Setup

Open a second terminal in the project root, then go to the frontend folder:

```bat
cd inspecai
```

Install frontend dependencies:

```bat
npm install
```

Run the frontend:

```bat
npm run dev
```

Frontend URL:

```text
http://localhost:3000
```

## API Endpoints

```text
GET  /api/health
GET  /api/models
POST /api/predict
POST /api/video/predict
```

Saved-history endpoints:

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

## Dataset Format

Batch benchmark mode expects:

```text
dataset/
  real/
    image_001.jpg
  fake/
    image_001.jpg
```

Supported image formats:

```text
JPG, JPEG, PNG, WEBP
```

Supported video formats:

```text
MP4, AVI, MOV, MKV, WEBM, MPEG, MPG
```

## Notes

- Image models are loaded lazily on first use.
- The CViT video model is loaded lazily on first video analysis.
- First runs can be slow while model weights load or download.
