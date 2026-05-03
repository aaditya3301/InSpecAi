# InSpec AI

InSpec AI is a local deepfake and AI-image analysis tool with a Next.js frontend and a FastAPI inference backend.

## Project Structure

```text
InSpecAI/
  backend/     FastAPI + Transformers inference server
  inspecai/    Next.js frontend
```

## Prerequisites

- Node.js 20+
- Python 3.10+
- `pip`

## Backend Setup

From the project root:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt
```

Optional Hugging Face token:

```bash
copy backend\.env.example backend\.env
```

Then set `HF_TOKEN` inside `backend/.env` if required.

Optional Neon benchmark storage:

```text
DATABASE_URL=postgresql://user:password@host/database?sslmode=require
```

When `DATABASE_URL` is present, batch benchmark runs are saved automatically and can be reopened from the saved benchmark cards in the Batch view.

Run the backend:

```bash
uvicorn backend.server:app --reload
```

Backend URL:

```text
http://localhost:8000
```

## Frontend Setup

From the frontend folder:

```bash
cd inspecai
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:3000
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

Supported formats: JPG, JPEG, PNG, WEBP.

## Notes

- First model runs may be slow because weights download locally.
- CPU is used by default.
- To use CUDA device 0:

```bash
set INSPEC_DEVICE=0
uvicorn backend.server:app --reload
```
