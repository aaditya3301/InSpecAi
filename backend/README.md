# InSpec AI Backend

Local FastAPI inference server for the InSpec AI Next.js frontend.

## Setup

From the project root:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt
```

## Run

From the project root:

```bash
uvicorn backend.server:app --reload
```

The API runs at:

```text
http://localhost:8000
```

## Optional Hugging Face Token

Create `backend/.env`:

```text
HF_TOKEN=hf_your_token_here
```

## Optional Neon Database

Add a Neon/Postgres URL to `backend/.env`:

```text
DATABASE_URL=postgresql://user:password@host/database?sslmode=require
```

The backend creates the benchmark tables automatically on startup. Batch benchmark results are saved after each completed run.

## Optional GPU Mode

```bash
set INSPEC_DEVICE=0
uvicorn backend.server:app --reload
```
