# InSpec AI Frontend

This folder contains the Next.js frontend for InSpec AI.

The FastAPI backend lives outside this folder at:

```text
../backend/
```

## Run Frontend

From this `inspecai` folder:

```bash
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

## Run Backend

From the project root, one level above this folder:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.server:app --reload
```

The frontend expects the backend at:

```text
http://localhost:8000
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

Supported image types: JPG, JPEG, PNG, and WEBP.
