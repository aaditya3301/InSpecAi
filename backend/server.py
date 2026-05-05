"""
InSpec AI unified local inference server.

The frontend sends image and video analysis requests to this FastAPI app.
Image models are lazy-loaded Hugging Face pipelines; video inference is routed
to the local CViT module in backend/video and returns per-frame evidence.

Run from the backend folder:
    pip install -r requirements.txt
    uvicorn server:app --reload

Optional:
    INSPEC_DEVICE=0      Use CUDA device 0 for Hugging Face image models.
    HF_TOKEN=...         Used by transformers/HF Hub if a model needs auth.
    DATABASE_URL=...     Enables saved analyses and benchmark history.
"""

import asyncio
import io
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional

import psycopg
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("inspec")

Image.MAX_IMAGE_PIXELS = 30_000_000


BACKEND_DIR = Path(__file__).resolve().parent


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(BACKEND_DIR / ".env")


MODEL_IDS = [
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "prithivMLmods/deepfake-detector-model-v1",
    "prithivMLmods/Deepfake-Detect-Siglip2",
    "prithivMLmods/Deepfake-Detection-Exp-02-21",
    "Wvolf/ViT_Deepfake_Detection",
    "dima806/deepfake_vs_real_image_detection",
    "Hemg/Deepfake-Detection",
    "Organika/sdxl-detector",
    "Heem2/AI-vs-Real-Image-Detection",
    "umm-maybe/AI-image-detector",
]

ALLOWED_MODELS = set(MODEL_IDS)

MODEL_META = [
    {"id": MODEL_IDS[0], "name": "Deep-Fake Detector v2", "author": "prithivMLmods"},
    {"id": MODEL_IDS[1], "name": "SigLIP Deepfake v1", "author": "prithivMLmods"},
    {"id": MODEL_IDS[2], "name": "SigLIP2 Deepfake", "author": "prithivMLmods"},
    {"id": MODEL_IDS[3], "name": "Deepfake Exp 02-21", "author": "prithivMLmods"},
    {"id": MODEL_IDS[4], "name": "ViT Deepfake Detector", "author": "Wvolf"},
    {"id": MODEL_IDS[5], "name": "Deepfake vs Real", "author": "dima806"},
    {"id": MODEL_IDS[6], "name": "Deepfake Image Detect", "author": "Hemg"},
    {"id": MODEL_IDS[7], "name": "SDXL Detector", "author": "Organika"},
    {"id": MODEL_IDS[8], "name": "AI vs Real Image", "author": "Heem2"},
    {"id": MODEL_IDS[9], "name": "AI Image Detector", "author": "umm-maybe"},
]

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}

DEVICE = int(os.getenv("INSPEC_DEVICE", "-1"))
DATABASE_URL = os.getenv("DATABASE_URL")

_pipelines: Dict[str, object] = {}
_load_locks: Dict[str, asyncio.Lock] = {}
_pipeline_build_lock = threading.Lock()
_db_ready = False


class BenchmarkModelResult(BaseModel):
    model_id: str
    model_name: str
    author: str = ""
    tag: str = ""
    total: int = 0
    correct: int = 0
    accuracy: float = 0
    avg_confidence: float = 0
    errors: int = 0
    rank: int


class BenchmarkCreate(BaseModel):
    dataset_name: str = "dataset"
    total_images: int = 0
    real_count: int = 0
    fake_count: int = 0
    skipped_count: int = 0
    elapsed_ms: int = 0
    results: List[BenchmarkModelResult] = Field(default_factory=list)


class SingleImageModelResult(BaseModel):
    model_id: str
    model_name: str
    author: str = ""
    tag: str = ""
    verdict: str
    confidence: float = 0
    raw_label: str = ""


class SingleImageCreate(BaseModel):
    file_name: str
    file_size: int = 0
    mime_type: str = "image/jpeg"
    image_data_url: str
    results: List[SingleImageModelResult] = Field(default_factory=list)


class VideoFrameResult(BaseModel):
    index: int
    frame_number: int = 0
    timestamp_ms: int = 0
    prediction: str
    confidence: float = 0
    face_count: int = 0
    thumbnail: str = ""


class VideoAnalysisCreate(BaseModel):
    file_name: str
    file_size: int = 0
    mime_type: str = "video/mp4"
    prediction: str
    confidence: float = 0
    frames: List[VideoFrameResult] = Field(default_factory=list)


def _build_pipeline(model_id: str):
    """Synchronous pipeline creation. Run from a worker thread.

    Transformers performs lazy imports and cache setup inside pipeline().
    Those paths are not reliably thread-safe during many first-time model
    loads, so cold pipeline construction is serialized here. Warm inference
    still runs normally after the pipeline is cached.
    """
    with _pipeline_build_lock:
        from transformers import pipeline

        log.info("Loading pipeline: %s on device %s", model_id, DEVICE)
        kwargs = {
            "task": "image-classification",
            "model": model_id,
            "device": DEVICE,
            "top_k": None,
        }

        token = os.getenv("HF_TOKEN")
        if token:
            kwargs["token"] = token

        pipe = pipeline(**kwargs)
        log.info("Loaded:           %s", model_id)
        return pipe


async def get_pipeline(model_id: str):
    """Return a cached pipeline, loading it on first use."""
    if model_id in _pipelines:
        return _pipelines[model_id]

    lock = _load_locks.setdefault(model_id, asyncio.Lock())
    async with lock:
        if model_id in _pipelines:
            return _pipelines[model_id]
        pipe = await asyncio.to_thread(_build_pipeline, model_id)
        _pipelines[model_id] = pipe
        return pipe


def _predict_video_sync(video_path: str) -> dict:
    try:
        from backend.video.predictor import predict_single
    except ModuleNotFoundError:
        from video.predictor import predict_single

    return predict_single(video_path)


def _db_connect():
    if not DATABASE_URL:
        raise HTTPException(503, "DATABASE_URL is not configured.")
    return psycopg.connect(DATABASE_URL)


def _init_db_sync() -> None:
    global _db_ready
    if _db_ready or not DATABASE_URL:
        return

    with _db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id BIGSERIAL PRIMARY KEY,
                    dataset_name TEXT NOT NULL,
                    total_images INTEGER NOT NULL DEFAULT 0,
                    real_count INTEGER NOT NULL DEFAULT 0,
                    fake_count INTEGER NOT NULL DEFAULT 0,
                    skipped_count INTEGER NOT NULL DEFAULT 0,
                    elapsed_ms INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_model_results (
                    id BIGSERIAL PRIMARY KEY,
                    run_id BIGINT NOT NULL REFERENCES benchmark_runs(id) ON DELETE CASCADE,
                    model_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    author TEXT NOT NULL DEFAULT '',
                    tag TEXT NOT NULL DEFAULT '',
                    total INTEGER NOT NULL DEFAULT 0,
                    correct INTEGER NOT NULL DEFAULT 0,
                    accuracy DOUBLE PRECISION NOT NULL DEFAULT 0,
                    avg_confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
                    errors INTEGER NOT NULL DEFAULT 0,
                    rank INTEGER NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_benchmark_model_results_run_id ON benchmark_model_results(run_id)"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS single_image_runs (
                    id BIGSERIAL PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    mime_type TEXT NOT NULL DEFAULT 'image/jpeg',
                    image_data_url TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS single_image_model_results (
                    id BIGSERIAL PRIMARY KEY,
                    run_id BIGINT NOT NULL REFERENCES single_image_runs(id) ON DELETE CASCADE,
                    model_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    author TEXT NOT NULL DEFAULT '',
                    tag TEXT NOT NULL DEFAULT '',
                    verdict TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
                    raw_label TEXT NOT NULL DEFAULT '',
                    sort_order INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_single_image_model_results_run_id ON single_image_model_results(run_id)"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS video_runs (
                    id BIGSERIAL PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    mime_type TEXT NOT NULL DEFAULT 'video/mp4',
                    prediction TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
                    frame_count INTEGER NOT NULL DEFAULT 0,
                    real_count INTEGER NOT NULL DEFAULT 0,
                    fake_count INTEGER NOT NULL DEFAULT 0,
                    no_face_count INTEGER NOT NULL DEFAULT 0,
                    thumbnail_data_url TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS video_frame_results (
                    id BIGSERIAL PRIMARY KEY,
                    run_id BIGINT NOT NULL REFERENCES video_runs(id) ON DELETE CASCADE,
                    frame_index INTEGER NOT NULL,
                    frame_number INTEGER NOT NULL DEFAULT 0,
                    timestamp_ms INTEGER NOT NULL DEFAULT 0,
                    prediction TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
                    face_count INTEGER NOT NULL DEFAULT 0,
                    thumbnail_data_url TEXT NOT NULL DEFAULT ''
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_video_frame_results_run_id ON video_frame_results(run_id)"
            )
        conn.commit()
    _db_ready = True


async def init_db() -> None:
    await asyncio.to_thread(_init_db_sync)


def _save_benchmark_sync(payload: BenchmarkCreate) -> dict:
    _init_db_sync()
    with _db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO benchmark_runs (
                    dataset_name, total_images, real_count, fake_count, skipped_count, elapsed_ms
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, created_at
                """,
                (
                    payload.dataset_name,
                    payload.total_images,
                    payload.real_count,
                    payload.fake_count,
                    payload.skipped_count,
                    payload.elapsed_ms,
                ),
            )
            run_id, created_at = cur.fetchone()

            for result in payload.results:
                cur.execute(
                    """
                    INSERT INTO benchmark_model_results (
                        run_id, model_id, model_name, author, tag, total, correct,
                        accuracy, avg_confidence, errors, rank
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        result.model_id,
                        result.model_name,
                        result.author,
                        result.tag,
                        result.total,
                        result.correct,
                        result.accuracy,
                        result.avg_confidence,
                        result.errors,
                        result.rank,
                    ),
                )

        conn.commit()

    return {"id": run_id, "created_at": created_at.isoformat()}


def _list_benchmarks_sync() -> List[dict]:
    if not DATABASE_URL:
        return []
    _init_db_sync()
    with _db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    r.id,
                    r.dataset_name,
                    r.total_images,
                    r.real_count,
                    r.fake_count,
                    r.skipped_count,
                    r.elapsed_ms,
                    r.created_at,
                    m.model_name,
                    m.accuracy,
                    m.avg_confidence
                FROM benchmark_runs r
                LEFT JOIN LATERAL (
                    SELECT model_name, accuracy, avg_confidence
                    FROM benchmark_model_results
                    WHERE run_id = r.id
                    ORDER BY rank ASC
                    LIMIT 1
                ) m ON TRUE
                ORDER BY r.created_at DESC
                LIMIT 20
                """
            )
            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "dataset_name": row[1],
            "total_images": row[2],
            "real_count": row[3],
            "fake_count": row[4],
            "skipped_count": row[5],
            "elapsed_ms": row[6],
            "created_at": row[7].isoformat(),
            "best_model": row[8],
            "best_accuracy": row[9],
            "best_confidence": row[10],
        }
        for row in rows
    ]


def _get_benchmark_sync(run_id: int) -> Optional[dict]:
    _init_db_sync()
    with _db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, dataset_name, total_images, real_count, fake_count,
                       skipped_count, elapsed_ms, created_at
                FROM benchmark_runs
                WHERE id = %s
                """,
                (run_id,),
            )
            run = cur.fetchone()
            if not run:
                return None

            cur.execute(
                """
                SELECT model_id, model_name, author, tag, total, correct,
                       accuracy, avg_confidence, errors, rank
                FROM benchmark_model_results
                WHERE run_id = %s
                ORDER BY rank ASC
                """,
                (run_id,),
            )
            rows = cur.fetchall()

    return {
        "id": run[0],
        "dataset_name": run[1],
        "total_images": run[2],
        "real_count": run[3],
        "fake_count": run[4],
        "skipped_count": run[5],
        "elapsed_ms": run[6],
        "created_at": run[7].isoformat(),
        "results": [
            {
                "model_id": row[0],
                "model_name": row[1],
                "author": row[2],
                "tag": row[3],
                "total": row[4],
                "correct": row[5],
                "accuracy": row[6],
                "avg_confidence": row[7],
                "errors": row[8],
                "rank": row[9],
            }
            for row in rows
        ],
    }


def _save_single_image_sync(payload: SingleImageCreate) -> dict:
    _init_db_sync()
    with _db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO single_image_runs (
                    file_name, file_size, mime_type, image_data_url
                )
                VALUES (%s, %s, %s, %s)
                RETURNING id, created_at
                """,
                (
                    payload.file_name,
                    payload.file_size,
                    payload.mime_type,
                    payload.image_data_url,
                ),
            )
            run_id, created_at = cur.fetchone()

            for index, result in enumerate(payload.results):
                cur.execute(
                    """
                    INSERT INTO single_image_model_results (
                        run_id, model_id, model_name, author, tag, verdict,
                        confidence, raw_label, sort_order
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        result.model_id,
                        result.model_name,
                        result.author,
                        result.tag,
                        result.verdict,
                        result.confidence,
                        result.raw_label,
                        index,
                    ),
                )
        conn.commit()

    return {"id": run_id, "created_at": created_at.isoformat()}


def _list_single_images_sync() -> List[dict]:
    if not DATABASE_URL:
        return []
    _init_db_sync()
    with _db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    r.id,
                    r.file_name,
                    r.file_size,
                    r.mime_type,
                    r.image_data_url,
                    r.created_at,
                    COUNT(m.id) AS model_count
                FROM single_image_runs r
                LEFT JOIN single_image_model_results m ON m.run_id = r.id
                GROUP BY r.id
                ORDER BY r.created_at DESC
                LIMIT 24
                """
            )
            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "file_name": row[1],
            "file_size": row[2],
            "mime_type": row[3],
            "image_data_url": row[4],
            "created_at": row[5].isoformat(),
            "model_count": row[6],
        }
        for row in rows
    ]


def _get_single_image_sync(run_id: int) -> Optional[dict]:
    _init_db_sync()
    with _db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, file_name, file_size, mime_type, image_data_url, created_at
                FROM single_image_runs
                WHERE id = %s
                """,
                (run_id,),
            )
            run = cur.fetchone()
            if not run:
                return None

            cur.execute(
                """
                SELECT model_id, model_name, author, tag, verdict, confidence, raw_label
                FROM single_image_model_results
                WHERE run_id = %s
                ORDER BY sort_order ASC
                """,
                (run_id,),
            )
            rows = cur.fetchall()

    return {
        "id": run[0],
        "file_name": run[1],
        "file_size": run[2],
        "mime_type": run[3],
        "image_data_url": run[4],
        "created_at": run[5].isoformat(),
        "results": [
            {
                "model_id": row[0],
                "model_name": row[1],
                "author": row[2],
                "tag": row[3],
                "verdict": row[4],
                "confidence": row[5],
                "raw_label": row[6],
            }
            for row in rows
        ],
    }


def _save_video_sync(payload: VideoAnalysisCreate) -> dict:
    _init_db_sync()
    real_count = sum(1 for frame in payload.frames if frame.prediction == "REAL")
    fake_count = sum(1 for frame in payload.frames if frame.prediction == "FAKE")
    no_face_count = sum(1 for frame in payload.frames if frame.prediction == "NO_FACE")
    thumbnail = next((frame.thumbnail for frame in payload.frames if frame.thumbnail), "")

    with _db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO video_runs (
                    file_name, file_size, mime_type, prediction, confidence, frame_count,
                    real_count, fake_count, no_face_count, thumbnail_data_url
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, created_at
                """,
                (
                    payload.file_name,
                    payload.file_size,
                    payload.mime_type,
                    payload.prediction,
                    payload.confidence,
                    len(payload.frames),
                    real_count,
                    fake_count,
                    no_face_count,
                    thumbnail,
                ),
            )
            run_id, created_at = cur.fetchone()

            for frame in payload.frames:
                cur.execute(
                    """
                    INSERT INTO video_frame_results (
                        run_id, frame_index, frame_number, timestamp_ms, prediction,
                        confidence, face_count, thumbnail_data_url
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        frame.index,
                        frame.frame_number,
                        frame.timestamp_ms,
                        frame.prediction,
                        frame.confidence,
                        frame.face_count,
                        frame.thumbnail,
                    ),
                )

        conn.commit()

    return {"id": run_id, "created_at": created_at.isoformat()}


def _list_videos_sync() -> List[dict]:
    if not DATABASE_URL:
        return []
    _init_db_sync()
    with _db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, file_name, file_size, mime_type, prediction, confidence,
                       frame_count, real_count, fake_count, no_face_count,
                       thumbnail_data_url, created_at
                FROM video_runs
                ORDER BY created_at DESC
                LIMIT 24
                """
            )
            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "file_name": row[1],
            "file_size": row[2],
            "mime_type": row[3],
            "prediction": row[4],
            "confidence": row[5],
            "frame_count": row[6],
            "real_count": row[7],
            "fake_count": row[8],
            "no_face_count": row[9],
            "thumbnail_data_url": row[10],
            "created_at": row[11].isoformat(),
        }
        for row in rows
    ]


def _get_video_sync(run_id: int) -> Optional[dict]:
    _init_db_sync()
    with _db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, file_name, file_size, mime_type, prediction, confidence,
                       frame_count, real_count, fake_count, no_face_count,
                       thumbnail_data_url, created_at
                FROM video_runs
                WHERE id = %s
                """,
                (run_id,),
            )
            run = cur.fetchone()
            if not run:
                return None

            cur.execute(
                """
                SELECT frame_index, frame_number, timestamp_ms, prediction,
                       confidence, face_count, thumbnail_data_url
                FROM video_frame_results
                WHERE run_id = %s
                ORDER BY frame_index ASC
                """,
                (run_id,),
            )
            rows = cur.fetchall()

    return {
        "id": run[0],
        "file_name": run[1],
        "file_size": run[2],
        "mime_type": run[3],
        "prediction": run[4],
        "confidence": run[5],
        "frame_count": run[6],
        "real_count": run[7],
        "fake_count": run[8],
        "no_face_count": run[9],
        "thumbnail_data_url": run[10],
        "created_at": run[11].isoformat(),
        "frames": [
            {
                "index": row[0],
                "frame_number": row[1],
                "timestamp_ms": row[2],
                "prediction": row[3],
                "confidence": row[4],
                "face_count": row[5],
                "thumbnail": row[6],
            }
            for row in rows
        ],
    }


app = FastAPI(title="InSpec Local Inference Server", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.on_event("startup")
async def startup():
    if DATABASE_URL:
        try:
            await init_db()
            log.info("Benchmark database ready.")
        except Exception:
            log.exception("Benchmark database initialization failed.")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "models": len(MODEL_IDS),
        "loaded": [model_id for model_id in MODEL_IDS if model_id in _pipelines],
        "loaded_count": len(_pipelines),
        "device": DEVICE,
        "hf_token_configured": bool(os.getenv("HF_TOKEN")),
        "database_configured": bool(DATABASE_URL),
    }


@app.get("/api/models")
async def models():
    return {
        "models": MODEL_META,
        "loaded": [model_id for model_id in MODEL_IDS if model_id in _pipelines],
    }


@app.post("/api/benchmarks")
async def create_benchmark(payload: BenchmarkCreate):
    try:
        return await asyncio.to_thread(_save_benchmark_sync, payload)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to save benchmark")
        raise HTTPException(500, f"Could not save benchmark. {type(e).__name__}: {str(e)[:200]}")


@app.get("/api/benchmarks")
async def list_benchmarks():
    try:
        return {"benchmarks": await asyncio.to_thread(_list_benchmarks_sync)}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to list benchmarks")
        raise HTTPException(500, f"Could not list benchmarks. {type(e).__name__}: {str(e)[:200]}")


@app.get("/api/benchmarks/{run_id}")
async def get_benchmark(run_id: int):
    try:
        benchmark = await asyncio.to_thread(_get_benchmark_sync, run_id)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to load benchmark")
        raise HTTPException(500, f"Could not load benchmark. {type(e).__name__}: {str(e)[:200]}")

    if not benchmark:
        raise HTTPException(404, "Benchmark not found.")
    return benchmark


@app.post("/api/single-images")
async def create_single_image(payload: SingleImageCreate):
    try:
        return await asyncio.to_thread(_save_single_image_sync, payload)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to save single image analysis")
        raise HTTPException(500, f"Could not save image analysis. {type(e).__name__}: {str(e)[:200]}")


@app.get("/api/single-images")
async def list_single_images():
    try:
        return {"images": await asyncio.to_thread(_list_single_images_sync)}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to list single image analyses")
        raise HTTPException(500, f"Could not list image analyses. {type(e).__name__}: {str(e)[:200]}")


@app.get("/api/single-images/{run_id}")
async def get_single_image(run_id: int):
    try:
        image = await asyncio.to_thread(_get_single_image_sync, run_id)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to load single image analysis")
        raise HTTPException(500, f"Could not load image analysis. {type(e).__name__}: {str(e)[:200]}")

    if not image:
        raise HTTPException(404, "Image analysis not found.")
    return image


@app.post("/api/videos")
async def create_video(payload: VideoAnalysisCreate):
    try:
        return await asyncio.to_thread(_save_video_sync, payload)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to save video analysis")
        raise HTTPException(500, f"Could not save video analysis. {type(e).__name__}: {str(e)[:200]}")


@app.get("/api/videos")
async def list_videos():
    try:
        return {"videos": await asyncio.to_thread(_list_videos_sync)}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to list video analyses")
        raise HTTPException(500, f"Could not list video analyses. {type(e).__name__}: {str(e)[:200]}")


@app.get("/api/videos/{run_id}")
async def get_video(run_id: int):
    try:
        video = await asyncio.to_thread(_get_video_sync, run_id)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to load video analysis")
        raise HTTPException(500, f"Could not load video analysis. {type(e).__name__}: {str(e)[:200]}")

    if not video:
        raise HTTPException(404, "Video analysis not found.")
    return video


@app.post("/api/video/predict")
async def predict_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(400, "File must be a supported video format.")

    if file.content_type and not file.content_type.startswith("video/") and file.content_type != "application/octet-stream":
        raise HTTPException(400, "File must be a video.")

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp_path = temp.name
            while chunk := await file.read(1024 * 1024):
                temp.write(chunk)

        return await asyncio.to_thread(_predict_video_sync, temp_path)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Video inference failed")
        raise HTTPException(500, f"Video inference failed. {type(e).__name__}: {str(e)[:200]}")
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


@app.post("/api/predict")
async def predict(
    model_id: str = Form(...),
    file: UploadFile = File(...),
):
    if model_id not in ALLOWED_MODELS:
        raise HTTPException(400, f"Model '{model_id}' is not in the allow-list.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")

    try:
        pipe = await get_pipeline(model_id)
    except Exception as e:
        log.exception("Failed to load %s", model_id)
        msg = str(e)
        if "Repository Not Found" in msg or "not a valid model identifier" in msg:
            raise HTTPException(502, f"Model is unavailable on Hugging Face: {model_id}")
        raise HTTPException(500, f"Could not load model. {type(e).__name__}: {msg[:200]}")

    try:
        result: List[dict] = await asyncio.to_thread(pipe, img)
    except Exception as e:
        log.exception("Inference failed for %s", model_id)
        raise HTTPException(500, f"Inference failed. {type(e).__name__}: {str(e)[:200]}")

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
