"use client";

import {
  ChangeEvent,
  DragEvent,
  MouseEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

const API_BASE = "http://localhost:8000";

type Verdict = "REAL" | "FAKE";
type Mode = "single" | "batch";
type StatusKind = "" | "ok" | "err" | "warn";
type ModelRunStatus = "IDLE" | "RUNNING" | "REAL" | "FAKE" | "ERROR";

type ModelMeta = {
  id: string;
  name: string;
  author: string;
  arch: string;
  params: string;
  year: string;
  tag: string;
  fakeLabels: string[];
};

type RawPrediction = {
  label: string;
  score: number;
};

type InterpretedResult = {
  verdict: Verdict;
  confidence: number;
  rawLabel: string;
};

type SingleModelState = {
  status: ModelRunStatus;
  result?: InterpretedResult;
  error?: string;
};

type DatasetFile = File & {
  webkitRelativePath?: string;
};

type DatasetImage = {
  file: DatasetFile;
  expected: Verdict;
  path: string;
};

type BatchStat = {
  model: ModelMeta;
  total: number;
  correct: number;
  confidenceSum: number;
  errors: number;
  running: boolean;
  done: boolean;
  lastError: string;
};

type SavedBenchmarkSummary = {
  id: number;
  dataset_name: string;
  total_images: number;
  real_count: number;
  fake_count: number;
  skipped_count: number;
  elapsed_ms: number;
  created_at: string;
  best_model: string | null;
  best_accuracy: number | null;
  best_confidence: number | null;
};

type SavedBenchmarkResult = {
  model_id: string;
  model_name: string;
  author: string;
  tag: string;
  total: number;
  correct: number;
  accuracy: number;
  avg_confidence: number;
  errors: number;
  rank: number;
};

type SavedBenchmarkDetail = SavedBenchmarkSummary & {
  results: SavedBenchmarkResult[];
};

type SavedSingleImageResult = {
  model_id: string;
  model_name: string;
  author: string;
  tag: string;
  verdict: Verdict;
  confidence: number;
  raw_label: string;
};

type SavedSingleImageSummary = {
  id: number;
  file_name: string;
  file_size: number;
  mime_type: string;
  image_data_url: string;
  created_at: string;
  model_count: number;
};

type SavedSingleImageDetail = SavedSingleImageSummary & {
  results: SavedSingleImageResult[];
};

const MODELS: ModelMeta[] = [
  {
    id: "prithivMLmods/Deep-Fake-Detector-v2-Model",
    name: "Deep-Fake Detector v2",
    author: "prithivMLmods",
    arch: "ViT-Base",
    params: "86M",
    year: "2025",
    tag: "FACES",
    fakeLabels: ["deepfake", "fake"],
  },
  {
    id: "prithivMLmods/deepfake-detector-model-v1",
    name: "SigLIP Deepfake v1",
    author: "prithivMLmods",
    arch: "SigLIP-Base",
    params: "93M",
    year: "2025",
    tag: "MIXED",
    fakeLabels: ["fake"],
  },
  {
    id: "prithivMLmods/Deepfake-Detect-Siglip2",
    name: "SigLIP2 Deepfake",
    author: "prithivMLmods",
    arch: "SigLIP2-Base",
    params: "93M",
    year: "2026",
    tag: "MIXED",
    fakeLabels: ["fake"],
  },
  {
    id: "prithivMLmods/Deepfake-Detection-Exp-02-21",
    name: "Deepfake Exp 02-21",
    author: "prithivMLmods",
    arch: "ViT-Base",
    params: "86M",
    year: "2025",
    tag: "MIXED",
    fakeLabels: ["deepfake", "fake"],
  },
  {
    id: "Wvolf/ViT_Deepfake_Detection",
    name: "ViT Deepfake Detector",
    author: "Wvolf",
    arch: "ViT",
    params: "~86M",
    year: "2024",
    tag: "MIXED",
    fakeLabels: ["fake"],
  },
  {
    id: "dima806/deepfake_vs_real_image_detection",
    name: "Deepfake vs Real",
    author: "dima806",
    arch: "ViT-Base",
    params: "86M",
    year: "2025",
    tag: "FACES",
    fakeLabels: ["fake"],
  },
  {
    id: "Hemg/Deepfake-image-detection",
    name: "Deepfake Image Detect",
    author: "Hemg",
    arch: "ViT-Base",
    params: "86M",
    year: "2024",
    tag: "FACES",
    fakeLabels: ["fake"],
  },
  {
    id: "Organika/sdxl-detector",
    name: "SDXL Detector",
    author: "Organika",
    arch: "EfficientNet",
    params: "~5M",
    year: "2024",
    tag: "AI-GEN",
    fakeLabels: ["artificial", "fake", "ai"],
  },
  {
    id: "Heem2/AI-vs-Real-Image-Detection",
    name: "AI vs Real Image",
    author: "Heem2",
    arch: "Unknown",
    params: "~86M",
    year: "2024",
    tag: "MIXED",
    fakeLabels: ["ai", "artificial", "fake"],
  },
  {
    id: "umm-maybe/AI-image-detector",
    name: "AI Image Detector",
    author: "umm-maybe",
    arch: "ViT-Base",
    params: "86M",
    year: "2023",
    tag: "AI-ART",
    fakeLabels: ["artificial", "ai", "fake"],
  },
];

const IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp", "image/jpg"]);
const IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"];
const TAGLINE =
  "A panel of ten neural networks examines your image - independently - and renders a forensic verdict.";

function makeInitialSingleStates(): Record<string, SingleModelState> {
  return Object.fromEntries(MODELS.map((model) => [model.id, { status: "IDLE" as const }]));
}

function makeInitialBatchStats(): Record<string, BatchStat> {
  return Object.fromEntries(
    MODELS.map((model) => [
      model.id,
      {
        model,
        total: 0,
        correct: 0,
        confidenceSum: 0,
        errors: 0,
        running: false,
        done: false,
        lastError: "",
      },
    ]),
  );
}

function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(2)} MB`;
}

function fmtTime(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function fmtDateTime(value: string): string {
  return new Date(value).toLocaleString();
}

function shortHash(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return `00000000${(hash >>> 0).toString(16)}`.slice(-8).toUpperCase();
}

function genReportId(): string {
  const time = Date.now().toString(36).toUpperCase().slice(-6);
  const rand = Math.random().toString(36).toUpperCase().slice(2, 6);
  return `IS-${time}-${rand}`;
}

function todayStr(): string {
  const date = new Date();
  const months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
  return `${String(date.getDate()).padStart(2, "0")} ${months[date.getMonth()]} ${date.getFullYear()}`;
}

function isImageFile(file: File): boolean {
  const name = file.name.toLowerCase();
  return Boolean(file.type && IMAGE_TYPES.has(file.type)) || IMAGE_EXTS.some((ext) => name.endsWith(ext));
}

function makeImageThumbnail(file: File, maxSize = 180): Promise<string> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();

    img.onload = () => {
      const scale = Math.min(1, maxSize / Math.max(img.width, img.height));
      const width = Math.max(1, Math.round(img.width * scale));
      const height = Math.max(1, Math.round(img.height * scale));
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");

      if (!ctx) {
        URL.revokeObjectURL(url);
        reject(new Error("Canvas is unavailable."));
        return;
      }

      ctx.drawImage(img, 0, 0, width, height);
      URL.revokeObjectURL(url);
      resolve(canvas.toDataURL("image/jpeg", 0.72));
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not create image thumbnail."));
    };

    img.src = url;
  });
}

function classNames(...parts: Array<string | false | null | undefined | 0>): string {
  return parts.filter(Boolean).join(" ");
}

function interpretResult(model: ModelMeta, raw: RawPrediction[]): InterpretedResult {
  if (!Array.isArray(raw)) throw new Error("Unexpected response format");
  const sorted = [...raw].sort((a, b) => (b.score || 0) - (a.score || 0));
  const top = sorted[0];
  if (!top || typeof top.score !== "number") throw new Error("No prediction returned");

  const labelLower = String(top.label || "").toLowerCase();
  const fakeKeywords = model.fakeLabels.map((label) => label.toLowerCase());
  let isFake = fakeKeywords.some((keyword) => labelLower.includes(keyword));
  if (labelLower.startsWith("label_")) isFake = labelLower === "label_0";

  return {
    verdict: isFake ? "FAKE" : "REAL",
    confidence: top.score,
    rawLabel: top.label,
  };
}

async function callServer(modelId: string, file: File): Promise<RawPrediction[]> {
  const formData = new FormData();
  formData.append("model_id", modelId);
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    body: formData,
  });

  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (!response.ok) {
    const detail =
      body && typeof body === "object" && ("detail" in body || "error" in body)
        ? String((body as { detail?: unknown; error?: unknown }).detail ?? (body as { error?: unknown }).error)
        : `HTTP ${response.status}`;
    throw new Error(detail);
  }

  return body as RawPrediction[];
}

function batchAccuracy(stat: BatchStat): number {
  return stat.total ? stat.correct / stat.total : 0;
}

function batchAvgConfidence(stat: BatchStat): number {
  return stat.total ? stat.confidenceSum / stat.total : 0;
}

function Corners() {
  return (
    <div className="card-corners">
      <span />
      <span />
      <span />
      <span />
    </div>
  );
}

function UploadIcon() {
  return (
    <svg viewBox="0 0 32 32" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
      <path d="M28 20v6a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-6" />
      <polyline points="22 11 16 5 10 11" />
      <line x1="16" y1="5" x2="16" y2="21" />
    </svg>
  );
}

function FolderIcon({ large = false }: { large?: boolean }) {
  const viewBox = large ? "0 0 32 32" : "0 0 24 24";
  return (
    <svg viewBox={viewBox} fill="none" stroke="currentColor" strokeWidth={large ? "1.4" : "1.8"} strokeLinecap="round" strokeLinejoin="round">
      {large ? (
        <path d="M28 24a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h7l3 4h12a2 2 0 0 1 2 2z" />
      ) : (
        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
      )}
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="5 3 19 12 5 21 5 3" />
    </svg>
  );
}

function Card({
  className,
  children,
}: Readonly<{
  className: string;
  children: React.ReactNode;
}>) {
  function handleMove(event: MouseEvent<HTMLDivElement>) {
    const card = event.currentTarget;
    const rect = card.getBoundingClientRect();
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    const rx = ((event.clientY - rect.top - cy) / cy) * -1.5;
    const ry = ((event.clientX - rect.left - cx) / cx) * 1.5;
    card.style.transform = `perspective(1200px) rotateX(${rx}deg) rotateY(${ry}deg)`;
  }

  function handleLeave(event: MouseEvent<HTMLDivElement>) {
    event.currentTarget.style.transform = "perspective(1200px) rotateX(0) rotateY(0)";
  }

  return (
    <div className={classNames("card", className)} onMouseMove={handleMove} onMouseLeave={handleLeave}>
      <Corners />
      {children}
    </div>
  );
}

export default function Home() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const [reportId, setReportId] = useState("IS-PENDING");
  const [reportDate, setReportDate] = useState("-- --- ----");
  const [tagline, setTagline] = useState("");
  const [statusKind, setStatusKind] = useState<StatusKind>("");
  const [statusMessage, setStatusMessage] = useState(`Checking ${API_BASE}...`);
  const [statusSpinning, setStatusSpinning] = useState(false);
  const [backendOK, setBackendOK] = useState(false);
  const [loadedModelIds, setLoadedModelIds] = useState<Set<string>>(new Set());
  const [mode, setModeState] = useState<Mode>("single");
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [thumbUrl, setThumbUrl] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [singleStates, setSingleStates] = useState<Record<string, SingleModelState>>(makeInitialSingleStates);
  const [datasetImages, setDatasetImages] = useState<DatasetImage[]>([]);
  const [folderMeta, setFolderMeta] = useState({ name: "dataset", real: 0, fake: 0, skipped: 0 });
  const [batchStats, setBatchStats] = useState<Record<string, BatchStat>>(makeInitialBatchStats);
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);
  const [benchmarkDone, setBenchmarkDone] = useState(false);
  const [donePredictions, setDonePredictions] = useState(0);
  const [currentBenchmarkPath, setCurrentBenchmarkPath] = useState("-");
  const [elapsedMs, setElapsedMs] = useState(0);
  const [benchmarkStartedAt, setBenchmarkStartedAt] = useState<number | null>(null);
  const [glow, setGlow] = useState({ x: 0, y: 0, visible: false });
  const [savedBenchmarks, setSavedBenchmarks] = useState<SavedBenchmarkSummary[]>([]);
  const [selectedBenchmark, setSelectedBenchmark] = useState<SavedBenchmarkDetail | null>(null);
  const [benchmarkSaveStatus, setBenchmarkSaveStatus] = useState("");
  const [savedSingleImages, setSavedSingleImages] = useState<SavedSingleImageSummary[]>([]);
  const [selectedSingleImage, setSelectedSingleImage] = useState<SavedSingleImageDetail | null>(null);
  const [singleSaveStatus, setSingleSaveStatus] = useState("");

  useEffect(() => {
    setReportId(genReportId());
    setReportDate(todayStr());
  }, []);

  useEffect(() => {
    let index = 0;
    const timer = window.setInterval(() => {
      setTagline(TAGLINE.slice(0, index));
      index++;
      if (index > TAGLINE.length) window.clearInterval(timer);
    }, 22);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    void checkBackend();
  }, []);

  useEffect(() => {
    if (backendOK) {
      void loadSavedBenchmarks();
      void loadSavedSingleImages();
    }
  }, [backendOK]);

  useEffect(() => {
    if (!currentFile) {
      setThumbUrl("");
      return;
    }

    const url = URL.createObjectURL(currentFile);
    setThumbUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [currentFile]);

  useEffect(() => {
    if (!benchmarkRunning || benchmarkStartedAt === null) return undefined;
    const timer = window.setInterval(() => {
      setElapsedMs(Date.now() - benchmarkStartedAt);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [benchmarkRunning, benchmarkStartedAt]);

  const finishedSingleResults = useMemo(
    () => Object.values(singleStates).filter((state) => state.result).map((state) => state.result as InterpretedResult),
    [singleStates],
  );

  const totalPredictions = datasetImages.length * MODELS.length;
  const progressPct = totalPredictions ? Math.round((donePredictions / totalPredictions) * 100) : 0;
  const rankedBatchStats = useMemo(
    () =>
      Object.values(batchStats).sort((a, b) => {
        const accDelta = batchAccuracy(b) - batchAccuracy(a);
        if (accDelta !== 0) return accDelta;
        const confDelta = batchAvgConfidence(b) - batchAvgConfidence(a);
        if (confDelta !== 0) return confDelta;
        return b.total - a.total;
      }),
    [batchStats],
  );

  async function checkBackend() {
    setStatusKind("");
    setStatusMessage(`Checking ${API_BASE}...`);
    setStatusSpinning(true);

    try {
      const response = await fetch(`${API_BASE}/api/health`, { method: "GET" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = (await response.json()) as { loaded?: string[]; loaded_count?: number; models?: number };
      const loaded = new Set(data.loaded ?? []);
      const loadedCount = data.loaded_count ?? 0;
      const total = data.models ?? MODELS.length;

      setBackendOK(true);
      setLoadedModelIds(loaded);
      setStatusKind("ok");
      if (loadedCount === 0) setStatusMessage(`Connected - 0/${total} models warm - first use downloads model weights`);
      else if (loadedCount < total) setStatusMessage(`Connected - ${loadedCount}/${total} models warm - cold models download on first use`);
      else setStatusMessage(`Connected - all ${total} models warm in RAM`);
    } catch {
      setBackendOK(false);
      setStatusKind("err");
      setStatusMessage(`Cannot reach server at ${API_BASE}. From the project root, run: uvicorn backend.server:app --reload`);
    } finally {
      setStatusSpinning(false);
    }
  }

  async function loadSavedBenchmarks() {
    try {
      const response = await fetch(`${API_BASE}/api/benchmarks`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = (await response.json()) as { benchmarks: SavedBenchmarkSummary[] };
      setSavedBenchmarks(data.benchmarks);
    } catch {
      setSavedBenchmarks([]);
    }
  }

  async function loadBenchmarkDetail(id: number) {
    try {
      const response = await fetch(`${API_BASE}/api/benchmarks/${id}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setSelectedBenchmark((await response.json()) as SavedBenchmarkDetail);
    } catch {
      setSelectedBenchmark(null);
      setBenchmarkSaveStatus("Could not load saved benchmark.");
    }
  }

  async function loadSavedSingleImages() {
    try {
      const response = await fetch(`${API_BASE}/api/single-images`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = (await response.json()) as { images: SavedSingleImageSummary[] };
      setSavedSingleImages(data.images);
    } catch {
      setSavedSingleImages([]);
    }
  }

  async function loadSingleImageDetail(id: number) {
    try {
      const response = await fetch(`${API_BASE}/api/single-images/${id}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setSelectedSingleImage((await response.json()) as SavedSingleImageDetail);
    } catch {
      setSelectedSingleImage(null);
      setSingleSaveStatus("Could not load saved image.");
    }
  }

  async function saveSingleImage(results: Array<{ model: ModelMeta; result: InterpretedResult }>) {
    if (!currentFile || !results.length) return;

    try {
      const imageDataUrl = await makeImageThumbnail(currentFile, 420);
      const payload = {
        file_name: currentFile.name,
        file_size: currentFile.size,
        mime_type: "image/jpeg",
        image_data_url: imageDataUrl,
        results: results.map(({ model, result }) => ({
          model_id: model.id,
          model_name: model.name,
          author: model.author,
          tag: model.tag,
          verdict: result.verdict,
          confidence: result.confidence,
          raw_label: result.rawLabel,
        })),
      };

      const response = await fetch(`${API_BASE}/api/single-images`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const saved = (await response.json()) as { id: number };
      setSingleSaveStatus("Image analysis saved.");
      await loadSavedSingleImages();
      await loadSingleImageDetail(saved.id);
    } catch {
      setSingleSaveStatus("Image analyzed, but could not save to Neon.");
    }
  }

  async function saveBenchmark(stats: Record<string, BatchStat>, elapsed: number) {
    const ranked = Object.values(stats).sort((a, b) => {
      const accDelta = batchAccuracy(b) - batchAccuracy(a);
      if (accDelta !== 0) return accDelta;
      const confDelta = batchAvgConfidence(b) - batchAvgConfidence(a);
      if (confDelta !== 0) return confDelta;
      return b.total - a.total;
    });
    const payload = {
      dataset_name: folderMeta.name,
      total_images: datasetImages.length,
      real_count: folderMeta.real,
      fake_count: folderMeta.fake,
      skipped_count: folderMeta.skipped,
      elapsed_ms: elapsed,
      results: ranked.map((stat, index) => ({
        model_id: stat.model.id,
        model_name: stat.model.name,
        author: stat.model.author,
        tag: stat.model.tag,
        total: stat.total,
        correct: stat.correct,
        accuracy: batchAccuracy(stat),
        avg_confidence: batchAvgConfidence(stat),
        errors: stat.errors,
        rank: index + 1,
      })),
    };

    try {
      const response = await fetch(`${API_BASE}/api/benchmarks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const saved = (await response.json()) as { id: number };
      setBenchmarkSaveStatus("Benchmark saved.");
      await loadSavedBenchmarks();
      await loadBenchmarkDetail(saved.id);
    } catch {
      setBenchmarkSaveStatus("Benchmark finished, but could not save to Neon.");
    }
  }

  function setMode(nextMode: Mode) {
    if (benchmarkRunning) return;
    setModeState(nextMode);
  }

  function handleFile(file: File) {
    if (!isImageFile(file)) {
      window.alert("Please upload an image file (JPG, PNG, or WEBP).");
      return;
    }

    setCurrentFile(file);
    setSingleStates(makeInitialSingleStates());
    setBenchmarkDone(false);
    setSingleSaveStatus("");
  }

  function removeFile() {
    setCurrentFile(null);
    setSingleStates(makeInitialSingleStates());
    setSelectedSingleImage(null);
    setSingleSaveStatus("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function runModel(model: ModelMeta): Promise<InterpretedResult | null> {
    if (!backendOK) {
      setStatusKind("err");
      setStatusMessage("Server unreachable. Click retry once it is running.");
      return null;
    }
    if (!currentFile) return null;

    setSingleStates((prev) => ({
      ...prev,
      [model.id]: { status: "RUNNING" },
    }));

    try {
      const raw = await callServer(model.id, currentFile);
      const interpreted = interpretResult(model, raw);
      setLoadedModelIds((prev) => new Set(prev).add(model.id));
      setSingleStates((prev) => ({
        ...prev,
        [model.id]: { status: interpreted.verdict, result: interpreted },
      }));
      return interpreted;
    } catch (error) {
      setSingleStates((prev) => ({
        ...prev,
        [model.id]: {
          status: "ERROR",
          error: error instanceof Error ? error.message : "Unknown error",
        },
      }));
      return null;
    }
  }

  async function runAllModels() {
    if (!currentFile || !backendOK) {
      if (!backendOK) {
        setStatusKind("err");
        setStatusMessage("Server unreachable. Click retry once it is running.");
      }
      return;
    }

    setSingleSaveStatus("");
    const savedResults: Array<{ model: ModelMeta; result: InterpretedResult }> = [];

    for (const model of MODELS) {
      const result = await runModel(model);
      if (result) savedResults.push({ model, result });
    }

    await saveSingleImage(savedResults);
  }

  function parseDatasetFiles(files: FileList): {
    images: DatasetImage[];
    root: string;
    real: number;
    fake: number;
    skipped: number;
  } {
    const images: DatasetImage[] = [];
    let root = "";
    let real = 0;
    let fake = 0;
    let skipped = 0;

    Array.from(files).forEach((file) => {
      const datasetFile = file as DatasetFile;
      const rel = datasetFile.webkitRelativePath || datasetFile.name;
      const parts = rel.split("/").filter(Boolean);
      if (!root && parts.length > 1) root = parts[0];
      const labelPart = parts.map((part) => part.toLowerCase()).find((part) => part === "real" || part === "fake");

      if (!labelPart || !isImageFile(datasetFile)) {
        skipped++;
        return;
      }

      const expected: Verdict = labelPart === "fake" ? "FAKE" : "REAL";
      images.push({ file: datasetFile, expected, path: rel });
      if (expected === "REAL") real++;
      else fake++;
    });

    images.sort((a, b) => a.path.localeCompare(b.path));
    return { images, root: root || "dataset", real, fake, skipped };
  }

  function handleFolderSelection(files: FileList | null) {
    if (!files?.length) return;
    const parsed = parseDatasetFiles(files);
    setDatasetImages(parsed.images);
    setFolderMeta({ name: parsed.root, real: parsed.real, fake: parsed.fake, skipped: parsed.skipped });
    setBatchStats(makeInitialBatchStats());
    setDonePredictions(0);
    setCurrentBenchmarkPath("-");
    setElapsedMs(0);
    setBenchmarkDone(false);
  }

  function clearDataset() {
    if (benchmarkRunning) return;
    setDatasetImages([]);
    setFolderMeta({ name: "dataset", real: 0, fake: 0, skipped: 0 });
    setBatchStats(makeInitialBatchStats());
    setDonePredictions(0);
    setCurrentBenchmarkPath("-");
    setElapsedMs(0);
    setBenchmarkDone(false);
    if (folderInputRef.current) folderInputRef.current.value = "";
  }

  async function runBenchmark() {
    if (!backendOK) {
      setStatusKind("err");
      setStatusMessage("Server unreachable. Click retry once it is running.");
      return;
    }
    if (!datasetImages.length || benchmarkRunning) return;

    setBenchmarkRunning(true);
    setBenchmarkDone(false);
    setBatchStats(makeInitialBatchStats());
    setDonePredictions(0);
    setCurrentBenchmarkPath("-");
    setElapsedMs(0);
    const startedAt = Date.now();
    setBenchmarkStartedAt(startedAt);
    setModeState("batch");
    setBenchmarkSaveStatus("");

    let completed = 0;
    const localStats = makeInitialBatchStats();

    for (const model of MODELS) {
      localStats[model.id].running = true;
      setBatchStats((prev) => ({
        ...prev,
        [model.id]: { ...prev[model.id], running: true, done: false },
      }));

      for (const item of datasetImages) {
        setCurrentBenchmarkPath(item.path);

        try {
          const raw = await callServer(model.id, item.file);
          const interpreted = interpretResult(model, raw);
          setLoadedModelIds((prev) => new Set(prev).add(model.id));
          localStats[model.id].total += 1;
          localStats[model.id].correct += interpreted.verdict === item.expected ? 1 : 0;
          localStats[model.id].confidenceSum += interpreted.confidence;
          localStats[model.id].lastError = "";
          setBatchStats((prev) => {
            const stat = prev[model.id];
            return {
              ...prev,
              [model.id]: {
                ...stat,
                total: stat.total + 1,
                correct: stat.correct + (interpreted.verdict === item.expected ? 1 : 0),
                confidenceSum: stat.confidenceSum + interpreted.confidence,
                lastError: "",
              },
            };
          });
        } catch (error) {
          localStats[model.id].errors += 1;
          localStats[model.id].lastError = error instanceof Error ? error.message : "Unknown error";
          setBatchStats((prev) => {
            const stat = prev[model.id];
            return {
              ...prev,
              [model.id]: {
                ...stat,
                errors: stat.errors + 1,
                lastError: error instanceof Error ? error.message : "Unknown error",
              },
            };
          });
        }

        completed++;
        setDonePredictions(completed);
        await new Promise((resolve) => window.setTimeout(resolve, 0));
      }

      localStats[model.id].running = false;
      localStats[model.id].done = true;
      setBatchStats((prev) => ({
        ...prev,
        [model.id]: { ...prev[model.id], running: false, done: true },
      }));
    }

    setBenchmarkRunning(false);
    setBenchmarkDone(true);
    setBenchmarkStartedAt(null);
    const elapsed = Date.now() - startedAt;
    setElapsedMs(elapsed);
    setCurrentBenchmarkPath("COMPLETE");
    await saveBenchmark(localStats, elapsed);
  }

  const anySingleRunning = Object.values(singleStates).some((state) => state.status === "RUNNING");
  const intakeStatus = datasetImages.length && mode === "batch" ? "DATASET LOADED" : currentFile ? "SAMPLE LOADED" : "AWAITING SAMPLE";
  const intakeDone = Boolean(currentFile || datasetImages.length);
  const ensembleStatus = benchmarkRunning
    ? "BENCHMARK RUNNING"
    : benchmarkDone
      ? "BENCHMARK COMPLETE"
      : mode === "batch" && datasetImages.length
        ? "BENCHMARK READY"
        : currentFile
          ? anySingleRunning
            ? "RUNNING"
            : finishedSingleResults.length === MODELS.length
              ? "COMPLETE"
              : "READY"
          : "IDLE";

  return (
    <main
      onMouseMove={(event) => setGlow({ x: event.clientX, y: event.clientY, visible: true })}
      onMouseLeave={() => setGlow((prev) => ({ ...prev, visible: false }))}
    >
      <div className="paper-bg" />
      <div className="grid-bg" />
      <div
        className="mouse-glow"
        style={{
          left: `${glow.x}px`,
          top: `${glow.y}px`,
          opacity: glow.visible ? 1 : 0,
        }}
      />

      <div className="wrap">
        <header className="site-header">
          <div className="brand">
            <div className="brand-mark">
              <svg viewBox="0 0 32 32" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="16" cy="16" r="13" />
                <circle cx="16" cy="16" r="7" />
                <circle cx="16" cy="16" r="2" fill="currentColor" />
                <line x1="16" y1="0" x2="16" y2="6" />
                <line x1="16" y1="26" x2="16" y2="32" />
                <line x1="0" y1="16" x2="6" y2="16" />
                <line x1="26" y1="16" x2="32" y2="16" />
              </svg>
            </div>
            <div className="brand-text">
              <h1>
                InSpec <em>AI</em>
              </h1>
              <div className="brand-sub">Forensic Deepfake Analysis</div>
            </div>
          </div>
          <div className="header-meta">
            <span className="meta-row">
              <span className="meta-key">REPORT</span>
              <span className="meta-val">{reportId}</span>
            </span>
            <span className="meta-row">
              <span className="meta-key">DATE</span>
              <span className="meta-val">{reportDate}</span>
            </span>
          </div>
        </header>

        <p className="tagline">
          {tagline}
          <span className="cursor" />
        </p>

        <div className={classNames("status-bar", statusKind)}>
          <div className="status-bar-inner">
            <div className="status-icon">
              <span className="status-dot" />
            </div>
            <div className="status-content">
              <div className="status-label">PROXY SERVER</div>
              <div className="status-msg">{statusMessage}</div>
            </div>
            <button className={classNames("status-retry", statusSpinning && "spinning")} onClick={checkBackend} title="Retry connection" type="button">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="23 4 23 10 17 10" />
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
              </svg>
            </button>
          </div>
        </div>

        <Card className="upload-card">
          <div className="card-header">
            <span className="card-num">01</span>
            <span className="card-title">SPECIMEN INTAKE</span>
            <span className={classNames("card-status", intakeDone && "done")}>{intakeStatus}</span>
          </div>

          <div className="mode-tabs" role="tablist">
            <button className={classNames("mode-tab", mode === "single" && "active")} role="tab" aria-selected={mode === "single"} onClick={() => setMode("single")} type="button">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <circle cx="9" cy="9" r="2" />
                <path d="M21 15l-5-5L5 21" />
              </svg>
              <span>SINGLE IMAGE</span>
            </button>
            <button className={classNames("mode-tab", mode === "batch" && "active")} role="tab" aria-selected={mode === "batch"} onClick={() => setMode("batch")} type="button">
              <FolderIcon />
              <span>BATCH BENCHMARK</span>
            </button>
          </div>

          <div className={classNames("mode-panel", mode === "single" && "active")}>
            <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={(event: ChangeEvent<HTMLInputElement>) => event.target.files?.[0] && handleFile(event.target.files[0])} />

            <div
              className={classNames("upload-zone", dragOver && "dragover", currentFile && "has-file")}
              onClick={(event) => {
                if (!currentFile && !(event.target as HTMLElement).closest(".remove-btn")) fileInputRef.current?.click();
              }}
              onDragOver={(event: DragEvent<HTMLDivElement>) => {
                event.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(event: DragEvent<HTMLDivElement>) => {
                event.preventDefault();
                setDragOver(false);
                const file = event.dataTransfer.files[0];
                if (file) handleFile(file);
              }}
            >
              <div className={classNames("upload-empty", currentFile && "hidden")}>
                <div className="icon-frame">
                  <UploadIcon />
                </div>
                <h3>Drop an image here</h3>
                <p>
                  or <span className="link">browse files</span> from your device
                </p>
                <div className="formats">
                  <span className="fmt-pill">JPG</span>
                  <span className="fmt-pill">PNG</span>
                  <span className="fmt-pill">WEBP</span>
                </div>
              </div>

              <div className={classNames("file-preview", currentFile && "visible")}>
                <div className={classNames("thumb-panel", anySingleRunning && "scanning")}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  {thumbUrl && <img className="thumb-img visible" src={thumbUrl} alt="Specimen preview" />}
                  <span className="type-badge">{currentFile?.name.split(".").pop()?.toUpperCase() || "JPG"}</span>
                  <div className="scan-line" />
                  <div className="thumb-corners">
                    <span />
                    <span />
                    <span />
                    <span />
                  </div>
                </div>

                <div className="info-panel">
                  <InfoRow label="FILE" value={currentFile?.name || "-"} />
                  <InfoRow label="SIZE" value={currentFile ? fmtSize(currentFile.size) : "-"} mono />
                  <InfoRow label="TYPE" value={currentFile?.type || "unknown"} mono />
                  <InfoRow label="HASH" value={currentFile ? shortHash(currentFile.name + currentFile.size + currentFile.lastModified) : "-"} mono />
                  <div className="info-bottom">
                    <div className="ready-badge">
                      <div className="ready-dot" />
                      <span>SPECIMEN READY</span>
                    </div>
                    <button
                      className="remove-btn"
                      title="Remove file"
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        removeFile();
                      }}
                    >
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <div className="action-row">
              <button className="run-all-btn" disabled={!currentFile || anySingleRunning} onClick={runAllModels} type="button">
                <PlayIcon />
                <span>{anySingleRunning ? "ANALYZING..." : "RUN ALL TEN MODELS"}</span>
              </button>
              <div className="action-hint">or click any card below to run individually</div>
            </div>
          </div>

          <div className={classNames("mode-panel", mode === "batch" && "active")}>
            <input
              ref={folderInputRef}
              type="file"
              multiple
              hidden
              {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
              onChange={(event: ChangeEvent<HTMLInputElement>) => handleFolderSelection(event.target.files)}
            />

            <div
              className={classNames("upload-zone batch-zone", datasetImages.length > 0 && "has-folder")}
              onClick={(event) => {
                if (!(event.target as HTMLElement).closest("button") && !datasetImages.length) folderInputRef.current?.click();
              }}
            >
              <div className={classNames("upload-empty", datasetImages.length > 0 && "hidden")}>
                <div className="icon-frame">
                  <FolderIcon large />
                </div>
                <h3>Select your dataset folder</h3>
                <p>
                  folder must contain <span className="link">real/</span> and <span className="link">fake/</span> subfolders
                </p>
                <button
                  className="folder-pick-btn"
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    folderInputRef.current?.click();
                  }}
                >
                  <FolderIcon />
                  <span>CHOOSE FOLDER</span>
                </button>
                <div className="folder-hint mono">expected: dataset/real/*.jpg + dataset/fake/*.jpg</div>
              </div>

              <div className={classNames("folder-summary", datasetImages.length > 0 && "visible")}>
                <FolderRow label="DATASET" value={folderMeta.name} />
                <FolderRow label="TOTAL" value={`${datasetImages.length} images`} mono />
                <div className="folder-stats">
                  <FolderStat kind="real" value={folderMeta.real} label="REAL" />
                  <FolderStat kind="fake" value={folderMeta.fake} label="FAKE" />
                  <FolderStat kind="skip" value={folderMeta.skipped} label="SKIPPED" />
                </div>
                <div className="folder-progress-wrap">
                  <div className="folder-progress-label">
                    <span>{benchmarkRunning ? "RUNNING" : benchmarkDone ? "COMPLETE" : datasetImages.length ? "READY" : "NO VALID IMAGES"}</span>
                    <span className="mono">{benchmarkDone ? 100 : progressPct}%</span>
                  </div>
                  <div className="folder-progress-bar">
                    <div className="folder-progress-fill" style={{ width: `${benchmarkDone ? 100 : progressPct}%` }} />
                  </div>
                </div>
                <div className="folder-actions">
                  <button className="run-all-btn" disabled={!datasetImages.length || benchmarkRunning} onClick={runBenchmark} type="button">
                    <PlayIcon />
                    <span>{benchmarkRunning ? "BENCHMARKING..." : "START BENCHMARK"}</span>
                  </button>
                  <button className="folder-clear-btn" disabled={benchmarkRunning} onClick={clearDataset} type="button">
                    CLEAR
                  </button>
                </div>
              </div>
            </div>
          </div>
        </Card>

        <Card className="models-card">
          <div className="card-header">
            <span className="card-num">02</span>
            <span className="card-title">MODEL ENSEMBLE · 10 DETECTORS</span>
            <span className={classNames("card-status", (currentFile || datasetImages.length) && "active", (finishedSingleResults.length === MODELS.length || benchmarkDone) && "done")}>{ensembleStatus}</span>
          </div>

          <div className="models-grid">
            {MODELS.map((model, index) => (
              <ModelCard
                key={model.id}
                index={index}
                model={model}
                mode={mode}
                state={singleStates[model.id]}
                batchStat={batchStats[model.id]}
                disabled={mode !== "single" || !currentFile || singleStates[model.id]?.status === "RUNNING" || benchmarkRunning}
                loaded={loadedModelIds.has(model.id)}
                onRun={() => runModel(model)}
              />
            ))}
          </div>
        </Card>

        <Card className={classNames("leaderboard-card", mode === "single" && "visible")}>
          <div className="card-header">
            <span className="card-num">03</span>
            <span className="card-title">SAVED IMAGE ANALYSES · NEON</span>
            <span className={classNames("card-status", savedSingleImages.length > 0 && "done")}>{savedSingleImages.length ? `${savedSingleImages.length} STORED` : "NO DATA"}</span>
          </div>

          {singleSaveStatus && <div className="save-status">{singleSaveStatus}</div>}

          <div className="single-history">
            {savedSingleImages.length === 0 ? (
              <div className="lb-empty">NO SAVED IMAGES</div>
            ) : (
              savedSingleImages.map((image) => (
                <button
                  key={image.id}
                  className={classNames("single-history-card", selectedSingleImage?.id === image.id && "active")}
                  type="button"
                  onClick={() => loadSingleImageDetail(image.id)}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={image.image_data_url} alt={image.file_name} />
                  <span>
                    <strong>{image.file_name}</strong>
                    <small>{fmtDateTime(image.created_at)}</small>
                  </span>
                  <span>
                    <strong>{image.model_count}</strong>
                    <small>models</small>
                  </span>
                </button>
              ))
            )}
          </div>

          {selectedSingleImage && (
            <div className="saved-table-wrap">
              <div className="single-detail-head">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={selectedSingleImage.image_data_url} alt={selectedSingleImage.file_name} />
                <div>
                  <strong>{selectedSingleImage.file_name}</strong>
                  <span>{fmtSize(selectedSingleImage.file_size)} · {fmtDateTime(selectedSingleImage.created_at)}</span>
                </div>
              </div>
              <table className="saved-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Verdict</th>
                    <th>Confidence</th>
                    <th>Raw Label</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedSingleImage.results.map((result) => (
                    <tr key={result.model_id}>
                      <td>
                        <strong>{result.model_name}</strong>
                        <small>{result.author} · {result.tag}</small>
                      </td>
                      <td>{result.verdict}</td>
                      <td>{Math.round(result.confidence * 100)}%</td>
                      <td>{result.raw_label}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card className={classNames("leaderboard-card", mode === "batch" && "visible")}>
          <div className="card-header">
            <span className="card-num">03</span>
            <span className="card-title">MODEL LEADERBOARD · LIVE</span>
            <span className={classNames("card-status", benchmarkRunning && "active", benchmarkDone && "done")}>{benchmarkRunning ? "RUNNING" : benchmarkDone ? "COMPLETE" : datasetImages.length ? "READY" : "AWAITING DATA"}</span>
          </div>

          <div className="leaderboard-meta">
            <MetaItem label="PROCESSED" value={`${donePredictions} / ${totalPredictions}`} />
            <MetaItem label="CURRENT" value={currentBenchmarkPath} />
            <MetaItem label="ELAPSED" value={fmtTime(elapsedMs)} />
          </div>

          <div className="leaderboard-list">
            {!datasetImages.length ? (
              <div className="lb-empty">LOAD DATASET TO START</div>
            ) : (
              rankedBatchStats.map((stat, index) => <LeaderboardRow key={stat.model.id} stat={stat} rank={index + 1} />)
            )}
          </div>
        </Card>

        <Card className={classNames("leaderboard-card", mode === "batch" && "visible")}>
          <div className="card-header">
            <span className="card-num">04</span>
            <span className="card-title">SAVED BENCHMARKS · NEON</span>
            <span className={classNames("card-status", savedBenchmarks.length > 0 && "done")}>{savedBenchmarks.length ? `${savedBenchmarks.length} STORED` : "NO DATA"}</span>
          </div>

          {benchmarkSaveStatus && <div className="save-status">{benchmarkSaveStatus}</div>}

          <div className="benchmark-history">
            {savedBenchmarks.length === 0 ? (
              <div className="lb-empty">NO SAVED BENCHMARKS</div>
            ) : (
              savedBenchmarks.map((benchmark) => (
                <button
                  key={benchmark.id}
                  className={classNames("benchmark-history-card", selectedBenchmark?.id === benchmark.id && "active")}
                  type="button"
                  onClick={() => loadBenchmarkDetail(benchmark.id)}
                >
                  <span>
                    <strong>{benchmark.dataset_name}</strong>
                    <small>{fmtDateTime(benchmark.created_at)}</small>
                  </span>
                  <span>
                    <strong>{benchmark.best_accuracy !== null ? `${Math.round(benchmark.best_accuracy * 100)}%` : "--"}</strong>
                    <small>{benchmark.best_model || "No winner"}</small>
                  </span>
                  <span>
                    <strong>{benchmark.total_images}</strong>
                    <small>images</small>
                  </span>
                </button>
              ))
            )}
          </div>

          {selectedBenchmark && (
            <div className="saved-table-wrap">
              <div className="saved-table-meta">
                <span>{selectedBenchmark.dataset_name}</span>
                <span>{selectedBenchmark.total_images} images</span>
                <span>{fmtTime(selectedBenchmark.elapsed_ms)}</span>
              </div>
              <table className="saved-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Correct</th>
                    <th>Accuracy</th>
                    <th>Avg Conf</th>
                    <th>Errors</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedBenchmark.results.map((result) => (
                    <tr key={result.model_id}>
                      <td>{String(result.rank).padStart(2, "0")}</td>
                      <td>
                        <strong>{result.model_name}</strong>
                        <small>{result.author} · {result.tag}</small>
                      </td>
                      <td>{result.correct}/{result.total}</td>
                      <td>{Math.round(result.accuracy * 100)}%</td>
                      <td>{Math.round(result.avg_confidence * 100)}%</td>
                      <td>{result.errors}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <footer className="site-footer">
          <span>InSpec AI · Neural Authentication Engine · v4.0</span>
          <span className="mono">- end of report -</span>
        </footer>
      </div>
    </main>
  );
}

function InfoRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="info-row">
      <span className="info-key">{label}</span>
      <span className={classNames("info-val", mono && "mono")}>{value}</span>
    </div>
  );
}

function FolderRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="folder-row">
      <span className="folder-key">{label}</span>
      <span className={classNames("folder-val", mono && "mono")}>{value}</span>
    </div>
  );
}

function FolderStat({ kind, value, label }: { kind: "real" | "fake" | "skip"; value: number; label: string }) {
  return (
    <div className={classNames("folder-stat", kind)}>
      <div className="folder-stat-num">{value}</div>
      <div className="folder-stat-label">{label}</div>
    </div>
  );
}

function ModelCard({
  index,
  model,
  mode,
  state,
  batchStat,
  disabled,
  loaded,
  onRun,
}: {
  index: number;
  model: ModelMeta;
  mode: Mode;
  state: SingleModelState;
  batchStat: BatchStat;
  disabled: boolean;
  loaded: boolean;
  onRun: () => void;
}) {
  const isBatch = mode === "batch";
  const statusClass = isBatch
    ? batchStat.running
      ? "benchmarking"
      : batchStat.done
        ? "done"
        : batchStat.errors && !batchStat.total
          ? "error"
          : ""
    : state.status === "REAL" || state.status === "FAKE"
      ? state.status.toLowerCase()
      : state.status === "RUNNING"
        ? "running"
        : state.status === "ERROR"
          ? "error"
          : "";
  const statusText = isBatch ? (batchStat.running ? "BENCHMARKING" : batchStat.done ? "DONE" : batchStat.errors && !batchStat.total ? "ERROR" : "QUEUED") : state.status;

  return (
    <div className={classNames("model-card", disabled && "disabled", isBatch && "batch-mode")} onClick={() => !disabled && onRun()}>
      <div className="model-card-header">
        <span className="model-num">{String(index + 1).padStart(2, "0")}</span>
        <span className={classNames("model-status", statusClass)}>{statusText}</span>
      </div>
      <div className="model-name">{model.name}</div>
      <div className="model-author">
        {model.author} - {model.year}
      </div>
      <div className="model-specs">
        <span className="model-spec arch">{model.arch}</span>
        <span className="model-spec">{model.params} PARAMS</span>
        <span className="model-spec">{model.tag}</span>
      </div>
      <div className="model-result">{isBatch ? <BatchModelResult stat={batchStat} /> : <SingleModelResult state={state} loaded={loaded} />}</div>
    </div>
  );
}

function SingleModelResult({ state, loaded }: { state: SingleModelState; loaded: boolean }) {
  if (state.status === "RUNNING") {
    return (
      <div className="model-result-loading">
        <div className="mini-spinner" />
        <span>{loaded ? "INFERENCE" : "DOWNLOADING WEIGHTS"}</span>
      </div>
    );
  }

  if (state.status === "ERROR") {
    return <div className="model-error">! {(state.error || "Unknown error").slice(0, 200)}</div>;
  }

  if (!state.result) return <div className="model-result-empty">CLICK TO ANALYZE</div>;

  const cls = state.result.verdict.toLowerCase();
  const pct = Math.round(state.result.confidence * 100);

  return (
    <>
      <div className="model-verdict">
        <div className={classNames("verdict-text", cls)}>{state.result.verdict}</div>
        <div className="verdict-conf">
          <div className="verdict-conf-num">{pct}%</div>
          <div className="verdict-conf-label">CONFIDENCE</div>
        </div>
      </div>
      <div className="model-conf-bar">
        <div className={classNames("model-conf-fill", cls)} style={{ width: `${pct}%` }} />
      </div>
    </>
  );
}

function BatchModelResult({ stat }: { stat: BatchStat }) {
  const acc = Math.round(batchAccuracy(stat) * 100);
  const avgConf = Math.round(batchAvgConfidence(stat) * 100);
  const accClass = acc >= 80 ? "high" : acc >= 60 ? "mid" : "low";

  return (
    <>
      <div className="model-batch">
        <div className={classNames("model-batch-acc", accClass)}>{stat.total ? `${acc}%` : "--"}</div>
        <div className="model-batch-detail">
          <div className="model-batch-detail-correct">
            {stat.correct}/{stat.total}
          </div>
          <div className="model-batch-detail-label">AVG CONF {stat.total ? `${avgConf}%` : "--"}</div>
        </div>
      </div>
      {stat.lastError && <div className="model-error">{stat.lastError.slice(0, 120)}</div>}
    </>
  );
}

function MetaItem({ label, value }: { label: string; value: string }) {
  return (
    <span className="lb-meta-item">
      <span className="lb-meta-key">{label}</span>
      <span className="lb-meta-val mono">{value}</span>
    </span>
  );
}

function LeaderboardRow({ stat, rank }: { stat: BatchStat; rank: number }) {
  const acc = Math.round(batchAccuracy(stat) * 100);
  const avgConf = Math.round(batchAvgConfidence(stat) * 100);
  const medal = rank === 1 ? "GOLD" : rank === 2 ? "SILVER" : rank === 3 ? "BRONZE" : "";
  const errorClass = stat.errors && !stat.total ? "error" : "";

  return (
    <div className={classNames("lb-row", `rank-${rank}`, errorClass)}>
      <div className="lb-rank">
        <div className="lb-rank-num">{String(rank).padStart(2, "0")}</div>
        <div className="lb-medal">{medal}</div>
      </div>
      <div className="lb-info">
        <div className="lb-name">{stat.model.name}</div>
        <div className="lb-meta">
          {stat.model.author} - {stat.model.tag}
        </div>
      </div>
      <div className="lb-stats">
        <div className="lb-numbers">
          <span className="lb-acc">{stat.total ? `${acc}%` : "--"}</span>
          <span className="lb-detail">
            {stat.correct}/{stat.total} correct - avg conf {stat.total ? `${avgConf}%` : "--"} - errors {stat.errors}
          </span>
        </div>
        <div className="lb-bar-wrap">
          <div className="lb-bar" style={{ width: `${stat.total ? acc : 0}%` }} />
        </div>
      </div>
    </div>
  );
}
