"""KuantLab web sunucusu: tek sayfa arayüz + kuantizasyon API'si."""
from __future__ import annotations

import os
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import quantizer as q

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = int(float(os.environ.get("KUANT_MAX_UPLOAD_GB", "40")) * 1024**3)
MAX_CHUNK_BYTES = 64 * 1024 * 1024

manager = q.JobManager()


@asynccontextmanager
async def lifespan(_: FastAPI):
    manager.start()
    yield


app = FastAPI(title="KuantLab", docs_url=None, redoc_url=None, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["Content-Length", "Content-Disposition"])


class JobRequest(BaseModel):
    source: str
    quant: str = q.DEFAULT_QUANT
    model: str | None = None
    upload_id: str | None = None
    hf_token: str | None = None


def _public_resolve(info: dict) -> dict:
    return {k: v for k, v in info.items() if not k.startswith("_")}


@app.get("/api/info")
def info() -> dict:
    return {
        "engine_ready": manager.engine_ready.is_set(),
        "engine_error": manager.engine_error,
        "quant_types": [{"id": k, **v} for k, v in q.QUANT_TYPES.items()],
        "default_quant": q.DEFAULT_QUANT,
        "max_model_gb": q.MAX_MODEL_GB,
        "max_upload_gb": MAX_UPLOAD_BYTES / 1024**3,
        "chunk_bytes": 32 * 1024 * 1024,
    }


@app.get("/api/resolve")
def resolve(query: str = Query("", alias="q", max_length=200),
            token: str | None = Header(None, alias="X-HF-Token")) -> dict:
    return _public_resolve(q.resolve_model(query, token))


@app.post("/api/uploads")
def create_upload() -> dict:
    uid = q.new_job_id()
    (q.UPLOADS_DIR / uid).mkdir(parents=True, exist_ok=True)
    return {"upload_id": uid}


def _upload_dir(uid: str) -> Path:
    if not uid.isalnum():
        raise HTTPException(400, "Geçersiz yükleme kimliği")
    d = q.UPLOADS_DIR / uid
    if not d.is_dir():
        raise HTTPException(404, "Yükleme bulunamadı")
    return d


@app.put("/api/uploads/{uid}/files/{filename}")
async def upload_chunk(uid: str, filename: str, request: Request, offset: int = 0, final: int = 0) -> dict:
    d = _upload_dir(uid)
    name = q.safe_filename(filename)
    if not name:
        raise HTTPException(400, "Desteklenmeyen dosya adı veya uzantısı")
    part = d / (name + ".part")
    current = part.stat().st_size if part.exists() else 0
    if offset != current:
        return JSONResponse({"error": "offset uyuşmuyor", "expected_offset": current}, status_code=409)
    used = q.dir_size(d)
    written = 0
    with part.open("ab") as fh:
        async for chunk in request.stream():
            written += len(chunk)
            if written > MAX_CHUNK_BYTES or used + written > MAX_UPLOAD_BYTES:
                fh.truncate(current)
                raise HTTPException(413, "Dosya boyutu sınırı aşıldı")
            fh.write(chunk)
    if final:
        part.rename(d / name)
    return {"ok": True, "size": current + written}


@app.post("/api/jobs")
def create_job(req: JobRequest) -> dict:
    if req.quant not in q.QUANT_TYPES:
        raise HTTPException(400, "Geçersiz kuantizasyon türü")
    if manager.engine_error:
        raise HTTPException(503, manager.engine_error)

    if req.source == "hf":
        info = q.resolve_model(req.model or "", req.hf_token)
        if not info.get("ok"):
            raise HTTPException(400, info.get("error") or "Model bulunamadı")
        job = q.Job(id=q.new_job_id(), source="hf", quant=req.quant, model_name=info["repo_id"],
                    brand=info.get("brand"), repo_id=info["repo_id"], hf_token=req.hf_token)
    elif req.source == "upload":
        d = _upload_dir(req.upload_id or "")
        files = [p for p in d.iterdir() if p.is_file() and not p.name.endswith(".part")]
        if not files:
            raise HTTPException(400, "Yüklenmiş dosya yok")
        name, brand = q.describe_upload(d)
        job = q.Job(id=q.new_job_id(), source="upload", quant=req.quant, model_name=name,
                    brand=brand, upload_id=d.name)
    else:
        raise HTTPException(400, "Geçersiz kaynak")
    manager.submit(job)
    return job.public(manager.queue_position(job.id))


def _get_job(job_id: str) -> q.Job:
    job = manager.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "İş bulunamadı (süresi dolmuş olabilir)")
    return job


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _get_job(job_id)
    data = job.public(manager.queue_position(job.id) if job.status == "queued" else None)
    if job.status == "queued" and not manager.engine_ready.is_set():
        data["stage"] = "Motor hazırlanıyor (llama.cpp kuruluyor)"
    return data


@app.api_route("/api/jobs/{job_id}/download", methods=["GET", "HEAD"])
def download(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    if job.status != "done" or not job.output_path or not job.output_path.exists():
        raise HTTPException(409, "Çıktı henüz hazır değil")
    return FileResponse(job.output_path, filename=job.output_path.name, media_type="application/octet-stream")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


class NoCacheStatic(StaticFiles):
    """Her istekte doğrulama ister (ETag ile 304); eski arayüzün önbellekte kalmasını önler."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


# Önemli: bu bağlamalar en sonda olmalı; API yolları yukarıda tanımlı.
app.mount("/static", NoCacheStatic(directory=STATIC_DIR), name="static")
app.mount("/", NoCacheStatic(directory=STATIC_DIR), name="site")
