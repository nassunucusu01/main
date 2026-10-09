"""Kuantizasyon iş kuyruğu: indir / aç -> GGUF'a dönüştür -> llama-quantize."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import uuid
import zipfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from . import setup_llama

DATA_DIR = setup_llama.ROOT / "data"
JOBS_DIR = DATA_DIR / "jobs"
UPLOADS_DIR = DATA_DIR / "uploads"
MAX_MODEL_GB = float(os.environ.get("KUANT_MAX_MODEL_GB", "32"))
JOB_TTL_SECONDS = int(os.environ.get("KUANT_JOB_TTL", str(3 * 3600)))

QUANT_TYPES = {
    "Q2_K": {"bpw": 3.2, "label": "En küçük boyut, belirgin kalite kaybı"},
    "Q3_K_M": {"bpw": 4.0, "label": "Çok küçük, orta kalite"},
    "Q4_K_M": {"bpw": 4.9, "label": "Önerilen: boyut / kalite dengesi"},
    "Q5_K_M": {"bpw": 5.7, "label": "Yüksek kalite"},
    "Q6_K": {"bpw": 6.6, "label": "Neredeyse kayıpsız"},
    "Q8_0": {"bpw": 8.5, "label": "Orijinale çok yakın, büyük boyut"},
}
DEFAULT_QUANT = "Q4_K_M"

REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.\-]{0,95}/[\w.\-]{1,95}$")
SAFE_NAME_RE = re.compile(r"^[\w.\- ()\[\]+]{1,200}$")
ARCHIVE_EXTS = (".zip", ".tar", ".tar.gz", ".tgz")
MODEL_FILE_EXTS = (".gguf", ".safetensors", ".bin", ".json", ".model", ".txt", ".tiktoken", ".jinja") + ARCHIVE_EXTS

BRANDS: list[tuple[str, tuple[str, ...]]] = [
    ("deepseek", ("deepseek",)),
    ("qwen", ("qwen", "qwq")),
    ("gemma", ("gemma",)),
    ("mistral", ("mistral", "mixtral", "ministral", "codestral", "devstral", "magistral", "pixtral")),
    ("microsoft", ("microsoft", "phi-", "phi2", "phi3", "phi4", "phimoe")),
    ("openai", ("openai", "gpt-oss", "gpt_oss")),
    ("nvidia", ("nvidia", "nemotron", "minitron")),
    ("huggingface", ("huggingfacetb", "smollm")),
    ("ibm", ("ibm-granite", "granite")),
    ("cohere", ("cohere", "command-r", "aya-")),
    ("yi", ("01-ai", "/yi-")),
    ("internlm", ("internlm",)),
    ("zhipu", ("zai-org", "thudm", "glm-", "chatglm", "glm4")),
    ("moonshot", ("moonshot", "kimi")),
    ("minimax", ("minimax",)),
    ("baichuan", ("baichuan",)),
    ("stability", ("stabilityai", "stablelm")),
    ("ai2", ("allenai", "olmo")),
    ("tii", ("tiiuae", "falcon")),
    ("xai", ("xai-org", "grok")),
    ("nousresearch", ("nousresearch", "hermes")),
    ("meta", ("meta-llama", "llama", "codellama", "facebook")),
    ("google", ("google/",)),
]


def detect_brand(*hints: str | None) -> str | None:
    text = " ".join(h for h in hints if h).lower()
    for brand, keys in BRANDS:
        if any(k in text for k in keys):
            return brand
    return None


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


class JobError(Exception):
    pass


@dataclass
class Job:
    id: str
    source: str
    quant: str
    model_name: str
    brand: str | None = None
    repo_id: str | None = None
    upload_id: str | None = None
    hf_token: str | None = field(default=None, repr=False)
    status: str = "queued"
    stage_label: str = "Sırada bekliyor"
    progress: float = 0.0
    eta_seconds: float | None = None
    error: str | None = None
    output_path: Path | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    log: deque = field(default_factory=lambda: deque(maxlen=60))
    _stages: list = field(default_factory=list)
    _stage_idx: int = 0
    _eta_smooth: float | None = None

    def public(self, queue_pos: int | None = None) -> dict:
        out = None
        if self.output_path and self.output_path.exists():
            out = {"filename": self.output_path.name, "size": self.output_path.stat().st_size,
                   "size_h": human_size(self.output_path.stat().st_size)}
        elapsed = None
        if self.started:
            elapsed = (self.finished or time.time()) - self.started
        return {
            "id": self.id,
            "status": self.status,
            "source": self.source,
            "quant": self.quant,
            "model_name": self.model_name,
            "brand": self.brand,
            "stage": self.stage_label,
            "progress": round(self.progress, 1),
            "eta_seconds": None if self.eta_seconds is None else int(self.eta_seconds),
            "elapsed_seconds": None if elapsed is None else int(elapsed),
            "queue_position": queue_pos,
            "error": self.error,
            "output": out,
        }

    # --- ilerleme -------------------------------------------------------
    def plan(self, stages: list[tuple[str, float]]) -> None:
        total = sum(w for _, w in stages)
        self._stages = [(label, w / total * 100) for label, w in stages]
        self._stage_idx = 0

    def stage(self, idx: int, frac: float = 0.0) -> None:
        self._stage_idx = idx
        label, weight = self._stages[idx]
        self.stage_label = label
        frac = max(0.0, min(1.0, frac))
        done = sum(w for _, w in self._stages[:idx])
        p = min(99.9, done + weight * frac)
        self.progress = max(self.progress, p)
        self._update_eta()

    def _update_eta(self) -> None:
        if not self.started or self.progress < 1.5:
            return
        elapsed = time.time() - self.started
        raw = elapsed * (100 - self.progress) / self.progress
        self._eta_smooth = raw if self._eta_smooth is None else 0.8 * self._eta_smooth + 0.2 * raw
        self.eta_seconds = self._eta_smooth


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.queue: deque[str] = deque()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.engine_ready = threading.Event()
        self.engine_error: str | None = None
        for d in (JOBS_DIR, UPLOADS_DIR):
            d.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        threading.Thread(target=self._prepare_engine, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._janitor, daemon=True).start()

    def _prepare_engine(self) -> None:
        try:
            setup_llama.setup()
            self.engine_ready.set()
        except Exception as e:  # noqa: BLE001
            self.engine_error = f"llama.cpp kurulamadı: {e}"
            print(self.engine_error, file=sys.stderr, flush=True)

    # --- kuyruk ---------------------------------------------------------
    def submit(self, job: Job) -> Job:
        with self.lock:
            self.jobs[job.id] = job
            self.queue.append(job.id)
        self.wake.set()
        return job

    def queue_position(self, job_id: str) -> int | None:
        with self.lock:
            try:
                return list(self.queue).index(job_id) + 1
            except ValueError:
                return None

    def _worker(self) -> None:
        while True:
            self.wake.wait()
            with self.lock:
                job_id = self.queue[0] if self.queue else None
                if job_id is None:
                    self.wake.clear()
                    continue
            job = self.jobs[job_id]
            if not self.engine_ready.is_set():
                job.stage_label = "Motor hazırlanıyor (llama.cpp kuruluyor)"
                while not self.engine_ready.wait(2):
                    if self.engine_error:
                        break
            try:
                if self.engine_error:
                    raise JobError(self.engine_error)
                job.status = "running"
                job.started = time.time()
                self._run(job)
                job.status = "done"
                job.progress = 100.0
                job.eta_seconds = 0
                job.stage_label = "İşlem bitti"
            except Exception as e:  # noqa: BLE001
                job.status = "error"
                job.error = str(e) if isinstance(e, JobError) else f"Beklenmeyen hata: {e}"
                job.stage_label = "Hata"
                print(f"[iş {job.id}] HATA: {e}", file=sys.stderr, flush=True)
            finally:
                job.finished = time.time()
                job.hf_token = None
                with self.lock:
                    if self.queue and self.queue[0] == job_id:
                        self.queue.popleft()

    def _janitor(self) -> None:
        while True:
            time.sleep(300)
            now = time.time()
            for job in list(self.jobs.values()):
                if job.finished and now - job.finished > JOB_TTL_SECONDS:
                    shutil.rmtree(JOBS_DIR / job.id, ignore_errors=True)
                    self.jobs.pop(job.id, None)
            for d in UPLOADS_DIR.iterdir():
                try:
                    if now - d.stat().st_mtime > JOB_TTL_SECONDS:
                        shutil.rmtree(d, ignore_errors=True)
                except OSError:
                    pass

    # --- işlem hattı ----------------------------------------------------
    def _run(self, job: Job) -> None:
        work = JOBS_DIR / job.id
        work.mkdir(parents=True, exist_ok=True)
        f16 = work / "model-f16.gguf"
        out = work / f"{_slug(job.model_name)}-{job.quant}.gguf"

        if job.source == "hf":
            job.plan([("Model Hugging Face'ten indiriliyor", 45), ("GGUF formatına dönüştürülüyor", 30),
                      (f"{job.quant} kuantizasyonu yapılıyor", 25)])
            src = work / "src"
            self._download_hf(job, src, 0)
            self._convert(job, src, f16, 1)
            shutil.rmtree(src, ignore_errors=True)
            self._quantize(job, f16, out, 2)
        else:
            upload = UPLOADS_DIR / (job.upload_id or "")
            if not upload.is_dir():
                raise JobError("Yüklenen dosyalar bulunamadı (süresi dolmuş olabilir)")
            files = [p for p in upload.iterdir() if p.is_file() and not p.name.endswith(".part")]
            ggufs = [p for p in files if p.name.lower().endswith(".gguf")]
            archives = [p for p in files if p.name.lower().endswith(ARCHIVE_EXTS)]
            if ggufs:
                job.plan([(f"{job.quant} kuantizasyonu yapılıyor", 100)])
                self._quantize(job, ggufs[0], out, 0)
            else:
                if archives:
                    job.plan([("Arşiv açılıyor", 10), ("GGUF formatına dönüştürülüyor", 50),
                              (f"{job.quant} kuantizasyonu yapılıyor", 40)])
                    src = work / "src"
                    _extract_archive(job, archives[0], src, 0)
                    conv_idx = 1
                else:
                    job.plan([("GGUF formatına dönüştürülüyor", 55), (f"{job.quant} kuantizasyonu yapılıyor", 45)])
                    src = upload
                    conv_idx = 0
                model_dir = _find_model_dir(src)
                self._convert(job, model_dir, f16, conv_idx)
                shutil.rmtree(work / "src", ignore_errors=True)
                self._quantize(job, f16, out, conv_idx + 1)
            shutil.rmtree(upload, ignore_errors=True)

        f16.unlink(missing_ok=True)
        if not out.exists():
            raise JobError("Çıktı dosyası oluşmadı")
        job.output_path = out

    def _download_hf(self, job: Job, dest: Path, idx: int) -> None:
        from huggingface_hub import hf_hub_download
        from tqdm.auto import tqdm as base_tqdm

        info = resolve_model(job.repo_id or "", job.hf_token)
        if not info.get("ok"):
            raise JobError(info.get("error") or "Model bulunamadı")
        files, total = info["_files"], max(1, info["_total_bytes"])
        job.stage(idx, 0)
        received = [0]

        class Progress(base_tqdm):
            def __init__(self, *args, **kwargs):
                kwargs["disable"] = True
                super().__init__(*args, **kwargs)

            def update(self, n=1):
                received[0] += n or 0
                job.stage(idx, received[0] / total)
                return super().update(n)

        try:
            for name in files:
                hf_hub_download(repo_id=info["repo_id"], filename=name, local_dir=str(dest),
                                token=job.hf_token or None, tqdm_class=Progress)
        except Exception as e:  # noqa: BLE001
            raise JobError(f"İndirme başarısız: {e}") from e
        shutil.rmtree(dest / ".cache", ignore_errors=True)
        job.stage(idx, 1.0)

    def _convert(self, job: Job, model_dir: Path, f16: Path, idx: int) -> None:
        script = setup_llama.find_convert()
        if script is None:
            raise JobError("convert_hf_to_gguf.py bulunamadı")
        job.stage(idx, 0)
        pat = re.compile(r"Writing:\s+(\d+)%")
        cmd = [sys.executable, str(script), str(model_dir), "--outfile", str(f16), "--outtype", "f16",
               "--model-name", _slug(job.model_name)]

        def on_line(line: str) -> None:
            m = pat.search(line)
            if m:
                job.stage(idx, 0.1 + 0.9 * int(m.group(1)) / 100)

        _run_proc(job, cmd, on_line, "GGUF dönüştürme")
        job.stage(idx, 1.0)

    def _quantize(self, job: Job, src: Path, out: Path, idx: int) -> None:
        qbin = setup_llama.find_quantize()
        if qbin is None:
            raise JobError("llama-quantize bulunamadı")
        job.stage(idx, 0)
        pat = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\]")
        threads = str(max(1, os.cpu_count() or 2))
        cmd = [str(qbin), "--allow-requantize", str(src), str(out), job.quant, threads]
        env = os.environ.copy()
        libdir = str(qbin.parent)
        for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            env[var] = libdir + (os.pathsep + env[var] if env.get(var) else "")

        def on_line(line: str) -> None:
            m = pat.search(line)
            if m and int(m.group(2)) > 0:
                job.stage(idx, int(m.group(1)) / int(m.group(2)))

        _run_proc(job, cmd, on_line, "Kuantizasyon", env=env)
        job.stage(idx, 1.0)


def _run_proc(job: Job, cmd: list[str], on_line, what: str, env: dict | None = None) -> None:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    buf = b""
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        for raw in parts:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                job.log.append(line)
                on_line(line)
    code = proc.wait()
    if code != 0:
        tail = [l for l in job.log if "error" in l.lower() or "exception" in l.lower()] or list(job.log)
        detail = tail[-1] if tail else ""
        if "not supported" in detail.lower() or "notimplemented" in detail.lower():
            detail = f"Bu model mimarisi llama.cpp tarafından desteklenmiyor. ({detail})"
        raise JobError(f"{what} başarısız oldu (kod {code}). {detail}".strip())


def _slug(name: str) -> str:
    base = name.split("/")[-1]
    return re.sub(r"[^\w.\-]+", "-", base).strip("-.") or "model"


def _extract_archive(job: Job, archive: Path, dest: Path, idx: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    job.stage(idx, 0)
    try:
        if archive.name.lower().endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                members = zf.infolist()
                for i, m in enumerate(members, 1):
                    target = (dest / m.filename).resolve()
                    if not str(target).startswith(str(root)):
                        raise JobError("Arşivde güvenli olmayan dosya yolu var")
                    zf.extract(m, dest)
                    job.stage(idx, i / max(1, len(members)))
        else:
            with tarfile.open(archive) as tf:
                try:
                    tf.extractall(dest, filter="data")
                except TypeError:
                    for m in tf.getmembers():
                        if not str((dest / m.name).resolve()).startswith(str(root)) or m.issym() or m.islnk():
                            raise JobError("Arşivde güvenli olmayan dosya yolu var")
                    tf.extractall(dest)
    except (zipfile.BadZipFile, tarfile.TarError) as e:
        raise JobError(f"Arşiv açılamadı: {e}") from e
    archive.unlink(missing_ok=True)
    job.stage(idx, 1.0)


def _find_model_dir(src: Path) -> Path:
    candidates = sorted(src.rglob("config.json"), key=lambda p: len(p.parts))
    for cfg in candidates:
        d = cfg.parent
        if any(d.glob("*.safetensors")) or any(d.glob("*.bin")):
            return d
    if candidates:
        raise JobError("config.json bulundu ama ağırlık dosyası (.safetensors / .bin) yok")
    raise JobError("Model klasörü bulunamadı: config.json + .safetensors (veya .bin) + tokenizer dosyaları gerekli")


# --- Hugging Face model çözümleme --------------------------------------------

def resolve_model(query: str, token: str | None = None) -> dict:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    q = query.strip().removeprefix("https://huggingface.co/").strip("/")
    if not q:
        return {"ok": False, "error": "Model adı boş"}
    api = HfApi(token=token or None)
    repo_id = q
    if "/" not in q:
        try:
            hits = list(api.list_models(search=q, sort="downloads", limit=20))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"Arama başarısız: {e}"}
        if not hits:
            return {"ok": False, "error": f"'{q}' adında model bulunamadı"}
        exact = [h for h in hits if h.id.split("/")[-1].lower() == q.lower()]
        repo_id = (exact or hits)[0].id
    if not REPO_ID_RE.match(repo_id):
        return {"ok": False, "error": "Geçersiz model adı. Örnek: Qwen/Qwen2.5-1.5B-Instruct"}
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except GatedRepoError:
        return {"ok": False, "repo_id": repo_id, "gated": True,
                "error": "Bu model erişim izni istiyor: Hugging Face'te lisansı kabul edip token girin"}
    except RepositoryNotFoundError:
        return {"ok": False, "error": f"'{repo_id}' bulunamadı (özel/gizli olabilir)"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"Model bilgisi alınamadı: {e}"}

    siblings = [(s.rfilename, s.size or 0) for s in (info.siblings or []) if "/" not in s.rfilename]
    names = [n for n, _ in siblings]
    st = [n for n in names if n.endswith(".safetensors")]
    pt = [n for n in names if n.endswith(".bin") and ("pytorch_model" in n or n.startswith("model"))]
    if st:
        weights = st
    elif pt:
        weights = pt
    elif any(n.endswith(".gguf") for n in names):
        return {"ok": False, "repo_id": repo_id,
                "error": "Bu depo zaten GGUF içeriyor. Orijinal (safetensors) modelin adını girin"}
    else:
        return {"ok": False, "repo_id": repo_id, "error": "Depoda .safetensors / .bin ağırlık dosyası yok"}

    extra = [n for n in names if n.endswith((".json", ".model", ".tiktoken", ".txt", ".jinja"))]
    files = sorted(set(weights + extra))
    total = sum(sz for n, sz in siblings if n in files)

    params = None
    if getattr(info, "safetensors", None) and getattr(info.safetensors, "total", None):
        params = int(info.safetensors.total)
    elif total:
        params = int(total / 2)

    gated = bool(getattr(info, "gated", False))
    model_type = None
    cfg = getattr(info, "config", None) or {}
    if isinstance(cfg, dict):
        model_type = cfg.get("model_type")
    result = {
        "ok": True,
        "repo_id": repo_id,
        "name": repo_id.split("/")[-1],
        "brand": detect_brand(repo_id, model_type),
        "gated": gated,
        "size_bytes": total,
        "size_h": human_size(total),
        "params_b": round(params / 1e9, 2) if params else None,
        "estimates": {k: human_size(params * v["bpw"] / 8) for k, v in QUANT_TYPES.items()} if params else {},
        "_files": files,
        "_total_bytes": total,
    }
    if total > MAX_MODEL_GB * 1024**3:
        result["ok"] = False
        result["error"] = (f"Model çok büyük ({human_size(total)}). Bu sunucu en fazla ~{MAX_MODEL_GB:.0f} GB "
                           f"(yaklaşık {MAX_MODEL_GB / 2:.0f}B parametre) modelleri işleyebilir")
    return result


def describe_upload(upload_dir: Path) -> tuple[str, str | None]:
    """Yüklenen dosyalardan model adı ve marka çıkarır."""
    files = [p for p in upload_dir.iterdir() if p.is_file()]
    for p in files:
        if p.name.lower().endswith(".gguf"):
            name, arch = _gguf_meta(p)
            stem = re.sub(r"(?i)[-_.](f16|f32|bf16|fp16)$", "", p.stem)
            display = name or stem
            return display, detect_brand(stem, name, arch)
    for p in files:
        low = p.name.lower()
        for ext in ARCHIVE_EXTS:
            if low.endswith(ext):
                stem = p.name[: -len(ext)]
                return stem, detect_brand(stem)
    cfg = upload_dir / "config.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text())
            name = data.get("_name_or_path") or (data.get("architectures") or [None])[0] or data.get("model_type")
            return str(name or "model"), detect_brand(str(name), data.get("model_type"))
        except (OSError, ValueError):
            pass
    first = files[0].stem if files else "model"
    return first, detect_brand(first)


def _gguf_meta(path: Path) -> tuple[str | None, str | None]:
    gp = setup_llama.gguf_py_dir()
    if gp.exists() and str(gp) not in sys.path:
        sys.path.insert(0, str(gp))
    try:
        from gguf import GGUFReader  # type: ignore

        r = GGUFReader(str(path))

        def get(key: str) -> str | None:
            f = r.fields.get(key)
            if not f or not f.data:
                return None
            return bytes(f.parts[f.data[0]]).decode("utf-8", "replace")

        return get("general.name"), get("general.architecture")
    except Exception:  # noqa: BLE001
        return None, None


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def safe_filename(name: str) -> str | None:
    name = os.path.basename(name.replace("\\", "/")).strip()
    if not name or name.startswith(".") or not SAFE_NAME_RE.match(name):
        return None
    if not name.lower().endswith(MODEL_FILE_EXTS):
        return None
    return name
