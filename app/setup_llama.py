"""llama.cpp araçlarını hazırlar: convert_hf_to_gguf.py (HF -> GGUF) ve llama-quantize."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("KUANT_HOME", Path(__file__).resolve().parent.parent / ".kuantlab"))
LLAMA_DIR = ROOT / "llama.cpp"
BIN_DIR = ROOT / "llama-bin"
RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30"
REPO_URL = "https://github.com/ggml-org/llama.cpp"

PY_DEPS = ["numpy", "sentencepiece", "protobuf", "transformers", "safetensors", "tqdm", "pyyaml", "requests"]


def _log(msg: str) -> None:
    print(f"[kurulum] {msg}", flush=True)


def _asset_suffix() -> str:
    system, machine = platform.system(), platform.machine().lower()
    if system == "Linux":
        return "bin-ubuntu-x64.tar.gz" if machine in ("x86_64", "amd64") else "bin-ubuntu-arm64.tar.gz"
    if system == "Darwin":
        return "bin-macos-arm64.tar.gz" if machine == "arm64" else "bin-macos-x64.tar.gz"
    raise RuntimeError(f"Desteklenmeyen işletim sistemi: {system}")


def find_quantize() -> Path | None:
    if BIN_DIR.exists():
        for p in BIN_DIR.rglob("llama-quantize"):
            if p.is_file():
                return p
    found = shutil.which("llama-quantize")
    return Path(found) if found else None


def find_convert() -> Path | None:
    p = LLAMA_DIR / "convert_hf_to_gguf.py"
    return p if p.exists() else None


def gguf_py_dir() -> Path:
    return LLAMA_DIR / "gguf-py"


def is_ready() -> bool:
    return find_quantize() is not None and find_convert() is not None


def _ssl_ctx():
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _open(url: str, accept: str = "*/*"):
    req = urllib.request.Request(url, headers={"User-Agent": "kuantlab", "Accept": accept})
    return urllib.request.urlopen(req, timeout=120, context=_ssl_ctx())


def _http_json(url: str):
    with _open(url, "application/vnd.github+json") as r:
        return json.load(r)


def _download(url: str, dest: Path) -> None:
    with _open(url) as r, dest.open("wb") as fh:
        shutil.copyfileobj(r, fh, 1024 * 1024)


def _pick_release() -> tuple[str, str]:
    suffix = _asset_suffix()
    for rel in _http_json(RELEASES_API):
        tag = rel["tag_name"]
        for asset in rel.get("assets", []):
            if asset["name"] == f"llama-{tag}-{suffix}":
                return tag, asset["browser_download_url"]
    raise RuntimeError("Uygun llama.cpp sürümü bulunamadı")


def _extract(archive: Path, dest: Path) -> None:
    with tarfile.open(archive) as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:
            tf.extractall(dest)


def _missing_py_deps() -> list[str]:
    import importlib.util

    names = {"pyyaml": "yaml", "protobuf": "google.protobuf"}
    missing = []
    for dep in PY_DEPS + ["torch"]:
        mod = names.get(dep, dep)
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(dep)
        except ModuleNotFoundError:
            missing.append(dep)
    return missing


def setup(force: bool = False) -> None:
    if is_ready() and not force and not _missing_py_deps():
        _log("llama.cpp zaten hazır")
        return
    ROOT.mkdir(parents=True, exist_ok=True)

    tag, url = _pick_release()
    _log(f"llama.cpp sürümü: {tag}")

    if force or find_quantize() is None:
        archive = ROOT / "llama-bin.tar.gz"
        _log("hazır derlenmiş llama-quantize indiriliyor...")
        _download(url, archive)
        shutil.rmtree(BIN_DIR, ignore_errors=True)
        _extract(archive, BIN_DIR)
        archive.unlink(missing_ok=True)
        q = find_quantize()
        if q is None:
            raise RuntimeError("llama-quantize arşivde bulunamadı")
        q.chmod(0o755)

    if force or find_convert() is None:
        _log("llama.cpp dönüştürme betikleri indiriliyor...")
        shutil.rmtree(LLAMA_DIR, ignore_errors=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", tag, REPO_URL, str(LLAMA_DIR)],
            check=True,
        )

    missing = _missing_py_deps()
    if missing:
        _log(f"Python paketleri kuruluyor: {', '.join(missing)}")
        cmd = [sys.executable, "-m", "pip", "install", "-q", *missing]
        if "torch" in missing:
            cmd += ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]
        subprocess.run(cmd, check=True)

    _log("hazır ✔")


if __name__ == "__main__":
    setup(force="--force" in sys.argv)
