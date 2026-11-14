"""Safe to publish: no checkpoint file and no secrets in the repository"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEY = re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}")
SKIP = {".git", ".venv", "venv", "__pycache__", "node_modules"}


def _files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not SKIP & set(path.relative_to(ROOT).parts):
            yield path


def test_no_api_key_in_files():
    """No API key appears in any file"""
    for path in _files():
        if path.stat().st_size < 2_000_000:
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert not KEY.search(text), f"{path.relative_to(ROOT)} holds an API key. Delete it, revoke the key and use a secret."


def test_checkpoints_ignored(src):
    """.gitignore keeps .env, ckpt.pt and out-*/ folders out of Git"""
    ignore = src / ".gitignore"
    lines = ignore.read_text(encoding="utf-8").split() if ignore.exists() else []
    for rule in (".env", "ckpt.pt", "out-*/"):
        assert rule in lines, f"Add the line {rule} to .gitignore. nanoGPT checkpoints are about 128 MB, over GitHub's 100 MiB limit, and a push with one is refused."


def test_no_checkpoint_file_in_repository():
    """No checkpoint or model weights file sits in the repository"""
    for path in _files():
        assert path.suffix not in {".pt", ".pth", ".ckpt", ".safetensors"}, f"{path.relative_to(ROOT)} is a model file. Delete it; your repository holds code, logs and samples, not weights."
