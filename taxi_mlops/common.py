"""Local persistence shared by the lifecycle commands."""

import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        sync_dir(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def code_revision():
    package = Path(__file__).resolve().parent
    source = identity({p.name: digest(p) for p in sorted(package.glob("*.py"))})
    git = subprocess.run(
        ["git", "-C", str(package), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return {"git_commit": git.stdout.strip() if git.returncode == 0 else None, "source_sha256": source}


def verify_files(directory, checksums):
    for name, expected in checksums.items():
        if digest(directory / name) != expected:
            raise ValueError(f"Checksum mismatch for {directory / name}")
