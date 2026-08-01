from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Sequence


def setup_realtime_output() -> None:
    """Make long-running research/training scripts visibly stream progress.

    The desktop terminal is only useful when Python, logging, CatBoost, and
    child processes flush promptly.  This function is intentionally idempotent
    and cheap so scripts can call it at import or main entry points.
    """
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(line_buffering=True, write_through=True)
            except Exception:
                pass
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        force=False,
    )


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_step(message: str) -> None:
    print(f"[{ts()}] {message}", flush=True)


def _pipe_reader(pipe: IO[str], prefix: str, log_file: IO[str] | None) -> None:
    try:
        for line in iter(pipe.readline, ""):
            text = line.rstrip("\n")
            if prefix:
                print(f"{prefix}{text}", flush=True)
            else:
                print(text, flush=True)
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def run_streaming(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    prefix: str = "",
    log_path: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run a child process and stream stdout/stderr to the terminal in real time."""
    setup_realtime_output()
    merged_env = os.environ.copy()
    merged_env["PYTHONUNBUFFERED"] = "1"
    if env:
        merged_env.update(env)
    log_file = None
    if log_path is not None:
        log_file = Path(log_path).open("a", encoding="utf-8")
    try:
        log_step(f"[subprocess] start: {' '.join(map(str, command))}")
        proc = subprocess.Popen(
            list(map(str, command)),
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=merged_env,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None
        out_thread = threading.Thread(
            target=_pipe_reader,
            args=(proc.stdout, prefix, log_file),
            daemon=True,
        )
        err_thread = threading.Thread(
            target=_pipe_reader,
            args=(proc.stderr, prefix, log_file),
            daemon=True,
        )
        out_thread.start()
        err_thread.start()
        return_code = proc.wait()
        out_thread.join()
        err_thread.join()
        log_step(f"[subprocess] done rc={return_code}: {' '.join(map(str, command))}")
        return return_code
    finally:
        if log_file is not None:
            log_file.close()
