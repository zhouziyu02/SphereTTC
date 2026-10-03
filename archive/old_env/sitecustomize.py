"""Runtime compatibility fixes for the managed execution container."""

from __future__ import annotations

try:
    import psutil

    _original_process = psutil.Process

    def _process_with_container_pid_fallback(pid=None):
        try:
            return _original_process(pid)
        except psutil.NoSuchProcess:
            if pid is None:
                return _original_process(1)
            raise

    psutil.Process = _process_with_container_pid_fallback
except Exception:
    pass
