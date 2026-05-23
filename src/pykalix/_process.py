"""Subprocess management for the kalix binary."""

from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Iterator
from typing import IO

logger = logging.getLogger(__name__)


class KalixProcess:
    """Manages the kalix child process and JSON-line communication.

    Spawns the ``kalix`` binary with the given arguments and communicates
    over stdin/stdout using newline-delimited JSON.  Stderr is captured
    and forwarded to Python logging.

    Raises:
        FileNotFoundError: The ``binary`` is not on ``PATH``.
        RuntimeError: The process exits unexpectedly.
    """

    def __init__(
        self,
        binary: str,
        config_path: str,
        mode: str,
        on_error: str,
    ) -> None:
        self._binary = binary
        self._config_path = config_path
        self._mode = mode
        self._on_error = on_error
        self._proc: subprocess.Popen[str] | None = None
        self._stderr_thread: threading.Thread | None = None

    # ── process lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        """Launch the kalix process."""
        cmd = [
            self._binary,
            "--config",
            self._config_path,
            "--mode",
            self._mode,
            "--on-error",
            self._on_error,
        ]
        logger.debug("spawning: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Drain stderr in a background thread so the pipe doesn't fill up.
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._proc.stderr,),
            daemon=True,
        )
        self._stderr_thread.start()

    def close(self) -> None:
        """Shut down the process gracefully."""
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        finally:
            self._proc = None
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=2)
                self._stderr_thread = None

    # ── I/O ────────────────────────────────────────────────────────────

    @property
    def stdin(self) -> IO[str]:
        """Writable pipe to the process stdin.  Raises if not started."""
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Kalix process not started")
        return self._proc.stdin

    @property
    def stdout(self) -> IO[str]:
        """Readable pipe from process stdout.  Raises if not started."""
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Kalix process not started")
        return self._proc.stdout

    def send_line(self, line: str) -> None:
        """Write a single JSON line to stdin and flush."""
        self.stdin.write(line)
        if not line.endswith("\n"):
            self.stdin.write("\n")
        self.stdin.flush()

    def read_line(self) -> str:
        """Read a single line from stdout.  Blocks until available.

        Returns:
            The raw JSON line (trailing newline stripped).

        Raises:
            EOFError: The process closed its stdout unexpectedly.
        """
        line = self.stdout.readline()
        if not line:
            raise EOFError("kalix process closed stdout unexpectedly")
        return line.rstrip("\n")

    def iter_lines(self) -> Iterator[str]:
        """Yield output lines lazily (non-blocking iterator over stdout)."""
        while True:
            line = self.stdout.readline()
            if not line:
                break
            yield line.rstrip("\n")

    @property
    def returncode(self) -> int | None:
        """Exit code if the process has terminated, or ``None``."""
        if self._proc is None:
            return None
        return self._proc.returncode

    # ── internals ──────────────────────────────────────────────────────

    def _drain_stderr(self, stream: IO[str] | None) -> None:
        """Read stderr lines and log them as warnings."""
        if stream is None:
            return
        try:
            for line in stream:
                msg = line.rstrip("\n")
                if msg:
                    # kalix sends structured JSON to stderr via tracing.
                    # We try to parse it for a nicer log message, falling
                    # back to the raw JSON string.
                    try:
                        import json

                        parsed = json.loads(msg)
                        level = parsed.get("level", "INFO")
                        message = parsed.get("message", msg)
                        fields = parsed.get("fields", parsed)
                        extra = {k: v for k, v in fields.items() if k != "message"}
                        log_msg = f"[kalix {level}] {message}"
                        if extra:
                            log_msg += f"  {extra}"
                        logger.debug("%s", log_msg)
                    except Exception:
                        logger.debug("[kalix stderr] %s", msg)
        except Exception:
            pass  # pipe closed
