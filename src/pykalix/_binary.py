"""Binary discovery for the kalix CLI.

Searches in this order:
1. User-provided path (``binary=`` parameter)
2. ``KALIX_BINARY`` environment variable
3. Bundled binary in the package (``pykalix/bin/kalix``)
4. ``kalix`` on ``PATH``

If none is found, returns ``None``.  Callers should raise a helpful
error suggesting ``cargo install kalix``.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def find_kalix(user_supplied: str | None = None) -> str | None:
    """Return the path to a usable ``kalix`` binary, or ``None``.

    Parameters:
        user_supplied: An explicit path or name supplied by the caller.
    """
    # 1. User-supplied (explicit path or name on PATH)
    if user_supplied is not None:
        resolved = _resolve(user_supplied)
        if resolved is not None:
            logger.debug("using user-supplied binary: %s", resolved)
            return resolved

    # 2. Environment variable
    env_val = os.environ.get("KALIX_BINARY")
    if env_val is not None:
        resolved = _resolve(env_val)
        if resolved is not None:
            logger.debug("using KALIX_BINARY: %s", resolved)
            return resolved

    # 3. Bundled binary
    bundled = _find_bundled()
    if bundled is not None:
        # Ensure it's executable
        _ensure_executable(bundled)
        logger.debug("using bundled binary: %s", bundled)
        return str(bundled)

    # 4. PATH lookup (last resort)
    path_found = shutil.which("kalix")
    if path_found is not None:
        logger.debug("using PATH binary: %s", path_found)
        return path_found

    return None


def _resolve(name_or_path: str) -> str | None:
    """If the value is an absolute/explicit path, check it exists.
    Otherwise look it up on PATH.
    """
    if os.path.isabs(name_or_path) or "/" in name_or_path or "\\" in name_or_path:
        p = Path(name_or_path)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        return None
    return shutil.which(name_or_path)


def _find_bundled() -> Path | None:
    """Find the bundled kalix binary shipped inside the package.

    Uses ``importlib.resources`` (Python ≥ 3.9) to locate the file
    regardless of whether the package is installed as a wheel, an
    editable install, or a zip.
    """
    try:
        # Python 3.9+
        from importlib.resources import files as _resource_files
    except ImportError:
        return None

    # Prefer the modern traversal API
    try:
        bin_dir = _resource_files("pykalix") / "bin"
        candidate = bin_dir / "kalix"
        if candidate.is_file():
            return Path(str(candidate))
    except Exception:
        pass

    # Fallback: resolve relative to this module's location
    # (covers editable installs and some edge cases)
    try:
        this_dir = Path(__file__).resolve().parent
        candidate = this_dir / "bin" / "kalix"
        if candidate.is_file():
            return candidate
    except Exception:
        pass

    return None


def _ensure_executable(path: Path) -> None:
    """Make sure the file has the executable bit set."""
    if os.access(path, os.X_OK):
        return
    try:
        path.chmod(path.stat().st_mode | 0o111)
    except OSError:
        logger.warning("cannot make %s executable", path)
