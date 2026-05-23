"""Setup script for pykalix.

Forces a platform-specific wheel with ``py3-none-<platform>`` tag because
the package ships a compiled Rust binary (``src/pykalix/bin/kalix``).

All metadata lives in ``pyproject.toml`` — this file exists solely
for the wheel-tag override.
"""

from __future__ import annotations

from setuptools import setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class bdist_wheel(_bdist_wheel):  # noqa: N801
    """Wheel builder that forces ``py3-none-<platform>`` tags.

    The bundled binary is a standalone executable, not a CPython
    extension module, so the wheel should not be ABI-tagged.
    """

    def finalize_options(self) -> None:
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False  # platform-specific (contains binary)

    def get_tag(self) -> tuple[str, str, str]:
        _, _, plat = _bdist_wheel.get_tag(self)
        # Force abi to "none" — the binary is standalone, not a C ext
        return "py3", "none", plat


setup(cmdclass={"bdist_wheel": bdist_wheel})
