# SPDX-License-Identifier: BSD-3-Clause
"""Register Gym tasks from the pip package `go2` (Unitree-Go2-Velocity, Unitree-Go2-Velocity-Play).

Install before enabling this extension::

    pip install -e /path/to/ABS-main/go2
"""

from __future__ import annotations

import importlib.util

if importlib.util.find_spec("go2") is None:
    raise ImportError(
        "isaaclab_go2_loader: package `go2` not found. Run: pip install -e <ABS-main>/go2"
    )

import go2  # noqa: F401
