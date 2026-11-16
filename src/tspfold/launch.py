"""Portable torchrun command construction.

Uses static rendezvous on 127.0.0.1 with an explicit free port: `--standalone` resolves `localhost`
via IPv6 reverse lookup, which hangs on some macOS hosts.
"""

from __future__ import annotations

import socket
import sys


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def torchrun_command(module: str, args: list[str], nproc: int = 2, master_port: int | None = None) -> list[str]:
    port = master_port or free_port()
    return [
        sys.executable, "-m", "torch.distributed.run",
        "--nnodes=1", f"--nproc_per_node={nproc}",
        "--master_addr=127.0.0.1", f"--master_port={port}",
        "-m", module, *args,
    ]  # fmt: skip
