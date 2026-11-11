"""
serfmad.backend
===============
Array-backend abstraction: CuPy (NVIDIA CUDA) when available, NumPy otherwise.

DESIGN RATIONALE
----------------
The physics kernels in this package are written once, against a duck-typed
array module `xp`. At import time we probe for CuPy + a usable CUDA device.
If found, `xp` is CuPy and the heavy kernels execute on the GPU; otherwise
`xp` is NumPy and everything runs on the CPU with identical semantics.

This matters because the workload here is genuinely GPU-shaped: the Bloch
solver integrates N_ensemble INDEPENDENT spin trajectories (Monte Carlo over
target moment / range / orientation / sea state) that share the same time
grid. That is an embarrassingly parallel SIMD problem -- exactly what a GPU
is for. It is NOT worth a GPU for a single trajectory, and this module does
not pretend otherwise: `should_use_gpu()` returns False below a threshold
ensemble size, because kernel-launch and host->device transfer overhead
dominates for small N.

HARDWARE VALIDATION STATUS
--------------------------
The CUDA path HAS now been executed on physical NVIDIA hardware: an RTX 3050
Laptop GPU (compute capability 8.6, driver 596.08, CUDA toolkit 12.4,
cupy-cuda12x 14.1.1), 2026-08-16. `main.py --gpu-verify` passes there --
max |error| 9.99e-15 against the closed-form Bloch fixed point, tolerance
1e-10, inter-replica spread exactly 0 (no race condition), fused RK4 kernel
in use. `--vv` and `--stress` produce output byte-identical to the CPU
backend, and every report figure regenerates byte-identical.

Exercising it exposed one real defect that the untested path had hidden: a
device array reached NumPy via np.asarray() in the V&V suite, which CuPy
refuses. Fixed; see validation/vv_suite.py v04.

Still unvalidated: non-Ampere architectures, multi-GPU, and float32 for
published results (see gpu.precision_study -- float32 is ~14x faster on this
part but carries ~6e-6 relative error, so it remains exploration-only).

This banner is deliberate, and stays specific about WHAT was tested on WHICH
hardware. A previous iteration of this project shipped a file labelled "GPU
driven" that contained no GPU code at all; the honest failure mode is to
state validation status explicitly rather than let a label imply a test that
never happened -- which cuts both ways once the test has actually been run.
"""

from __future__ import annotations
import os
import warnings
from typing import Any, Tuple

# ---------------------------------------------------------------------------
# Backend probe
# ---------------------------------------------------------------------------

_FORCE_CPU = os.environ.get("SERFMAD_FORCE_CPU", "0") == "1"


def _locate_cuda_toolkit():
    """
    Newest installed CUDA toolkit that has usable headers, or None.

    CuPy JIT-compiles every kernel at run time, so it needs the toolkit
    HEADERS, not merely a driver. With CUDA_PATH unset it imports fine,
    enumerates the device fine, allocates fine -- then dies on the first
    arithmetic op with "Failed to find CUDA headers", which the probe below
    catches as a plain CPU fallback. A perfectly good GPU then looks absent.

    Prefers a toolkit whose major version matches the installed CuPy wheel:
    cupy-cuda12x is built against CUDA 12.x and will not accept 13.x headers
    (a live concern on a machine whose driver already advertises 13.x). The
    wheel name comes from package metadata rather than by importing CuPy,
    because importing CuPy is precisely what this prepares for.
    """
    import glob
    import re

    want = None
    try:
        from importlib.metadata import distributions
        for d in distributions():
            name = ((d.metadata["Name"] or "") if d.metadata else "").lower()
            if name.startswith("cupy-cuda") and name.endswith("x"):
                want = name[len("cupy-cuda"):-1]      # cupy-cuda12x -> "12"
                break
    except Exception:
        pass

    roots = glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*")
    roots += sorted(glob.glob("/usr/local/cuda-*"))
    roots += ["/usr/local/cuda"]
    # NVIDIA's Windows installer also exports CUDA_PATH_V12_4 and friends.
    roots += [v for k, v in os.environ.items() if k.startswith("CUDA_PATH_V")]

    def version_of(path):
        m = re.search(r"(\d+)\.(\d+)", os.path.basename(path.rstrip("/\\")))
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    usable = [p for p in dict.fromkeys(roots)
              if os.path.isfile(os.path.join(p, "include", "cuda_runtime.h"))]
    if not usable:
        return None
    matching = [p for p in usable if want and str(version_of(p)[0]) == want]
    return max(matching or usable, key=version_of)


# Must happen before `import cupy` below, and therefore lives here rather than
# in main.py -- `python -m serfmad.gpu` and friends never run main.py at all.
# setdefault semantics: an explicit CUDA_PATH from the user always wins.
if not _FORCE_CPU and "CUDA_PATH" not in os.environ:
    _toolkit = _locate_cuda_toolkit()
    if _toolkit:
        os.environ["CUDA_PATH"] = _toolkit

GPU_AVAILABLE: bool = False
GPU_NAME: str = "none"
_BACKEND_DETAIL: str = ""

if _FORCE_CPU:
    import numpy as xp  # type: ignore
    _BACKEND_DETAIL = "CPU (forced via SERFMAD_FORCE_CPU=1)"
else:
    try:
        import cupy as xp  # type: ignore
        # Importing CuPy is not sufficient -- it can import successfully and
        # still fail at runtime with no driver/device. Force a real allocation.
        _probe = xp.zeros(8, dtype=xp.float64)
        _ = float((_probe + 1.0).sum())
        GPU_AVAILABLE = True
        try:
            _props = xp.cuda.runtime.getDeviceProperties(xp.cuda.runtime.getDevice())
            GPU_NAME = _props["name"].decode() if isinstance(_props["name"], bytes) \
                else str(_props["name"])
        except Exception:
            GPU_NAME = "CUDA device (name query failed)"
        _BACKEND_DETAIL = f"GPU / CuPy -- {GPU_NAME}"
        del _probe
    except Exception as _exc:            # ModuleNotFoundError, CUDARuntimeError, ...
        import numpy as xp  # type: ignore
        GPU_AVAILABLE = False
        GPU_NAME = "none"
        # Report the MESSAGE, not just the exception class. Showing only
        # "RuntimeError" hid "Failed to find CUDA headers" -- a one-line,
        # self-describing fix -- behind a name that suggested nothing at all,
        # and a working GPU looked simply absent for as long as that lasted.
        _msg = " ".join(str(_exc).split())
        if len(_msg) > 200:
            _msg = _msg[:197] + "..."
        _BACKEND_DETAIL = (f"CPU / NumPy (CuPy unavailable: "
                           f"{type(_exc).__name__}: {_msg})" if _msg
                           else f"CPU / NumPy (CuPy unavailable: "
                                f"{type(_exc).__name__})")

BACKEND_NAME: str = "cupy" if GPU_AVAILABLE else "numpy"

# Ensemble size below which the GPU is not worth the transfer/launch overhead.
# Chosen as an engineering heuristic, not a measured crossover; re-measure on
# target hardware with `benchmark()` and tune.
GPU_MIN_ENSEMBLE: int = 4096


def should_use_gpu(n_ensemble: int) -> bool:
    """True iff a GPU is present AND the problem is big enough to justify it."""
    return GPU_AVAILABLE and n_ensemble >= GPU_MIN_ENSEMBLE


def to_device(a):
    """Move a host array to the active device (no-op on CPU backend)."""
    return xp.asarray(a)


def to_host(a):
    """
    Move a device array back to host NumPy.

    All plotting/IO paths must call this: Matplotlib cannot consume CuPy
    arrays directly, and silently relying on __array__ coercion is a common
    source of hard-to-trace slowdowns.
    """
    if GPU_AVAILABLE:
        try:
            return xp.asnumpy(a)
        except (AttributeError, TypeError):
            return a
    return a


def sync():
    """Block until all queued device work completes (no-op on CPU)."""
    if GPU_AVAILABLE:
        try:
            xp.cuda.Stream.null.synchronize()
        except Exception:
            pass


def backend_report() -> str:
    """One-line human-readable backend status, shown in the GUI title bar."""
    return _BACKEND_DETAIL


def describe() -> dict:
    """Structured backend info for logging and the V&V report."""
    return {
        "backend": BACKEND_NAME,
        "gpu_available": GPU_AVAILABLE,
        "gpu_name": GPU_NAME,
        "detail": _BACKEND_DETAIL,
        "gpu_min_ensemble": GPU_MIN_ENSEMBLE,
        "hardware_validated": GPU_AVAILABLE,
    }


def benchmark(n_ensemble: int = 100_000, n_steps: int = 500) -> dict:
    """
    Micro-benchmark of the inner Bloch-style kernel shape on the active backend.

    Deliberately mirrors the real solver's memory access pattern (N x 3 state,
    N x 3 field, elementwise cross product + axpy per step) rather than timing
    an unrelated GEMM, so the number means something for THIS workload.
    """
    import time
    P = xp.zeros((n_ensemble, 3), dtype=xp.float64)
    P[:, 2] = 1.0
    B = xp.ones((n_ensemble, 3), dtype=xp.float64) * 1e-9
    sync()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        cx = P[:, 1] * B[:, 2] - P[:, 2] * B[:, 1]
        cy = P[:, 2] * B[:, 0] - P[:, 0] * B[:, 2]
        cz = P[:, 0] * B[:, 1] - P[:, 1] * B[:, 0]
        P[:, 0] += 1e-6 * cx
        P[:, 1] += 1e-6 * cy
        P[:, 2] += 1e-6 * cz
    sync()
    dt = time.perf_counter() - t0
    return {
        "backend": BACKEND_NAME,
        "n_ensemble": n_ensemble,
        "n_steps": n_steps,
        "wall_seconds": dt,
        "trajectory_steps_per_second": n_ensemble * n_steps / dt if dt > 0 else float("inf"),
    }


if __name__ == "__main__":
    print("serfmad backend self-check")
    print("-" * 50)
    for k, v in describe().items():
        print(f"  {k:22s}: {v}")
    print("-" * 50)
    if not GPU_AVAILABLE:
        print("NOTE: running on CPU. To validate the CUDA path, run this on a")
        print("      machine with an NVIDIA GPU and `pip install cupy-cuda12x`.")
    print("\nBenchmark (small, for smoke-test only):")
    for k, v in benchmark(n_ensemble=10_000, n_steps=100).items():
        print(f"  {k:30s}: {v}")
