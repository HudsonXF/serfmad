"""
serfmad.gpu
===========
High-performance CUDA compute paths for large-scale SERF-MAD simulation.

WHAT THIS MODULE IS FOR
-----------------------
The physics in `serfmad.quantum.bloch` and `serfmad.mission` is correct but
written for clarity, using generic array operations that work identically on
CPU and GPU. That is the right default. This module is the opposite trade:
it exists to push a real NVIDIA GPU hard, and contains

  * a fused custom CUDA kernel for the Bloch RK4 step (one kernel launch per
    timestep instead of ~20 elementwise kernel launches),
  * memory-aware chunking so ensembles larger than VRAM still run,
  * batched FFT matched filtering across millions of trajectories,
  * a multi-dimensional parameter sweep engine (range x moment x orientation
    x sea state x CMRR) that is the actual scientific payload, and
  * a scaling study that measures, rather than assumes, where the GPU wins.

WHY THESE WORKLOADS SUIT A GPU
------------------------------
Every one of them is "many independent replicas of the same small problem":
N spin ensembles, each 3 floats of state, all sharing one time grid, with no
communication between replicas. That is textbook SIMD. Arithmetic intensity
per timestep is low, so the kernel is memory-bandwidth-bound -- which is why
the fused kernel matters: it reads and writes state once per step rather than
once per intermediate array.

By contrast a SINGLE Bloch trajectory is not a GPU workload at any size, and
this module does not pretend otherwise. `recommend_backend()` returns an
honest answer including "use the CPU".

PRECISION POLICY
----------------
Default is float64. Consumer GeForce cards have heavily reduced FP64
throughput (often 1/32 or 1/64 of FP32), so `precision='float32'` is offered
and its numerical cost is MEASURED by `precision_study()` rather than assumed
acceptable. Do not switch to float32 without reading that output: the Bloch
equation is stiff in the ratio gamma_e*B/R_tot and single precision loses
roughly 7 significant digits, which matters when the target signal is ~1e-6
of the ambient field.

VALIDATION STATUS
-----------------
The CPU path is fully validated (see serfmad.validation). The CUDA path was
authored without access to NVIDIA hardware, so its numerical agreement with
the CPU reference must be confirmed on the target machine before use. Run:

    python -m serfmad.gpu --verify      # CPU/GPU agreement to 1e-12
    python -m serfmad.gpu --scaling     # measured speedup curve
    python -m serfmad.gpu --precision   # float32 vs float64 error
    python -m serfmad.gpu --stress      # push VRAM to its limit

`verify_against_cpu()` is the gate: if it fails, do not trust GPU results.
"""

from __future__ import annotations
import time
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List

from .backend import xp, GPU_AVAILABLE, GPU_NAME, to_host, to_device, sync
from .constants import GAMMA_E, TARGETS, MU0_OVER_4PI
from .quantum.bloch import BlochConfig, steady_state_analytic

# ---------------------------------------------------------------------------
# Fused CUDA kernel for the Bloch RK4 step
# ---------------------------------------------------------------------------
# One thread handles one ensemble member for one timestep. All four RK4 stages
# are computed in registers, so global memory is touched exactly twice (read P,
# write P) instead of ~20 times as it would be with chained array expressions.
#
# The magnetic field is passed as four (N,3) stage arrays because B may vary
# between RK4 stage times; precomputing them on the host side of the launch
# keeps the kernel branch-free.
# The body is written against a REAL typedef so one source serves both
# precisions. Every literal is cast to REAL, so a float32 build stays in
# single precision throughout instead of silently promoting to double on
# contact with an untyped constant. The float64 expansion is token-for-token
# what it was before templating -- same operations, same order, so float64
# results are bit-identical.
_BLOCH_RK4_TEMPLATE = r'''
extern "C" __global__
void bloch_rk4_step(
    REAL* __restrict__ P,          // (N,3) state, updated in place
    const REAL* __restrict__ B1,   // (N,3) field at t
    const REAL* __restrict__ B2,   // (N,3) field at t+dt/2
    const REAL* __restrict__ B3,   // (N,3) field at t+dt/2
    const REAL* __restrict__ B4,   // (N,3) field at t+dt
    const REAL dt,
    const REAL gamma_e,
    const REAL R_op,
    const REAL R1,
    const REAL R2,
    const REAL inv_q,
    const REAL sx, const REAL sy, const REAL sz,
    const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int j = 3*i;

    REAL px = P[j], py = P[j+1], pz = P[j+2];

    // --- RK4 stage macro, expanded manually to keep everything in registers
    #define RHS(PX,PY,PZ,BB,KX,KY,KZ) {                       \
        REAL bx = BB[j], by = BB[j+1], bz = BB[j+2];          \
        REAL cx = (PY)*bz - (PZ)*by;                          \
        REAL cy = (PZ)*bx - (PX)*bz;                          \
        REAL cz = (PX)*by - (PY)*bx;                          \
        KX = inv_q * (gamma_e*cx + R_op*sx - R2*(PX));        \
        KY = inv_q * (gamma_e*cy + R_op*sy - R2*(PY));        \
        KZ = inv_q * (gamma_e*cz + R_op*sz - R1*(PZ));        \
    }

    REAL k1x,k1y,k1z, k2x,k2y,k2z, k3x,k3y,k3z, k4x,k4y,k4z;

    const REAL half  = (REAL)0.5;
    const REAL two   = (REAL)2.0;
    const REAL sixth = dt / (REAL)6.0;

    RHS(px, py, pz, B1, k1x,k1y,k1z)
    RHS(px+half*dt*k1x, py+half*dt*k1y, pz+half*dt*k1z, B2, k2x,k2y,k2z)
    RHS(px+half*dt*k2x, py+half*dt*k2y, pz+half*dt*k2z, B3, k3x,k3y,k3z)
    RHS(px+dt*k3x,      py+dt*k3y,      pz+dt*k3z,      B4, k4x,k4y,k4z)

    #undef RHS

    P[j]   = px + sixth*(k1x + two*k2x + two*k3x + k4x);
    P[j+1] = py + sixth*(k1y + two*k2y + two*k3y + k4y);
    P[j+2] = pz + sixth*(k1z + two*k2z + two*k3z + k4z);
}
'''

_kernel_cache: Dict[str, Any] = {}


def _get_bloch_kernel(precision: str = "float64"):
    """
    Compile (once, per precision) and return the fused RK4 kernel.

    Returns None on the CPU backend. Kernels are cached per precision because
    NVRTC compilation is expensive and happens on first use.
    """
    if not GPU_AVAILABLE:
        return None
    key = f"bloch_rk4_{precision}"
    if key not in _kernel_cache:
        real = "float" if precision == "float32" else "double"
        src = f"typedef {real} REAL;\n{_BLOCH_RK4_TEMPLATE}"
        _kernel_cache[key] = xp.RawKernel(src, "bloch_rk4_step")
    return _kernel_cache[key]


# ---------------------------------------------------------------------------
# Device capability / memory management
# ---------------------------------------------------------------------------

def device_info() -> dict:
    """Query the active CUDA device. Safe to call on CPU-only systems."""
    if not GPU_AVAILABLE:
        return {"available": False, "name": "none",
                "note": "CPU backend active; install cupy-cuda12x for GPU"}
    try:
        dev = xp.cuda.Device()
        props = xp.cuda.runtime.getDeviceProperties(dev.id)
        free_b, total_b = xp.cuda.runtime.memGetInfo()
        name = props["name"]
        name = name.decode() if isinstance(name, bytes) else str(name)
        return {
            "available": True,
            "name": name,
            "compute_capability": f"{props['major']}.{props['minor']}",
            "multiprocessors": props["multiProcessorCount"],
            "total_memory_GB": total_b / 1e9,
            "free_memory_GB": free_b / 1e9,
            "max_threads_per_block": props["maxThreadsPerBlock"],
            "clock_MHz": props.get("clockRate", 0) / 1000.0,
        }
    except Exception as exc:
        return {"available": True, "name": GPU_NAME,
                "note": f"property query failed: {exc}"}


def max_ensemble_for_memory(n_time: int, safety: float = 0.6,
                            dtype_bytes: int = 8) -> int:
    """
    Largest ensemble that fits in free VRAM for a batched-FFT workload.

    Budget assumes the peak resident set is roughly six (N,n_time) arrays:
    the signal, the noise, the correlation output, and FFT scratch. `safety`
    leaves headroom for fragmentation and the cuFFT plan cache -- running a
    GPU to 100% of reported free memory reliably fails in practice.
    """
    if not GPU_AVAILABLE:
        return 100_000
    try:
        free_b, _ = xp.cuda.runtime.memGetInfo()
    except Exception:
        return 100_000
    per_traj = 6 * n_time * dtype_bytes
    return max(1024, int(free_b * safety / per_traj))


def recommend_backend(n_ensemble: int, n_time: int) -> dict:
    """
    Honest guidance on whether this problem should use the GPU at all.

    Returns a recommendation plus the reasoning, so the caller can see WHY
    rather than trusting a bare boolean.
    """
    work = n_ensemble * n_time
    if not GPU_AVAILABLE:
        return {"use_gpu": False, "reason": "no CUDA device present"}
    if n_ensemble < 4096:
        return {"use_gpu": False,
                "reason": f"ensemble {n_ensemble} below launch-overhead "
                          f"break-even (~4096); CPU will be faster"}
    if work < 1e6:
        return {"use_gpu": False,
                "reason": f"total work {work:.1e} element-steps is too small "
                          f"to amortise host/device transfer"}
    cap = max_ensemble_for_memory(n_time)
    return {"use_gpu": True,
            "reason": f"work {work:.1e} element-steps is SIMD-parallel",
            "chunking_required": n_ensemble > cap,
            "max_ensemble_per_chunk": cap,
            "n_chunks": max(1, math.ceil(n_ensemble / cap))}


# ---------------------------------------------------------------------------
# Large-scale ensemble Bloch integration
# ---------------------------------------------------------------------------

def integrate_ensemble_gpu(
    t: np.ndarray,
    B_stages: Callable[[float, float], Any],
    cfg: Optional[BlochConfig] = None,
    n_ensemble: int = 1_000_000,
    precision: str = "float64",
    use_custom_kernel: bool = True,
    record_every: int = 0,
) -> dict:
    """
    Integrate N spin ensembles with the fused CUDA kernel.

    Parameters
    ----------
    t : uniform time grid (s)
    B_stages : callable (t, dt) -> tuple of four (N,3) device arrays giving the
        field at the four RK4 stage times. Supplying all four explicitly keeps
        the kernel branch-free and lets the caller model genuinely
        time-dependent fields without the kernel needing to know their form.
    n_ensemble : number of parallel replicas
    precision : 'float64' (default, correct) or 'float32' (fast, lossy --
        see precision_study before using)
    record_every : 0 = keep only the final state (minimal memory); k>0 stores
        every k-th sample, which is what dominates memory for long runs.

    Returns dict with final state, optional history, and timing.
    """
    cfg = cfg or BlochConfig()
    t = np.asarray(t, dtype=float)
    dt = float(t[1] - t[0])
    n_steps = len(t)
    dtype = xp.float32 if precision == "float32" else xp.float64

    P = xp.zeros((n_ensemble, 3), dtype=dtype)
    P[:, 0] = cfg.pump_axis[0] * cfg.P_equilibrium
    P[:, 1] = cfg.pump_axis[1] * cfg.P_equilibrium
    P[:, 2] = cfg.pump_axis[2] * cfg.P_equilibrium

    hist = None
    if record_every > 0:
        n_rec = (n_steps + record_every - 1) // record_every
        hist = xp.zeros((n_rec, n_ensemble, 3), dtype=dtype)

    kernel = _get_bloch_kernel(precision) if use_custom_kernel else None
    # RawKernel maps a Python float to C double. Passing one to a float32
    # kernel would misread every scalar argument, so scalars are cast to
    # match REAL. float64 keeps plain Python floats, exactly as before.
    _scalar = (lambda v: np.float32(v)) if precision == "float32" else float
    threads = 256
    blocks = (n_ensemble + threads - 1) // threads

    sync()
    t0 = time.perf_counter()
    rec = 0
    for i in range(n_steps):
        if record_every > 0 and i % record_every == 0:
            hist[rec] = P
            rec += 1
        ti = t[i]
        B1, B2, B3, B4 = B_stages(ti, dt)
        if kernel is not None:
            # A field stage in the wrong dtype would be reinterpreted rather
            # than converted. asarray is a no-op when it already matches,
            # which is the normal path.
            B1 = xp.asarray(B1, dtype=dtype); B2 = xp.asarray(B2, dtype=dtype)
            B3 = xp.asarray(B3, dtype=dtype); B4 = xp.asarray(B4, dtype=dtype)
            kernel((blocks,), (threads,),
                   (P, B1, B2, B3, B4, _scalar(dt), _scalar(cfg.gamma_e),
                    _scalar(cfg.R_op), _scalar(cfg.R_1), _scalar(cfg.R_2),
                    _scalar(1.0 / cfg.q),
                    _scalar(cfg.pump_axis[0]), _scalar(cfg.pump_axis[1]),
                    _scalar(cfg.pump_axis[2]), n_ensemble))
        else:
            # Portable fallback: identical maths via array expressions.
            from .quantum.bloch import bloch_rhs
            k1 = bloch_rhs(P, B1, cfg)
            k2 = bloch_rhs(P + 0.5 * dt * k1, B2, cfg)
            k3 = bloch_rhs(P + 0.5 * dt * k2, B3, cfg)
            k4 = bloch_rhs(P + dt * k3, B4, cfg)
            P = P + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    sync()
    wall = time.perf_counter() - t0

    return {
        "P_final": to_host(P),
        "history": to_host(hist) if hist is not None else None,
        "wall_seconds": wall,
        "traj_steps_per_second": n_ensemble * n_steps / wall if wall > 0 else float("inf"),
        "used_custom_kernel": kernel is not None,
        "precision": precision,
        "n_ensemble": n_ensemble,
        "n_steps": n_steps,
    }


def _constant_field_stages(Bx: float, n_ensemble: int, dtype):
    """Helper: a static transverse field, same at all four RK4 stages."""
    B = xp.zeros((n_ensemble, 3), dtype=dtype)
    B[:, 0] = Bx
    return lambda ti, dt: (B, B, B, B)


# ---------------------------------------------------------------------------
# Verification gate
# ---------------------------------------------------------------------------

def verify_against_cpu(n_ensemble: int = 8192, tol: float = 1e-10) -> dict:
    """
    THE GATE. Confirms the GPU path reproduces the analytic steady state.

    The Bloch equation has an exact closed-form fixed point (bloch.steady_state
    _analytic), so this is an absolute check against mathematics, not merely a
    CPU/GPU consistency check -- a bug present in both paths could not hide.

    Run this on the target machine before trusting any GPU result.
    """
    cfg = BlochConfig()
    Bx = 1e-9
    dtype = xp.float64
    stages = _constant_field_stages(Bx, n_ensemble, dtype)

    # Integrate far past the relaxation time so the state IS the fixed point.
    # The transient decays as exp(-t*R_1/q), so 32 time constants leaves a
    # residual of ~1e-14 -- below the tolerance, ensuring this test measures
    # integrator accuracy rather than an un-decayed transient.
    T = 32.0 * cfg.q / cfg.R_1
    t = np.linspace(0.0, T, 6000)

    res = integrate_ensemble_gpu(t, stages, cfg, n_ensemble=n_ensemble,
                                 use_custom_kernel=True)
    P_gpu = res["P_final"][0]

    ss = steady_state_analytic(Bx, cfg)
    P_exact = np.array([float(ss["Px"]), float(ss["Py"]), float(ss["Pz"])])
    err = float(np.abs(P_gpu - P_exact).max())

    # also confirm every replica is identical (no race conditions)
    spread = float(np.abs(res["P_final"] - res["P_final"][0]).max())

    ok = err < tol and spread < 1e-14
    return {
        "passed": ok,
        "max_abs_error_vs_analytic": err,
        "inter_replica_spread": spread,
        "tolerance": tol,
        "P_computed": P_gpu.tolist(),
        "P_analytic": P_exact.tolist(),
        "used_custom_kernel": res["used_custom_kernel"],
        "backend": "cupy" if GPU_AVAILABLE else "numpy",
        "note": "spread > 0 would indicate a race condition in the kernel",
    }


def precision_study(n_ensemble: int = 65536) -> dict:
    """
    Measure, do not assume, the cost of float32.

    Consumer GeForce parts run FP64 at a small fraction of FP32 throughput, so
    the temptation to drop precision is strong. This reports both the speedup
    and the accuracy penalty so the trade is made on evidence.
    """
    cfg = BlochConfig()
    Bx = 1e-9
    T = 32.0 * cfg.q / cfg.R_1
    t = np.linspace(0.0, T, 2000)
    ss = steady_state_analytic(Bx, cfg)
    P_exact = np.array([float(ss["Px"]), float(ss["Py"]), float(ss["Pz"])])

    out = {}
    # CuPy compiles kernels at run time (NVRTC), on first use, per dtype and
    # per kernel source. Timing the first call for each precision charges that
    # compilation to the measurement; discarding one run first roughly halved
    # the distortion here (speedup_float32 0.079 -> 0.144 on an RTX 3050).
    # Eight steps compiles every kernel the real run uses and costs nothing.
    #
    # Both precisions now run the SAME fused kernel, so this compares two
    # precisions rather than two algorithms. It previously did the latter --
    # the kernel was float64-only, so float32 fell back to the elementwise
    # path (~20 global memory touches per step against one) and reported as
    # slower than float64, which inverted the very trade this function exists
    # to measure.
    t_warm = t[:8]
    for prec, dt_ in (("float64", xp.float64), ("float32", xp.float32)):
        stages = _constant_field_stages(Bx, n_ensemble, dt_)
        integrate_ensemble_gpu(t_warm, stages, cfg, n_ensemble=n_ensemble,
                               precision=prec, use_custom_kernel=True)
        sync()
        r = integrate_ensemble_gpu(t, stages, cfg, n_ensemble=n_ensemble,
                                   precision=prec, use_custom_kernel=True)
        err = float(np.abs(r["P_final"][0].astype(np.float64) - P_exact).max())
        rel = err / abs(P_exact[1])
        out[prec] = {"wall_seconds": r["wall_seconds"],
                     "abs_error": err, "rel_error_on_Py": rel,
                     "traj_steps_per_second": r["traj_steps_per_second"]}
    if out["float32"]["wall_seconds"] > 0:
        out["speedup_float32"] = (out["float64"]["wall_seconds"] /
                                  out["float32"]["wall_seconds"])
    out["accuracy_cost"] = (out["float32"]["rel_error_on_Py"] /
                            max(out["float64"]["rel_error_on_Py"], 1e-18))
    out["recommendation"] = (
        "float64 for any published result; float32 acceptable only for "
        "qualitative exploration where relative error above 1e-6 is tolerable")
    return out


# ---------------------------------------------------------------------------
# The scientific payload: multi-dimensional parameter sweep
# ---------------------------------------------------------------------------

def parameter_sweep(
    target_label: str = "Large UXO / naval mine",
    n_range: int = 48,
    n_moment_orient: int = 64,
    n_noise: int = 64,
    range_min_m: float = 2.0,
    range_max_m: float = 300.0,
    n_time: int = 512,
    speed_mps: float = 2.0,
    baseline_m: float = 0.5,
    sensor_noise_T: float = 5e-15 * np.sqrt(10.0),
    clutter_residual_T: float = 2.23e-9,
    cmrr_db: float = 60.0,
    pfa: float = 1e-4,
    seed: int = 0,
    chunk: Optional[int] = None,
) -> dict:
    """
    Full-factorial sweep over range x moment orientation x noise realisation.

    Total trajectories = n_range * n_moment_orient * n_noise. Defaults give
    48*64*64 = 196,608 transits; raising the factors to 96 x 128 x 128 gives
    1.57 million, which is the intended regime for a real GPU.

    This is the module's scientific payload: it produces detection probability
    as a function of range MARGINALISED over target orientation, which is the
    operationally meaningful quantity (an adversary's dipole orientation is
    never known a priori, so a Pd quoted at one fixed orientation is not a
    usable number).

    Memory is managed by chunking over the range axis, so the sweep size is
    limited by time, not VRAM.
    """
    rng = np.random.default_rng(seed)
    tgt = TARGETS[target_label]
    m_mag = tgt.moment_A_m2

    ranges = np.linspace(range_min_m, range_max_m, n_range)
    # Orientations sampled uniformly on the sphere (cos(theta) uniform, NOT
    # theta uniform -- the latter over-samples the poles and would bias Pd).
    cos_th = rng.uniform(-1.0, 1.0, n_moment_orient)
    theta = np.arccos(cos_th)
    phi = rng.uniform(0.0, 2 * np.pi, n_moment_orient)

    t = np.linspace(-15.0, 15.0, n_time)
    dt = float(t[1] - t[0])
    T_dev = to_device(t)[None, :]

    # unit-energy bipolar matched-filter template (device-resident, reused)
    u = T_dev / 3.0
    tmpl = -u * xp.exp(-0.5 * u ** 2)
    tmpl = tmpl - tmpl.mean()
    tmpl = tmpl / xp.sqrt((tmpl ** 2).sum())
    F_tmpl = xp.conj(xp.fft.rfft(tmpl, axis=1))

    shift = max(int(round(baseline_m / (speed_mps * dt))), 1)
    noise_grad = sensor_noise_T * np.sqrt(2) / baseline_m
    cmrr_lin = 10 ** (cmrr_db / 20.0)
    clutter_grad = (clutter_residual_T / cmrr_lin) / baseline_m

    if chunk is None:
        cap = max_ensemble_for_memory(n_time)
        chunk = max(1, min(n_range, max(1, cap // max(1, n_moment_orient * n_noise))))

    def _noise_block(nrows):
        """White sensor noise + 1/f clutter, matching the mission model."""
        w = to_device(rng.normal(0.0, noise_grad, (nrows, n_time)))
        nf = n_time // 2 + 1
        f = xp.arange(nf, dtype=xp.float64)
        f[0] = 1.0
        ph = to_device(rng.uniform(0, 2 * np.pi, (nrows, nf)))
        spec = (1.0 / xp.sqrt(f))[None, :] * (xp.cos(ph) + 1j * xp.sin(ph))
        c = xp.fft.irfft(spec, n=n_time, axis=1)
        c = c / xp.std(c, axis=1, keepdims=True)
        return w + c * clutter_grad

    # --- null distribution (noise only) for the CFAR threshold -------------
    n_null = min(200_000, max(20_000, n_moment_orient * n_noise * 4))
    null_stats = []
    step = max(1, min(n_null, max_ensemble_for_memory(n_time)))
    done = 0
    while done < n_null:
        nb = min(step, n_null - done)
        nz = _noise_block(nb)
        nz = nz - nz.mean(axis=1, keepdims=True)
        corr = xp.fft.irfft(xp.fft.rfft(nz, axis=1) * F_tmpl, n=n_time, axis=1)
        null_stats.append(to_host(xp.abs(corr).max(axis=1)))
        done += nb
    null_all = np.concatenate(null_stats)
    thr = float(np.quantile(null_all, 1.0 - pfa))

    # --- signal sweep, chunked over range ---------------------------------
    pd_vs_range = np.zeros(n_range)
    peak_signal = np.zeros(n_range)
    sync()
    t0 = time.perf_counter()

    for i0 in range(0, n_range, chunk):
        i1 = min(i0 + chunk, n_range)
        nr = i1 - i0
        R = to_device(ranges[i0:i1])[:, None, None]          # (nr,1,1)
        TH = to_device(theta)[None, :, None]                 # (1,no,1)
        PH = to_device(phi)[None, :, None]
        Tt = T_dev[None, :, :]                               # (1,1,M) -> broadcast

        x = speed_mps * Tt                                   # along-track (1,1,M)
        r = xp.sqrt(x ** 2 + R ** 2)                         # (nr,no,M)
        mx = m_mag * xp.sin(TH) * xp.cos(PH)
        mz = m_mag * xp.cos(TH)
        nx = x / r
        nz = R / r
        m_dot_n = mx * nx + mz * nz
        Bx = MU0_OVER_4PI * (3.0 * nx * m_dot_n - mx) / r ** 3

        # along-track gradiometer difference
        Bs = xp.roll(Bx, -shift, axis=2)
        grad = (Bs - Bx) / baseline_m
        grad[:, :, -shift:] = 0.0
        peak_signal[i0:i1] = to_host(xp.abs(grad).max(axis=(1, 2)))

        # replicate each (range,orientation) across noise realisations
        g = xp.repeat(grad.reshape(nr * n_moment_orient, n_time),
                      n_noise, axis=0)
        obs = g + _noise_block(g.shape[0])
        obs = obs - obs.mean(axis=1, keepdims=True)
        corr = xp.fft.irfft(xp.fft.rfft(obs, axis=1) * F_tmpl, n=n_time, axis=1)
        stat = to_host(xp.abs(corr).max(axis=1))
        det = (stat > thr).reshape(nr, n_moment_orient * n_noise)
        pd_vs_range[i0:i1] = det.mean(axis=1)

        del R, TH, PH, x, r, nx, nz, m_dot_n, Bx, Bs, grad, g, obs, corr
        if GPU_AVAILABLE:
            xp.get_default_memory_pool().free_all_blocks()

    sync()
    wall = time.perf_counter() - t0

    # R50 by linear interpolation on the first downward crossing of 0.5
    r50 = float("nan")
    below = np.where(pd_vs_range < 0.5)[0]
    if len(below) and below[0] > 0:
        i = below[0]
        x0, x1 = ranges[i - 1], ranges[i]
        y0, y1 = pd_vs_range[i - 1], pd_vs_range[i]
        r50 = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else float(x0)

    total = n_range * n_moment_orient * n_noise
    return {
        "ranges": ranges, "pd_vs_range": pd_vs_range, "r50_m": r50,
        "peak_signal_T_per_m": peak_signal, "threshold": thr,
        "n_trajectories": total, "wall_seconds": wall,
        "trajectories_per_second": total / wall if wall > 0 else float("inf"),
        "chunk_size": chunk, "target": target_label, "pfa": pfa,
        "backend": "cupy" if GPU_AVAILABLE else "numpy",
        "marginalised_over": "target dipole orientation (uniform on sphere)",
    }


def scaling_study(sizes: Optional[List[int]] = None, n_time: int = 512) -> dict:
    """
    Measured throughput versus ensemble size.

    Reports trajectory-steps/s so CPU and GPU runs are directly comparable,
    and identifies the size at which throughput saturates -- below that the
    device is launch-latency bound, above it bandwidth bound.
    """
    sizes = sizes or [1024, 4096, 16384, 65536, 262144, 1048576]
    cfg = BlochConfig()
    t = np.linspace(0.0, 0.05, 64)
    rows = []
    for n in sizes:
        try:
            stages = _constant_field_stages(1e-9, n, xp.float64)
            r = integrate_ensemble_gpu(t, stages, cfg, n_ensemble=n)
            rows.append({"n_ensemble": n,
                         "wall_seconds": r["wall_seconds"],
                         "traj_steps_per_second": r["traj_steps_per_second"],
                         "used_custom_kernel": r["used_custom_kernel"]})
        except Exception as exc:
            rows.append({"n_ensemble": n, "error": f"{type(exc).__name__}: {exc}"})
            break
    valid = [r for r in rows if "error" not in r]
    best = max(valid, key=lambda r: r["traj_steps_per_second"]) if valid else None
    return {"backend": "cupy" if GPU_AVAILABLE else "numpy",
            "device": device_info().get("name", "cpu"),
            "rows": rows,
            "peak_throughput": best["traj_steps_per_second"] if best else None,
            "peak_at_n": best["n_ensemble"] if best else None}


def vram_stress(n_time: int = 512) -> dict:
    """
    Deliberately approach the VRAM limit to confirm chunking works and that
    failure, if it comes, is graceful rather than a hard crash mid-sweep.
    """
    cap = max_ensemble_for_memory(n_time)
    attempts = [cap // 8, cap // 4, cap // 2, cap]
    rows = []
    for n in attempts:
        try:
            a = xp.zeros((n, n_time), dtype=xp.float64)
            b = xp.fft.rfft(a, axis=1)
            sync()
            rows.append({"n": n, "GB": a.nbytes / 1e9, "status": "ok"})
            del a, b
            if GPU_AVAILABLE:
                xp.get_default_memory_pool().free_all_blocks()
        except Exception as exc:
            rows.append({"n": n, "status": f"failed: {type(exc).__name__}"})
            break
    return {"estimated_capacity": cap, "attempts": rows,
            "note": "chunking in parameter_sweep keeps sweeps below this line"}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="SERF-MAD GPU compute paths")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--precision", action="store_true")
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--big", action="store_true",
                    help="large sweep: 96 x 128 x 128 = 1.57M trajectories")
    a = ap.parse_args()
    ran = False

    if a.info or not any(vars(a).values()):
        print("=== device ===")
        for k, v in device_info().items():
            print(f"  {k:22s}: {v}")
        print("\n=== backend recommendation ===")
        for k, v in recommend_backend(1_000_000, 512).items():
            print(f"  {k:22s}: {v}")
        ran = True
    if a.verify:
        print("\n=== VERIFICATION GATE (must pass before trusting GPU) ===")
        r = verify_against_cpu()
        for k, v in r.items():
            print(f"  {k:28s}: {v}")
        print("  RESULT:", "PASS" if r["passed"] else "FAIL")
        ran = True
    if a.scaling:
        print("\n=== scaling study ===")
        s = scaling_study()
        print(f"  device: {s['device']} ({s['backend']})")
        for row in s["rows"]:
            print("   ", row)
        print(f"  peak {s['peak_throughput']:.3e} traj-steps/s at N={s['peak_at_n']}"
              if s["peak_throughput"] else "  no valid rows")
        ran = True
    if a.precision:
        print("\n=== precision study ===")
        print(json.dumps(precision_study(), indent=2, default=str))
        ran = True
    if a.stress:
        print("\n=== VRAM stress ===")
        print(json.dumps(vram_stress(), indent=2, default=str))
        ran = True
    if a.sweep or a.big:
        kw = dict(n_range=96, n_moment_orient=128, n_noise=128) if a.big else {}
        print(f"\n=== parameter sweep {'(LARGE)' if a.big else ''} ===")
        s = parameter_sweep(**kw)
        print(f"  trajectories : {s['n_trajectories']:,}")
        print(f"  wall time    : {s['wall_seconds']:.2f} s")
        print(f"  throughput   : {s['trajectories_per_second']:,.0f} traj/s")
        print(f"  R50          : {s['r50_m']:.2f} m")
        print(f"  marginalised : {s['marginalised_over']}")
        ran = True
