# SERF-MAD Simulator

Quantum-physics-based feasibility simulation of a ruggedised SERF (Spin-Exchange
Relaxation-Free) MEMS vapour-cell magnetometer used for Magnetic Anomaly
Detection (MAD) from an Autonomous Underwater Vehicle.

## Quick start

```bash
pip install -r requirements.txt
python main.py              # launch GUI viewer
python main.py --vv         # run the 17-test V&V suite
python main.py --figures    # generate all report figures headlessly
python main.py --vv         # 17 verification & validation tests
python main.py --stress     # 10 stress / robustness tests
python main.py --backend    # check GPU/CPU backend status
```

## GPU quick start (CUDA)

```bash
pip install cupy-cuda12x            # match your CUDA toolkit
python main.py --gpu-verify         # GATE: must pass before trusting GPU output
python main.py --gpu-scaling        # measured throughput vs ensemble size
python -m serfmad.gpu --precision   # float32 vs float64 accuracy/speed trade
python -m serfmad.gpu --stress      # push VRAM to its limit
python main.py --gpu-sweep          # 1.57M-trajectory full-factorial sweep
```

`--gpu-verify` is the gate. It integrates a large ensemble and compares against
the closed-form Bloch fixed point, so it checks the CUDA kernel against
*mathematics*, not merely against the CPU path. It also confirms every replica
is bit-identical, which would expose a race condition. If it fails, do not use
GPU results.

## GPU acceleration

The package uses a dual backend: **CuPy (NVIDIA CUDA)** when available,
**NumPy** otherwise, selected at import time in `serfmad/backend.py`.

> **HARDWARE VALIDATION STATUS.** The CUDA path **has been validated on
> hardware** (RTX 3050 Laptop, CC 8.6, CUDA 12.4, cupy-cuda12x 14.1.1,
> 2026-08-16): `--gpu-verify` passes at 9.99e-15 against the closed-form
> solution with zero inter-replica spread, and `--vv` / `--stress` output is
> byte-identical to the CPU backend. Not yet validated: non-Ampere parts,
> multi-GPU, or float32 for published results. Run `python main.py
> --gpu-verify` on your own machine first regardless — it is a hard gate, and
> everything downstream depends on it passing.

**CUDA toolkit discovery.** CuPy JIT-compiles kernels at run time, so it needs
the toolkit *headers*, not just a driver. With `CUDA_PATH` unset it imports,
enumerates the device and allocates — then fails on the first arithmetic op,
which looks exactly like "no GPU". `serfmad/backend.py` therefore locates a
toolkit itself (matching your CuPy wheel's CUDA major version) before importing
CuPy. Set `CUDA_PATH` explicitly to override.

GPU is used only where it is genuinely warranted — the Monte Carlo detection
sweep (`serfmad.mission.monte_carlo_detection`), which runs N independent
transits sharing one time grid. `should_use_gpu()` returns `False` below
`GPU_MIN_ENSEMBLE = 4096` because kernel-launch and transfer overhead dominate
for small batches. A single Bloch trajectory is *not* a GPU workload and the
code does not pretend otherwise.

## Architecture

```
main.py                     CLI / GUI entry point
serfmad/
  constants.py              literature-cited constants; EngineeringAssumption
                            is a distinct type so modelling choices can never
                            be mistaken for measured data
  backend.py                CuPy/NumPy abstraction + honest capability report
  mission.py                end-to-end transit sim + GPU Monte Carlo
  quantum/
    lindblad.py             QuTiP master equation -- REFERENCE TRUTH
    bloch.py                semiclassical SERF Bloch solver -- PRODUCTION
  physics/
    dipole.py               vector dipole field + analytic gradient tensor
    platform.py             AUV kinematics, ambient nulling chain, CAPE
  processing/
    compensation.py         Tolles-Lawson + reference-channel cancellation
    detection.py            matched filter, CFAR, ROC
  validation/vv_suite.py    17 verification & validation tests
  gpu.py                    fused CUDA RK4 kernel, chunked sweeps,
                            scaling / precision / VRAM studies
  gui/
    app.py                  Tkinter viewer (5 tabs)
    plots.py                pure plotting fns shared by GUI and report
  validation/
    vv_suite.py             17 verification & validation tests
    stress.py               10 stress / robustness tests
```

The GUI and the report use the **same** plotting functions, so a figure in the
report cannot drift from what the operator sees.

## Two headline findings

1. **Meeting the nominal SERF field window is necessary but not sufficient.**
   At exactly 5 nT residual, Cross-Axis Projection Error tilts the sensing axis
   58.8° and costs 48% of the gain. The derived requirement is **≤0.94 nT**
   for ≤5% loss — ~53,000× (94 dB) suppression, not the naive 10,000×.

2. **The system is clutter-limited, not sensor-limited.** A 5 fT/√Hz MEMS cell
   sits ~6 orders of magnitude below the measured post-compensation UUV
   interference residual (2.23 nT, Oehler 2024). Modelling only sensor noise
   returns Pd = 1.0 at every range for every target — physically meaningless.

## What the GPU path actually does

- **Fused CUDA kernel** (`gpu._BLOCH_RK4_SRC`): all four RK4 stages computed in
  registers, so global memory is touched once per timestep instead of ~20 times.
  The workload is bandwidth-bound, so this is where the speedup lives.
- **Batched FFT matched filtering** across the full ensemble.
- **Chunked sweeps** sized against free VRAM, so sweep size is limited by time
  rather than memory. `--gpu-sweep` runs 96 x 128 x 128 = 1,572,864 transits.
- **Orientation marginalisation**: dipole orientation is sampled uniformly on
  the sphere (uniform in cos-theta, not in theta, which would bias Pd toward
  the poles). An adversary's orientation is never known, so a Pd quoted at one
  fixed orientation is not a usable number.
- **Honest sizing**: `recommend_backend()` returns *"use the CPU"* below ~4096
  ensemble members, because launch and transfer overhead dominate there. A
  single Bloch trajectory is not a GPU workload at any size.

## Figure style

All figures — GUI tabs and report images alike — are produced by the same
functions in `serfmad/gui/plots.py` and `serfmad/gui/hero.py`, styled by
`serfmad/gui/style.py`. A published figure therefore cannot drift from what
the operator sees on screen.

**LaTeX typesetting.** The style uses real LaTeX (Computer Modern) so figure
maths matches the report body. This needs:

```bash
sudo apt-get install cm-super dvipng texlive-fonts-recommended
```

Without `cm-super` (which supplies `type1ec.sty`) Matplotlib's LaTeX call fails
and **no file is written** — a silent failure. `style.use_style()` probes for a
working pipeline and falls back to mathtext with a serif face, so it always
works; `latex_available()` reports which is active.

The **GUI deliberately uses mathtext even when LaTeX is installed**. Real
`usetex` shells out to `latex.exe` once per label, which measured 25–28 s per
tab here and froze the window. Report export re-detects independently and
still uses real LaTeX, so published figures are unaffected — the difference is
the glyph renderer, never the data. Tab renders also run on a background
thread, so the window stays responsive (worst measured UI block: 1.4 s).

The GUI's **Surfaces** tab shows matte-shaded 3D surfaces for quantities that
genuinely live on a 2D parameter plane (sensor response over (Bx,By), anomaly
field over the survey plane, CAPE penalty, detection performance). 3D is never
used to decorate a line plot.

## Surface rendering performance

Matplotlib's 3D renderer depth-sorts every quad in Python, so wall time scales
with cell **count**, not data resolution. Measured at 100 dpi:

| plotted mesh | quads | time | mean pixel diff vs finest |
|---|---|---|---|
| 520x520 | 270k | 28.6 s | reference |
| 260x260 | 68k | 7.4 s | 0.14% |
| 175x175 | 30k | 3.3 s | 0.24% |
| 104x104 | 11k | 1.2 s | 0.40% |

`matte_surface(..., max_cells=N)` decimates the plotted mesh accordingly.
The GUI uses `quality="interactive"` (**3.7x faster**, worst-case pixel
difference under 1%); report export uses `quality="publication"`.

### Why not pyqtgraph/OpenGL?

Evaluated and **not adopted**. Two reasons, both checked rather than assumed:

1. `pyqtgraph.opengl` could not create a GL context in the development
   environment (`QOpenGLWidget: Failed to create context`; no image file was
   produced at all), so the "must look identical" requirement could not be
   verified — and shipping an unverified renderer is exactly the failure mode
   this project exists to avoid.
2. `GLSurfacePlotItem` shades per-vertex through a GL shader. The matte look
   here comes from `LightSource.shade()`, a CPU relief-shading blend of a
   colormap with topographic lighting. These are different algorithms; the
   output would be visibly different, not identical.

If you want to pursue it on your hardware, the honest path is a separate
renderer behind the same call signature, compared against the matplotlib
output pixel-by-pixel before replacing anything.

## Detection capability envelope

```bash
python main.py --capability          # cached
python main.py --refresh-capability  # recompute
```

Sweeps the operating space per target class and records where detection is
**credible** — SERF lock held, target above the target-free clutter floor, and
the matched filter peaking at closest approach. A bare threshold crossing does
not count: an ungated model reports "detections" driven entirely by
uncancelled platform motion. Cached to `outputs/capability_cache.json`.

| target | moment (A·m²) | max credible depth |
|---|---|---|
| Submarine, undegaussed | 1e7 | 1194 m* |
| Submarine, degaussed | 5e5 | 643 m* |
| Large UXO / naval mine | 500 | 139 m |
| Small UXO / ordnance cluster | 50 | 78 m |
| Micro-UUV, low signature | 5 | 44 m |

\* Beyond the 120 m transit track the signature approaches a DC offset that
real compensation would remove — treat submarine rows as upper bounds.
Small-target rows follow the expected m^(1/4) gradiometer scaling
(78.5/44.1 = 1.78 vs 1.78 predicted).

## Documentation

- `docs/SR_report.pdf` -- 13-page Scientific Report, 46 cited references
- `docs/REFERENCES.md` -- short-key to full-citation map used in code comments
- `docs/PHYSICS_NOTES.md` -- conventions, relaxation anisotropy, numerics

## Measured GPU throughput

RTX 3050 Laptop vs the NumPy path, same trajectories (`--gpu-scaling`):

| ensemble | CPU traj-steps/s | GPU traj-steps/s | speedup |
|---|---|---|---|
| 1,024 | 5.6M | 4.9M | 0.87x |
| 4,096 | 9.9M | 200M | 20x |
| 65,536 | 3.8M | 288M | 76x |
| 1,048,576 | 2.1M | 321M | 150x |

The 1,024 row is *slower* on GPU, which is why `GPU_MIN_ENSEMBLE = 4096`
exists — the heuristic is confirmed by measurement, not assumed.

## Known limitations

- Single-GPU, Ampere-only validation; other architectures untested.
- float32 is ~14x faster than float64 on a GeForce part but carries ~6e-6
  relative error (`python -m serfmad.gpu --precision`) — exploration only.
- Single-spin/ensemble-averaged treatment; no spin-squeezing or atom-number
  shot-noise term, so the model cannot predict an absolute fT/√Hz figure.
- Platform self-field gradient across the baseline is modelled as a finite
  CMRR rather than from vehicle geometry.
