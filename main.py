#!/usr/bin/env python3
"""
SERF-MAD Simulator -- main entry point.

Usage
-----
    python main.py                 launch the GUI viewer
    python main.py --vv            run the V&V suite (text)
    python main.py --figures       generate all report figures headlessly
    python main.py --backend       report GPU/CPU backend status
    python main.py --benchmark     benchmark the active backend
"""
import argparse, sys, os

# ---------------------------------------------------------------------------
# Must run BEFORE any import that pulls Matplotlib (QuTiP imports qutip.bloch,
# which imports Matplotlib). On Windows the first Matplotlib import builds a
# font cache that routinely takes minutes -- and rebuilds on EVERY run if its
# config directory is not writable, which looks exactly like a hang.
# Pinning MPLCONFIGDIR to a writable folder inside the project makes the cost
# a one-time hit, and forcing the Agg backend avoids probing GUI toolkits.
# ---------------------------------------------------------------------------
os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   ".mplcache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
if not any(f in sys.argv for f in ("--gui",)) and len(sys.argv) > 1:
    os.environ.setdefault("MPLBACKEND", "Agg")


# CUDA toolkit discovery lives in serfmad/backend.py, immediately before the
# CuPy import it exists to serve, so `python -m serfmad.gpu` gets it too.


def _warm_matplotlib():
    """Build the font cache once, with a visible message instead of a hang."""
    import time
    cache_flag = os.path.join(os.environ["MPLCONFIGDIR"], ".warmed")
    if os.path.exists(cache_flag):
        return
    print("First run: building the Matplotlib font cache "
          "(one time, can take a few minutes on Windows)...", flush=True)
    t0 = time.perf_counter()
    import matplotlib.font_manager  # noqa: F401  -- triggers the build
    open(cache_flag, "w").write("ok")
    print(f"  done in {time.perf_counter()-t0:.1f}s "
          f"(cached in {os.environ['MPLCONFIGDIR']})", flush=True)


def main():
    _warm_matplotlib()
    ap = argparse.ArgumentParser(description="SERF-MAD Simulator")
    ap.add_argument("--gui", action="store_true",
                    help="launch the GUI viewer (the default with no arguments)")
    ap.add_argument("--vv", action="store_true", help="run V&V suite")
    ap.add_argument("--figures", action="store_true", help="generate figures headlessly")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--backend", action="store_true", help="report backend status")
    ap.add_argument("--benchmark", action="store_true", help="benchmark backend")
    ap.add_argument("--constants", action="store_true", help="print constant set")
    ap.add_argument("--stress", action="store_true", help="run stress-test suite")
    ap.add_argument("--capability", action="store_true",
                    help="detection capability envelope (cached)")
    ap.add_argument("--refresh-capability", action="store_true",
                    help="recompute the capability cache")
    ap.add_argument("--gpu-verify", action="store_true", help="GPU verification gate")
    ap.add_argument("--gpu-scaling", action="store_true", help="GPU scaling study")
    ap.add_argument("--gpu-sweep", action="store_true", help="large GPU parameter sweep")
    a = ap.parse_args()

    if a.backend:
        from serfmad.backend import describe
        for k, v in describe().items():
            print(f"{k:22s}: {v}")
        return
    if a.benchmark:
        from serfmad.backend import benchmark, describe
        print(describe()["detail"])
        for k, v in benchmark().items():
            print(f"  {k:32s}: {v}")
        return
    if a.constants:
        from serfmad.constants import summary_table
        print(summary_table())
        return
    if a.vv:
        from serfmad.validation.vv_suite import run_all, summary
        print("=" * 74)
        print("SERF-MAD VERIFICATION & VALIDATION SUITE")
        print("=" * 74)
        print("loading physics modules (QuTiP first import is slow)...", flush=True)
        res = run_all()
        s = summary(res)
        print("-" * 74)
        print(f"TOTAL {s['total']}  PASS {s['passed']}  FAIL {s['failed']}  SKIP {s['skipped']}")
        print(f"backend: {s['backend']['detail']}")
        sys.exit(1 if s["failed"] else 0)
    if a.capability or a.refresh_capability:
        from serfmad.capability import build, report
        print(report(build(refresh=a.refresh_capability)))
        return
    if a.stress:
        from serfmad.validation.stress import run_all as sr, summary as ss
        print("=" * 78); print("SERF-MAD STRESS TEST SUITE"); print("=" * 78)
        r = sr(); s2 = ss(r); print("-" * 78)
        print(f"TOTAL {s2['total']}  PASS {s2['passed']}  FAIL {s2['failed']}")
        sys.exit(1 if s2["failed"] else 0)
    if a.gpu_verify:
        from serfmad.gpu import verify_against_cpu, device_info
        for k, v in device_info().items(): print(f"  {k:24s}: {v}")
        r = verify_against_cpu()
        print()
        for k, v in r.items(): print(f"  {k:28s}: {v}")
        sys.exit(0 if r["passed"] else 1)
    if a.gpu_scaling:
        from serfmad.gpu import scaling_study
        s3 = scaling_study()
        print(f"device: {s3['device']} ({s3['backend']})")
        for row in s3["rows"]: print("  ", row)
        return
    if a.gpu_sweep:
        from serfmad.gpu import parameter_sweep
        s4 = parameter_sweep(n_range=96, n_moment_orient=128, n_noise=128)
        print(f"trajectories : {s4['n_trajectories']:,}")
        print(f"wall         : {s4['wall_seconds']:.2f} s")
        print(f"throughput   : {s4['trajectories_per_second']:,.0f} traj/s")
        print(f"R50          : {s4['r50_m']:.2f} m")
        return
    if a.figures:
        from serfmad.gui.plots import generate_all
        generate_all(a.outdir)
        return

    from serfmad.gui.app import main as gui_main
    gui_main()

if __name__ == "__main__":
    main()
