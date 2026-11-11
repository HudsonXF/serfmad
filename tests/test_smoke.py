"""Smoke tests -- run with: python -m pytest tests/ (or python tests/test_smoke.py)"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serfmad.validation.vv_suite import run_all, summary
from serfmad.mission import run_mission, MissionConfig, monte_carlo_detection
from serfmad.backend import describe


def test_vv_suite_all_pass():
    res = run_all(verbose=False)
    s = summary(res)
    assert s["failed"] == 0, f"{s['failed']} V&V tests failed"
    assert s["total"] >= 17


def test_mission_runs_and_detects():
    r = run_mission(MissionConfig(target_label="Large UXO / naval mine",
                                  target_depth_m=12.0))
    assert r.detection.detected
    assert abs(r.detection.peak_time - r.cpa_time) < 1.0


def test_monte_carlo_produces_finite_r50():
    mc = monte_carlo_detection(n_trials=2000,
                               target_label="Large UXO / naval mine",
                               range_min_m=2.0, range_max_m=300.0)
    assert np.isfinite(mc["r50_m"]) and mc["r50_m"] > 0


def test_clutter_matters():
    """Omitting clutter must materially overstate detection range."""
    kw = dict(n_trials=2000, target_label="Large UXO / naval mine",
              range_min_m=2.0, range_max_m=300.0)
    with_c = monte_carlo_detection(include_clutter=True, **kw)["r50_m"]
    without = monte_carlo_detection(include_clutter=False, **kw)["r50_m"]
    assert without > with_c


def test_backend_reports_honestly():
    d = describe()
    assert d["hardware_validated"] == d["gpu_available"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"[ OK ] {name}")
    print("all smoke tests passed")


def test_stress_suite_all_pass():
    from serfmad.validation.stress import run_all as sa, summary as ss
    r = sa(verbose=False)
    s = ss(r)
    assert s["failed"] == 0, f"{s['failed']} stress tests failed"


def test_gpu_verification_gate():
    """Passes on CPU backend too -- checks against the analytic fixed point."""
    from serfmad.gpu import verify_against_cpu
    r = verify_against_cpu(n_ensemble=2048)
    assert r["passed"], r


def test_false_detection_is_rejected():
    """A run outside the SERF window must NOT report a credible detection."""
    from serfmad.mission import run_mission, MissionConfig
    r = run_mission(MissionConfig(target_label="Small UXO / ordnance cluster",
                                  target_depth_m=60.0,
                                  passive_shield=5.0, active_null=80.0))
    assert not r.serf_valid, "expected SERF lock loss at 400x suppression"
    assert not r.detection_credible, "clutter-driven peak reported as detection"
    assert r.invalid_reason


def test_nominal_detection_is_credible():
    from serfmad.mission import run_mission, MissionConfig
    r = run_mission(MissionConfig(target_label="Large UXO / naval mine",
                                  target_depth_m=12.0))
    assert r.serf_valid and r.detection_credible
    assert r.signal_to_clutter > 3.0


def test_capability_scaling_is_physical():
    """Max depth must scale as m^(1/4) for a gradiometer (r^-4 signal)."""
    from serfmad.capability import build
    c = build()
    rows = {r["target"]: r for r in c["max_depth"]}
    small = rows["Small UXO / ordnance cluster"]["max_credible_depth_m"]
    micro = rows["Adversary micro-UUV, low-signature hull"]["max_credible_depth_m"]
    assert small and micro
    observed = small / micro
    predicted = 10 ** 0.25          # moments differ by 10x
    assert abs(observed - predicted) / predicted < 0.15, (observed, predicted)
