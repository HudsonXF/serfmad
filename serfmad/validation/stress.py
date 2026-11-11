"""
serfmad.validation.stress
=========================
Stress testing and robustness analysis.

Where `vv_suite` asks "is each component correct?", this module asks
"do the headline results survive being pushed?". It covers:

  S-1  Monte Carlo convergence      -- does R50 stabilise as N grows?
  S-2  Seed stability               -- spread of R50 across RNG seeds
  S-3  Time-grid independence       -- results invariant to sampling rate
  S-4  Parameter sensitivity        -- which inputs actually move the answer
  S-5  Extreme regimes              -- behaviour at the edges of validity
  S-6  Energy/physical bounds       -- |P| <= 1 always, no runaway
  S-7  Detection-theory consistency -- measured Pfa matches design Pfa
  S-8  Literature closure           -- independent cross-checks vs published data

Run:  python -m serfmad.validation.stress
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List

from ..constants import (GAMMA_E, EARTH_FIELD_NOMINAL_T, TARGETS,
                         SERF_RESIDUAL_FIELD_TOLERANCE_T,
                         OEHLER_RAW_INTERFERENCE_T, OEHLER_COMPENSATED_RESIDUAL_T)
from ..quantum.bloch import BlochConfig, integrate, steady_state_analytic
from ..physics.dipole import dipole_field, on_axis_peak_field, detection_range
from ..physics.platform import (AUVPlatform, SEA_STATES, NullingSystem,
                                field_budget, cape_error)
from ..mission import monte_carlo_detection, run_mission, MissionConfig
from ..backend import to_device


@dataclass
class StressResult:
    test_id: str
    name: str
    status: str          # PASS | FAIL | INFO
    metric: str
    criterion: str
    detail: str = ""


# ---------------------------------------------------------------------------
def s1_monte_carlo_convergence() -> StressResult:
    """R50 must converge as the trial count increases, not wander."""
    target = "Large UXO / naval mine"
    r50s = []
    Ns = [1000, 2000, 4000, 8000, 16000]
    for n in Ns:
        m = monte_carlo_detection(n_trials=n, target_label=target,
                                  range_min_m=2.0, range_max_m=300.0, seed=0)
        r50s.append(m["r50_m"])
    r50s = np.array(r50s)
    # relative change between the two largest N
    drift = abs(r50s[-1] - r50s[-2]) / r50s[-1]
    spread = (r50s[2:].max() - r50s[2:].min()) / r50s[2:].mean()
    ok = drift < 0.10 and spread < 0.15
    return StressResult(
        "S-1", "Monte Carlo convergence", "PASS" if ok else "FAIL",
        f"R50 = {[f'{r:.1f}' for r in r50s]} m for N={Ns}",
        "drift < 10% and spread < 15% over top three N",
        f"final drift {drift*100:.2f}%, spread {spread*100:.2f}%")


def s2_seed_stability() -> StressResult:
    """Results must not depend on the RNG seed beyond statistical noise."""
    target = "Large UXO / naval mine"
    r50s = []
    for seed in range(8):
        m = monte_carlo_detection(n_trials=6000, target_label=target,
                                  range_min_m=2.0, range_max_m=300.0, seed=seed)
        r50s.append(m["r50_m"])
    r50s = np.array(r50s)
    cv = r50s.std() / r50s.mean()
    ok = cv < 0.08
    return StressResult(
        "S-2", "Seed stability", "PASS" if ok else "FAIL",
        f"R50 = {r50s.mean():.1f} +/- {r50s.std():.1f} m over 8 seeds (CV {cv*100:.2f}%)",
        "coefficient of variation < 8%",
        f"min {r50s.min():.1f}, max {r50s.max():.1f} m")


def s3_time_grid_independence() -> StressResult:
    """The physical answer must not depend on the sampling rate chosen."""
    peaks, times = [], []
    for n in [1500, 3000, 6000, 12000]:
        r = run_mission(MissionConfig(target_label="Large UXO / naval mine",
                                      target_depth_m=12.0, n_samples=n))
        peaks.append(np.abs(r.gradient_true).max())
        times.append(r.detection.peak_time)
    peaks = np.array(peaks); times = np.array(times)
    rel = (peaks.max() - peaks.min()) / peaks.mean()
    tspread = times.max() - times.min()
    ok = rel < 0.02 and tspread < 0.5
    return StressResult(
        "S-3", "Time-grid independence", "PASS" if ok else "FAIL",
        f"peak gradient varies {rel*100:.3f}%; detection time spread {tspread*1000:.0f} ms",
        "< 2% amplitude, < 500 ms timing",
        f"N = 1500..12000 samples")


def s4_parameter_sensitivity() -> StressResult:
    """
    One-at-a-time sensitivity: which inputs actually drive detection range?
    Reported as elasticity dlnR50/dlnX so parameters are directly comparable.
    """
    base = dict(n_trials=6000, target_label="Large UXO / naval mine",
                range_min_m=2.0, range_max_m=300.0, seed=1)
    r0 = monte_carlo_detection(**base)["r50_m"]
    elast = {}
    perturb = {
        "sensor_noise_T": 2.0,
        "clutter_residual_T": 2.0,
        "cmrr_db": None,          # handled additively below
        "baseline_m": 2.0,
    }
    for k, factor in perturb.items():
        kw = dict(base)
        if k == "cmrr_db":
            kw[k] = 66.0          # +6 dB = 2x rejection
            ratio_x = 2.0
        else:
            default = {"sensor_noise_T": 5e-15 * np.sqrt(10.0),
                       "clutter_residual_T": 2.23e-9,
                       "baseline_m": 0.5}[k]
            kw[k] = default * factor
            ratio_x = factor
        r1 = monte_carlo_detection(**kw)["r50_m"]
        elast[k] = np.log(r1 / r0) / np.log(ratio_x)
    dominant = max(elast, key=lambda k: abs(elast[k]))
    # sensor noise should have near-zero influence (clutter-limited claim)
    ok = abs(elast["sensor_noise_T"]) < 0.05 and abs(elast["clutter_residual_T"]) > 0.1
    return StressResult(
        "S-4", "Parameter sensitivity (elasticity)", "PASS" if ok else "FAIL",
        "; ".join(f"{k}: {v:+.3f}" for k, v in elast.items()),
        "|elasticity| of sensor noise < 0.05 (confirms clutter-limited)",
        f"dominant driver: {dominant}")


def s5_extreme_regimes() -> StressResult:
    """Behaviour at the edges: huge fields, zero field, extreme sea state."""
    issues = []
    cfg = BlochConfig()

    # zero field -> pure pumped polarisation, no transverse component
    ss0 = steady_state_analytic(0.0, cfg)
    if abs(float(ss0["Py"])) > 1e-15:
        issues.append("nonzero Py at B=0")
    if abs(float(ss0["Pz"]) - cfg.P_equilibrium) > 1e-12:
        issues.append("Pz != P_eq at B=0")

    # very large field -> response must saturate and decay, never diverge
    ss_big = steady_state_analytic(1e-3, cfg)
    if not np.isfinite(float(ss_big["Py"])) or abs(float(ss_big["Py"])) > 1.0:
        issues.append("unphysical Py at B=1 mT")

    # polarisation magnitude bound |P| <= 1 across a wide sweep
    B = np.logspace(-13, -3, 400)
    ss = steady_state_analytic(B, cfg)
    mag = np.sqrt(ss["Px"]**2 + ss["Py"]**2 + ss["Pz"]**2)
    if mag.max() > 1.0 + 1e-9:
        issues.append(f"|P| exceeded 1 (max {mag.max():.6f})")

    # extreme sea state must not silently pass the SERF gate
    t = np.linspace(0, 60, 2000)
    p = AUVPlatform(sea_state=SEA_STATES["rough (SS5-6)"],
                    nulling=NullingSystem(passive_factor=1.0, active_factor=1.0))
    fb = field_budget(p, t)
    if fb["serf_viable"]:
        issues.append("SERF reported viable with NO shielding -- gate broken")

    return StressResult(
        "S-5", "Extreme regimes", "PASS" if not issues else "FAIL",
        f"|P|max = {mag.max():.6f}; unshielded rough-sea gate correctly rejects",
        "no divergence, |P| <= 1, gates behave",
        "; ".join(issues) if issues else "no anomalies")


def s6_dynamic_bounds() -> StressResult:
    """Time-domain integration must respect |P| <= 1 under harsh driving."""
    cfg = BlochConfig()
    t = np.linspace(0, 2.0, 8000)

    def B_fn(tt):
        # aggressive swept drive spanning far beyond the linear regime
        # Host array: integrate() picks the array module by ensemble size and
        # a single trajectory stays on the CPU, so pushing this to the device
        # every step only forced an immediate copy back (32k round trips).
        return np.array([[5e-8 * np.sin(2 * np.pi * 30.0 * tt),
                          2e-8 * np.cos(2 * np.pi * 17.0 * tt),
                          1e-8]])

    r = integrate(t, B_fn, cfg, n_ensemble=1, record_every=10)
    P = r["P"][:, 0, :]
    mag = np.linalg.norm(P, axis=1)
    ok = np.all(np.isfinite(P)) and mag.max() <= 1.0 + 1e-6
    return StressResult(
        "S-6", "Dynamic physical bounds", "PASS" if ok else "FAIL",
        f"max |P| = {mag.max():.6f} under 50 nT swept drive",
        "|P| <= 1, all finite",
        "harsh multi-frequency drive well outside linear regime")


def s7_false_alarm_calibration() -> StressResult:
    """
    The realised false-alarm rate must match the design Pfa.
    A detector whose measured Pfa differs by orders of magnitude from its
    design value gives meaningless Pd numbers.
    """
    design = 1e-3
    m = monte_carlo_detection(n_trials=20000, target_label="Large UXO / naval mine",
                              range_min_m=2.0, range_max_m=300.0,
                              pfa=design, seed=3)
    realised = float((m["null_statistic"] > m["threshold"]).mean())
    ratio = realised / design if design > 0 else np.inf
    ok = 0.3 <= ratio <= 3.0
    return StressResult(
        "S-7", "False-alarm calibration", "PASS" if ok else "FAIL",
        f"design Pfa {design:.0e}, realised {realised:.2e} (ratio {ratio:.2f})",
        "realised within 3x of design",
        "threshold set from empirical null quantile")


def s8_literature_closure() -> StressResult:
    """
    Independent numerical closure against published values not already used
    to build the model.
    """
    checks, fails = [], []

    # (a) Allred 2002: 1.1 Hz linewidth -> implied relaxation rate
    #     Lorentzian FWHM (Hz) relates to rate via Gamma = pi * FWHM
    gamma_from_linewidth = np.pi * 1.1
    if not (1.0 < gamma_from_linewidth < 10.0):
        fails.append("Allred linewidth -> rate out of range")
    checks.append(f"Allred 1.1 Hz -> Gamma ~ {gamma_from_linewidth:.2f} s^-1 "
                  f"(model uses R_rel = 5 s^-1)")

    # (b) Field-units linewidth of that device must sit in the SERF window
    hw = gamma_from_linewidth / GAMMA_E
    if not (0.0 < hw < SERF_RESIDUAL_FIELD_TOLERANCE_T):
        fails.append("Allred implied field width outside SERF window")
    checks.append(f"implied HWHM {hw*1e12:.1f} pT < {SERF_RESIDUAL_FIELD_TOLERANCE_T*1e9:.0f} nT window")

    # (c) Oehler 2024 compensation ratio
    ratio = OEHLER_RAW_INTERFERENCE_T / OEHLER_COMPENSATED_RESIDUAL_T
    if not (50 < ratio < 200):
        fails.append("Oehler ratio unexpected")
    checks.append(f"Oehler compensation {ratio:.0f}x (40.9 dB)")

    # (d) Oehler sub-10 nT class vs our hardest target at its standoff
    hard = TARGETS["Adversary micro-UUV, low-signature hull"]
    b = hard.peak_field_at(hard.typical_standoff_m)
    if not (0.1e-9 <= b <= 10e-9):
        fails.append("hardest target outside demonstrated detection class")
    checks.append(f"hardest target {b*1e9:.2f} nT at {hard.typical_standoff_m:.0f} m")

    # (e) required nulling consistency
    req = EARTH_FIELD_NOMINAL_T / SERF_RESIDUAL_FIELD_TOLERANCE_T
    if abs(req - 1e4) / 1e4 > 0.01:
        fails.append("nulling requirement inconsistent")
    checks.append(f"nominal nulling {req:,.0f}x")

    return StressResult(
        "S-8", "Literature closure", "PASS" if not fails else "FAIL",
        "; ".join(checks), "all independent checks consistent",
        "; ".join(fails) if fails else "no inconsistencies")


def s9_cape_monotonicity() -> StressResult:
    """CAPE gain must decrease monotonically with residual field, no artefacts."""
    R_tot = BlochConfig().R_tot
    B = np.logspace(-12, -7, 500)
    g = np.array([cape_error(b, 1e-9, R_tot, GAMMA_E)["gain_factor"] for b in B])
    monotone = np.all(np.diff(g) <= 1e-12)
    bounded = (g.min() >= 0.0) and (g.max() <= 1.0 + 1e-12)
    ok = monotone and bounded
    return StressResult(
        "S-9", "CAPE monotonicity and bounds", "PASS" if ok else "FAIL",
        f"gain spans {g.min():.4f} to {g.max():.4f}, monotone decreasing = {monotone}",
        "monotone, bounded in [0,1]",
        "no numerical artefacts over 5 decades of residual field")


def s10_range_law_scaling() -> StressResult:
    """
    Detection range must follow the physical scaling law for a gradiometer:
    signal ~ 1/r^4  =>  R50 ~ m^(1/4).
    This is an emergent property, not imposed, so agreement is meaningful.
    """
    labels = ["Adversary micro-UUV, low-signature hull",
              "Small UXO / ordnance cluster",
              "Large UXO / naval mine"]
    rmax = {labels[0]: 80.0, labels[1]: 150.0, labels[2]: 300.0}
    moments, r50s = [], []
    for l in labels:
        m = monte_carlo_detection(n_trials=8000, target_label=l,
                                  range_min_m=2.0, range_max_m=rmax[l], seed=2)
        moments.append(TARGETS[l].moment_A_m2)
        r50s.append(m["r50_m"])
    moments = np.array(moments); r50s = np.array(r50s)
    slope = np.polyfit(np.log(moments), np.log(r50s), 1)[0]
    ok = 0.20 <= slope <= 0.32
    return StressResult(
        "S-10", "Range-law scaling exponent", "PASS" if ok else "FAIL",
        f"fitted d(lnR50)/d(ln m) = {slope:.3f}",
        "0.25 predicted for 1/r^4 gradiometer signal (accept 0.20-0.32)",
        f"moments {moments.tolist()}, R50 {[f'{r:.1f}' for r in r50s]} m")


ALL_STRESS = [s1_monte_carlo_convergence, s2_seed_stability,
              s3_time_grid_independence, s4_parameter_sensitivity,
              s5_extreme_regimes, s6_dynamic_bounds,
              s7_false_alarm_calibration, s8_literature_closure,
              s9_cape_monotonicity, s10_range_law_scaling]


def run_all(verbose: bool = True) -> List[StressResult]:
    out = []
    for fn in ALL_STRESS:
        try:
            r = fn()
        except Exception as exc:
            r = StressResult(fn.__name__, fn.__name__, "FAIL",
                             f"exception: {type(exc).__name__}: {exc}", "-")
        out.append(r)
        if verbose:
            mark = {"PASS": "[ OK ]", "FAIL": "[FAIL]", "INFO": "[INFO]"}[r.status]
            print(f"{mark} {r.test_id:6s} {r.name}")
            print(f"         {r.metric}")
            print(f"         criterion: {r.criterion}")
            if r.detail:
                print(f"         {r.detail}")
    return out


def summary(res: List[StressResult]) -> dict:
    return {"total": len(res),
            "passed": sum(r.status == "PASS" for r in res),
            "failed": sum(r.status == "FAIL" for r in res)}


if __name__ == "__main__":
    print("=" * 78)
    print("SERF-MAD STRESS TEST SUITE")
    print("=" * 78)
    r = run_all()
    s = summary(r)
    print("-" * 78)
    print(f"TOTAL {s['total']}  PASS {s['passed']}  FAIL {s['failed']}")
