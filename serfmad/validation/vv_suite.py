"""
serfmad.validation.vv_suite
===========================
Verification & Validation suite.

TERMINOLOGY (used precisely throughout)
---------------------------------------
  VERIFICATION : "are we solving the equations right?"  -- internal
      consistency: solver vs analytic solution, analytic vs finite
      difference, conservation laws, convergence order.
  VALIDATION   : "are we solving the right equations?"  -- agreement with
      EXTERNAL measured reality: published device performance, published
      compensation benchmarks, published physical constants.

Each test returns a structured record with PASS/FAIL, the numbers, and the
tolerance, so results can be tabulated directly into the report. Tests that
compare against literature carry the citation inline.

A test that cannot be run (e.g. QuTiP missing) reports SKIP -- never PASS.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from typing import List

from ..constants import (
    GAMMA_E, MU0_OVER_4PI, SERF_RESIDUAL_FIELD_TOLERANCE_T,
    EARTH_FIELD_NOMINAL_T, OEHLER_RAW_INTERFERENCE_T,
    OEHLER_COMPENSATED_RESIDUAL_T, SENSOR_ENVELOPE, RB87,
)
from ..quantum.bloch import BlochConfig, integrate, steady_state_analytic, check_stability
from ..quantum import lindblad as lb
from ..physics.dipole import (dipole_field, gradient_tensor, gradient_tensor_fd,
                              on_axis_peak_field, detection_range)
from ..physics.platform import (AUVPlatform, SeaState, SEA_STATES, NullingSystem,
                                field_budget, cape_error, rotation_matrix)
from ..processing.compensation import reference_cancel, tolles_lawson_compensate
from ..processing.detection import dipole_transit_template, matched_filter, detect
from ..backend import xp, to_device, to_host, describe as backend_describe


@dataclass
class VVResult:
    test_id: str
    category: str            # 'verification' | 'validation'
    name: str
    status: str              # 'PASS' | 'FAIL' | 'SKIP'
    measured: str
    expected: str
    tolerance: str
    citation: str = ""
    detail: str = ""

    def row(self):
        return (self.test_id, self.category, self.name, self.status,
                self.measured, self.expected, self.tolerance, self.citation)


# ===========================================================================
# VERIFICATION TESTS
# ===========================================================================

def v01_dipole_vs_closed_form() -> VVResult:
    """Full vector dipole field must reproduce the textbook on-axis formula."""
    m, r = 1e5, 50.0
    B_vec = dipole_field(np.array([0., 0., r]), np.array([0., 0., m]))
    meas = float(np.linalg.norm(B_vec))
    exp = on_axis_peak_field(m, r)
    rel = abs(meas - exp) / exp
    return VVResult("V-01", "verification", "Dipole field vs closed form",
                    "PASS" if rel < 1e-12 else "FAIL",
                    f"{meas:.6e} T", f"{exp:.6e} T", "rel < 1e-12",
                    detail=f"rel err = {rel:.2e}")


def v02_gradient_vs_finite_difference() -> VVResult:
    """Analytic gradient tensor vs central finite differences, off-axis."""
    m = np.array([0.3, -0.5, 1.0]) * 1e5
    r = np.array([30., 12., 45.])
    Ga = gradient_tensor(r, m)
    Gf = gradient_tensor_fd(r, m, h=1e-3)
    rel = np.abs(Ga - Gf).max() / np.abs(Gf).max()
    return VVResult("V-02", "verification", "Gradient tensor vs finite difference",
                    "PASS" if rel < 1e-6 else "FAIL",
                    f"rel err {rel:.2e}", "0", "rel < 1e-6")


def v03_gradient_maxwell_constraints() -> VVResult:
    """
    In source-free space div B = 0 (trace = 0) and curl B = 0 (symmetric).
    An incorrect gradient expression essentially never satisfies both.
    """
    m = np.array([0.3, -0.5, 1.0]) * 1e5
    r = np.array([30., 12., 45.])
    G = gradient_tensor(r, m)
    scale = np.abs(G).max()
    tr = abs(np.trace(G)) / scale
    sym = np.abs(G - G.T).max() / scale
    ok = tr < 1e-10 and sym < 1e-10
    return VVResult("V-03", "verification", "Gradient tensor Maxwell constraints",
                    "PASS" if ok else "FAIL",
                    f"|trace|/scale={tr:.2e}, asym={sym:.2e}",
                    "both 0", "< 1e-10",
                    detail="div B = 0 and curl B = 0 in free space")


def v04_bloch_analytic_is_fixed_point() -> VVResult:
    """The closed-form Bloch steady state must null the RHS exactly."""
    from ..quantum.bloch import bloch_rhs
    cfg = BlochConfig()
    worst = 0.0
    for Bx in [1e-12, 1e-10, 1e-9, 5e-9]:
        ss = steady_state_analytic(Bx, cfg)
        P = to_device(np.array([[float(ss["Px"]), float(ss["Py"]), float(ss["Pz"])]]))
        B = to_device(np.array([[Bx, 0.0, 0.0]]))
        # bloch_rhs returns a DEVICE array on the CuPy backend, and CuPy
        # refuses implicit conversion to NumPy. np.asarray() therefore raised
        # TypeError the moment a real GPU was present -- invisible for as long
        # as the CUDA path went unexercised. to_host() is the sanctioned way
        # back (see backend.to_host) and is a no-op on the NumPy backend.
        worst = max(worst, float(np.abs(to_host(bloch_rhs(P, B, cfg))).max()))
    return VVResult("V-04", "verification", "Bloch analytic steady state is a fixed point",
                    "PASS" if worst < 1e-10 else "FAIL",
                    f"max|dP/dt| = {worst:.2e}", "0", "< 1e-10")


def v05_bloch_integrator_converges_to_analytic() -> VVResult:
    """Time integration from an arbitrary start must relax to the analytic steady state."""
    cfg = BlochConfig()
    Bx = 1e-9
    # Integrate 30 relaxation time constants, not 505. The transient decays
    # as exp(-t*R_1/q), so e^-30 ~ 1e-13 -- already far below the 1e-6
    # tolerance. The previous 5.0 s / 20,001-step grid spent 94% of its
    # runtime re-confirming a fixed point it had already reached, in a
    # pure-Python loop over a (1,3) array.
    tau = cfg.q / cfg.R_1
    t = np.linspace(0, 30.0 * tau, 1201)
    # Host array -- integrate() keeps a single trajectory on the CPU.
    B0 = np.array([[Bx, 0.0, 0.0]])
    res = integrate(t, lambda tt: B0, cfg, n_ensemble=1, record_every=1200)
    P_end = res["P"][-1, 0, :]
    ss = steady_state_analytic(Bx, cfg)
    exp = np.array([float(ss["Px"]), float(ss["Py"]), float(ss["Pz"])])
    rel = np.abs(P_end - exp).max() / np.abs(exp).max()
    return VVResult("V-05", "verification", "Bloch RK4 converges to analytic steady state",
                    "PASS" if rel < 1e-6 else "FAIL",
                    f"rel err {rel:.2e}", "0", "rel < 1e-6",
                    detail=f"P_end={P_end}, P_analytic={exp}")


def v06_rk4_convergence_order() -> VVResult:
    """
    RK4 must exhibit 4th-order convergence: halving dt reduces error ~16x.
    This verifies the integrator implementation itself, not just its answer.
    """
    cfg = BlochConfig()
    Bx = 2e-9
    # Host array -- integrate() keeps a single trajectory on the CPU.
    B0 = np.array([[Bx, 0.0, 0.0]])

    # Measure DURING the transient, not after it. Once the system has relaxed
    # to steady state the error is set by the fixed point itself (which every
    # step size reproduces exactly), so the convergence order becomes
    # unmeasurable -- an earlier version of this test made exactly that
    # mistake and reported ratios of 1.0.
    T = 3.0 / (cfg.R_1 / cfg.q)      # ~3 time constants: transient still live
    n_ref = 6401     # >=8x finer than the finest test grid (801): ample
    t_ref = np.linspace(0, T, n_ref)
    r_ref = integrate(t_ref, lambda tt: B0, cfg, n_ensemble=1,
                      record_every=n_ref - 1)
    ref = r_ref["P"][-1, 0, :]

    errs, dts = [], []
    for n in [201, 401, 801]:
        t = np.linspace(0, T, n)
        r = integrate(t, lambda tt: B0, cfg, n_ensemble=1, record_every=n - 1)
        errs.append(float(np.abs(r["P"][-1, 0, :] - ref).max()))
        dts.append(T / (n - 1))

    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1) if errs[i + 1] > 0]
    orders = [np.log2(rr) for rr in ratios]
    ok = len(orders) > 0 and all(o > 3.0 for o in orders)
    return VVResult("V-06", "verification", "RK4 convergence order",
                    "PASS" if ok else "FAIL",
                    f"observed order {[f'{o:.2f}' for o in orders]}",
                    "4.0 (RK4)", "order > 3.0",
                    detail=f"errors {[f'{e:.2e}' for e in errs]} at dt "
                           f"{[f'{d:.1e}' for d in dts]}")


def v07_rotation_matrix_orthonormal() -> VVResult:
    """Attitude rotation matrices must be orthonormal with det = +1."""
    worst_o, worst_d = 0.0, 0.0
    for r_, p_, y_ in [(0.1, 0.2, 0.3), (1.0, -0.5, 2.0), (-0.7, 0.9, -1.2)]:
        R = rotation_matrix(r_, p_, y_)
        worst_o = max(worst_o, float(np.abs(R @ R.T - np.eye(3)).max()))
        worst_d = max(worst_d, abs(float(np.linalg.det(R)) - 1.0))
    ok = worst_o < 1e-12 and worst_d < 1e-12
    return VVResult("V-07", "verification", "Attitude rotation orthonormality",
                    "PASS" if ok else "FAIL",
                    f"|RR^T-I|={worst_o:.2e}, |det-1|={worst_d:.2e}",
                    "0", "< 1e-12")


def v08_bloch_vs_lindblad() -> VVResult:
    """
    CROSS-MODEL VERIFICATION -- the most important test in the suite.

    The fast semiclassical Bloch solver (production) must agree with the
    QuTiP Lindblad master equation (reference truth). Note P = 2<S>.
    """
    if not lb._ensure_qutip():
        return VVResult("V-08", "verification", "Bloch vs QuTiP Lindblad", "SKIP",
                        "QuTiP unavailable", "agreement", "rel < 1e-6",
                        detail=lb.availability()["error"])
    # q=1 so both models describe the same bare-electron spin.
    # isotropic=False selects the exact Lindblad-derived R_1/R_2, which is
    # what the QuTiP collapse-operator set actually produces (verified by
    # diagonalising the Liouvillian: eigenvalues -505 and -257.5 s^-1).
    # Using isotropic=True here would compare two DIFFERENT physical models
    # and the ~30% disagreement would be a property of the comparison, not
    # a solver bug.
    cfg_b = BlochConfig(R_op=500.0, R_rel=5.0, q=1.0, isotropic=False)
    cfg_l = lb.LindbladConfig(R_op=500.0, Gamma_sd=5.0, q=1.0)

    worst = 0.0
    rows = []
    for Bx in [1e-11, 1e-10, 5e-10, 1e-9]:
        q_ss = lb.steady_state_response(Bx, cfg_l)
        P_q = np.array([2 * q_ss["Sx"], 2 * q_ss["Sy"], 2 * q_ss["Sz"]])
        b_ss = steady_state_analytic(Bx, cfg_b)
        P_b = np.array([float(b_ss["Px"]), float(b_ss["Py"]), float(b_ss["Pz"])])
        rel = np.abs(P_q - P_b).max() / max(np.abs(P_q).max(), 1e-30)
        worst = max(worst, rel)
        rows.append(f"Bx={Bx:.0e}: Py_L={P_q[1]:.5e} Py_B={P_b[1]:.5e}")
    return VVResult("V-08", "verification", "Bloch vs QuTiP Lindblad (cross-model)",
                    "PASS" if worst < 1e-6 else "FAIL",
                    f"max rel diff {worst:.2e}", "0", "rel < 1e-6",
                    detail="; ".join(rows))


def v09_matched_filter_localisation() -> VVResult:
    """Matched filter must localise a known transient to within one sample."""
    t = np.linspace(0, 20, 2000)
    true_cpa = 12.34
    s = dipole_transit_template(t, true_cpa, 1.0, "bipolar")
    rng = np.random.default_rng(1)
    x = s + rng.normal(0, 0.05 * np.abs(s).max(), len(t))
    res = detect(x, t, template=dipole_transit_template(t, t.mean(), 1.0, "bipolar"))
    err = abs(res.peak_time - true_cpa)
    dt = t[1] - t[0]
    return VVResult("V-09", "verification", "Matched-filter time localisation",
                    "PASS" if err < 5 * dt else "FAIL",
                    f"{err*1000:.1f} ms error", "0", f"< 5 samples ({5*dt*1000:.1f} ms)")


# ===========================================================================
# VALIDATION TESTS (against external measured reality)
# ===========================================================================

def x01_serf_linewidth_vs_tolerance() -> VVResult:
    """
    The modelled SERF resonance half-width must be consistent with the
    published operating window of +/-5 nT. If our model said the linewidth
    were 1000 nT or 0.001 nT, the model would be describing a different device.
    """
    cfg = BlochConfig(R_op=500.0, R_rel=5.0, q=5.0)
    hw = cfg.B_hwhm
    ok = 0.1e-9 <= hw <= SERF_RESIDUAL_FIELD_TOLERANCE_T
    return VVResult("X-01", "validation", "SERF resonance width vs published window",
                    "PASS" if ok else "FAIL",
                    f"B_HWHM = {hw*1e9:.2f} nT",
                    f"within (0.1, {SERF_RESIDUAL_FIELD_TOLERANCE_T*1e9:.0f}] nT",
                    "order-of-magnitude", "[Borna22]",
                    detail="published SERF window is +/-5 nT residual static field")


def x02_sensor_noise_envelope() -> VVResult:
    """
    The MEMS design point must sit inside the envelope of MEASURED atomic
    magnetometer performance -- not better than the best ever published.
    """
    best = min(s.noise_floor_T_rtHz for s in SENSOR_ENVELOPE)
    worst = max(s.noise_floor_T_rtHz for s in SENSOR_ENVELOPE)
    design = 5e-15   # Griffith 2010 MEMS
    ok = best <= design <= worst
    return VVResult("X-02", "validation", "MEMS noise floor within measured envelope",
                    "PASS" if ok else "FAIL",
                    f"{design*1e15:.1f} fT/rtHz",
                    f"[{best*1e15:.2f}, {worst*1e15:.1f}] fT/rtHz",
                    "inside envelope", "[Griffith10],[Dang10],[Allred02]")


def x03_earth_field_nulling_requirement() -> VVResult:
    """
    Validate the headline feasibility number: the required ambient-field
    suppression to reach SERF operation.
    """
    req = EARTH_FIELD_NOMINAL_T / SERF_RESIDUAL_FIELD_TOLERANCE_T
    ok = 5e3 <= req <= 2e4
    return VVResult("X-03", "validation", "Required ambient-field suppression",
                    "PASS" if ok else "FAIL",
                    f"{req:,.0f}x ({20*np.log10(req):.0f} dB)",
                    "~10,000x (80 dB)", "within 2x", "[NCEI],[Borna22]")


def x04_compensation_vs_oehler_benchmark() -> VVResult:
    """
    VALIDATION AGAINST A PUBLISHED FIELD RESULT.

    [Oehler24] reduced real UUV magnetic interference from 246.90 nT RMS to
    2.23 nT RMS (a 110x improvement) using attitude-based compensation. Our
    compensation chain, run on synthetic data with the same raw interference
    level, must achieve a comparable improvement ratio. If our simulated
    compensation vastly outperforms the published field result, our noise
    model is unrealistically benign.
    """
    rng = np.random.default_rng(7)
    t = np.linspace(0, 120, 6000)
    plat = AUVPlatform(sea_state=SEA_STATES["moderate (SS3-4)"])
    roll, pitch, yaw = plat.sea_state.attitude(t)

    # Synthesise interference with the published RAW magnitude
    B_body = plat.ambient_field_body(t)
    interference = B_body[:, 0] - B_body[:, 0].mean()
    interference *= OEHLER_RAW_INTERFERENCE_T / np.std(interference)
    # add non-modelled residual + white noise so perfect cancellation is impossible
    interference += rng.normal(0, 0.02 * OEHLER_RAW_INTERFERENCE_T, len(t))

    out = tolles_lawson_compensate(interference, roll, pitch, yaw)
    ratio = out["improvement_ratio"]
    published_ratio = OEHLER_RAW_INTERFERENCE_T / OEHLER_COMPENSATED_RESIDUAL_T
    # Require same order of magnitude, and specifically NOT wildly better
    ok = 0.1 * published_ratio <= ratio <= 10 * published_ratio
    return VVResult("X-04", "validation", "Motion compensation vs published UUV benchmark",
                    "PASS" if ok else "FAIL",
                    f"{ratio:.1f}x (RMS {out['rms_before']*1e9:.1f} -> "
                    f"{out['rms_after']*1e9:.2f} nT)",
                    f"{published_ratio:.1f}x (246.90 -> 2.23 nT)",
                    "within 10x of published", "[Oehler24]")


def x05_detection_range_law() -> VVResult:
    """
    Validate the 1/r^3 range law against the PEER-REVIEWED AUV benchmark.

    SOURCE-QUALITY NOTE (recorded deliberately):
    An earlier revision of this test validated against a widely-repeated
    open-web claim that degaussed submarines present "1000-2000 nT signatures
    at 200-600 m". That claim was REJECTED after direct checking: producing
    1000 nT at 200 m requires a dipole moment of ~4e7 A*m^2, which is an
    order of magnitude ABOVE the literature value for an UNdegaussed hull
    (~1e7 A*m^2) and therefore self-contradictory for a degaussed one. The
    source (a scraped-content wiki) is not citable. Retained here as an
    explicit record that the number was tested and discarded rather than
    quietly dropped.

    We validate instead against [Oehler24] (Geophysical Prospecting, peer
    reviewed), which reports AUV magnetometry resolving anomalies <10 nT.
    Our hardest target classes must fall in that regime.
    """
    from ..constants import TARGETS
    hard = TARGETS["Adversary micro-UUV, low-signature hull"]
    b = hard.peak_field_at(hard.typical_standoff_m)
    # must be in the sub-10 nT class demonstrated by Oehler et al.
    ok = 0.1e-9 <= b <= 10e-9

    # independent check that the 1/r^3 inversion is self-consistent
    r_back = detection_range(hard.moment_A_m2, b)
    consistent = abs(r_back - hard.typical_standoff_m) / hard.typical_standoff_m < 1e-9

    return VVResult("X-05", "validation", "Range law vs peer-reviewed AUV benchmark",
                    "PASS" if (ok and consistent) else "FAIL",
                    f"micro-UUV target: {b*1e9:.2f} nT at "
                    f"{hard.typical_standoff_m:.0f} m; inverse-law self-consistent="
                    f"{consistent}",
                    "sub-10 nT detection class", "0.1-10 nT", "[Oehler24]",
                    detail="Grokipedia '1000-2000 nT' claim tested and REJECTED "
                           "as physically inconsistent (would need ~4e7 A m^2)")


def x06_cape_requirement_derivation() -> VVResult:
    """
    Derive the residual-field requirement from the CAPE gain penalty and
    check it is TIGHTER than the nominal SERF window -- a non-obvious design
    consequence that the naive +/-5 nT spec hides.
    """
    R_tot = 505.0
    alpha_5pct = np.arccos(0.95)
    B_req = np.tan(alpha_5pct) * R_tot / GAMMA_E
    at_tol = cape_error(SERF_RESIDUAL_FIELD_TOLERANCE_T, 1e-9, R_tot, GAMMA_E)
    ok = B_req < SERF_RESIDUAL_FIELD_TOLERANCE_T
    return VVResult("X-06", "validation", "CAPE-derived residual field requirement",
                    "PASS" if ok else "FAIL",
                    f"{B_req*1e9:.2f} nT for <=5% gain loss; "
                    f"at 5 nT gain = {at_tol['gain_factor']:.2f}",
                    f"tighter than nominal {SERF_RESIDUAL_FIELD_TOLERANCE_T*1e9:.0f} nT",
                    "must be stricter", "[Borna22]",
                    detail="CAPE tilts the sensitive axis; meeting the nominal "
                           "SERF window is necessary but NOT sufficient")


def x07_gradiometer_cmrr_realism() -> VVResult:
    """
    Assert that the model does NOT assume perfect common-mode rejection.
    A model with infinite CMRR trivially 'solves' platform noise and is the
    single most common way MAD simulations mislead.
    """
    from ..processing.compensation import GradiometerModel
    g = GradiometerModel()
    finite = np.isfinite(g.cmrr_linear) and g.cmrr_linear < 1e6
    return VVResult("X-07", "validation", "Gradiometer CMRR is finite (anti-idealisation)",
                    "PASS" if finite else "FAIL",
                    f"CMRR = {g.cmrr_db:.0f} dB ({g.cmrr_linear:.0f}x)",
                    "finite, <= 120 dB", "must not be infinite",
                    detail="platform self-field has a real gradient across "
                           "the baseline; perfect cancellation is unphysical")


def x08_stability_guard() -> VVResult:
    """The integrator must reject a step size that violates RK4 stability."""
    cfg = BlochConfig()
    bad = check_stability(1.0, cfg, B_max=1e-9)
    good = check_stability(1e-4, cfg, B_max=1e-9)
    ok = (not bad["stable"]) and good["stable"]
    return VVResult("V-10", "verification", "RK4 stability guard",
                    "PASS" if ok else "FAIL",
                    f"dt=1.0 -> stable={bad['stable']}; dt=1e-4 -> stable={good['stable']}",
                    "False / True", "guard must trip",
                    detail=f"recommended max dt = {good['recommended_max_dt']:.2e} s")


ALL_TESTS = [
    v01_dipole_vs_closed_form,
    v02_gradient_vs_finite_difference,
    v03_gradient_maxwell_constraints,
    v04_bloch_analytic_is_fixed_point,
    v05_bloch_integrator_converges_to_analytic,
    v06_rk4_convergence_order,
    v07_rotation_matrix_orthonormal,
    v08_bloch_vs_lindblad,
    v09_matched_filter_localisation,
    x08_stability_guard,
    x01_serf_linewidth_vs_tolerance,
    x02_sensor_noise_envelope,
    x03_earth_field_nulling_requirement,
    x04_compensation_vs_oehler_benchmark,
    x05_detection_range_law,
    x06_cape_requirement_derivation,
    x07_gradiometer_cmrr_realism,
]


def run_all(verbose: bool = True) -> List[VVResult]:
    results = []
    for fn in ALL_TESTS:
        try:
            r = fn()
        except Exception as exc:
            r = VVResult(getattr(fn, "__name__", "?"), "verification",
                         fn.__doc__.strip().splitlines()[0] if fn.__doc__ else fn.__name__,
                         "FAIL", f"exception: {type(exc).__name__}: {exc}",
                         "-", "-")
        results.append(r)
        if verbose:
            mark = {"PASS": "[ OK ]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[r.status]
            print(f"{mark} {r.test_id:6s} {r.name}")
            print(f"         measured: {r.measured}")
            print(f"         expected: {r.expected}  (tol: {r.tolerance})")
            if r.citation:
                print(f"         source  : {r.citation}")
    return results


def summary(results: List[VVResult]) -> dict:
    return {
        "total": len(results),
        "passed": sum(r.status == "PASS" for r in results),
        "failed": sum(r.status == "FAIL" for r in results),
        "skipped": sum(r.status == "SKIP" for r in results),
        "verification": sum(r.category == "verification" for r in results),
        "validation": sum(r.category == "validation" for r in results),
        "backend": backend_describe(),
    }


if __name__ == "__main__":
    print("=" * 74)
    print("SERF-MAD VERIFICATION & VALIDATION SUITE")
    print("=" * 74)
    res = run_all()
    print("-" * 74)
    s = summary(res)
    print(f"TOTAL {s['total']}  PASS {s['passed']}  FAIL {s['failed']}  SKIP {s['skipped']}")
    print(f"backend: {s['backend']['detail']}")
