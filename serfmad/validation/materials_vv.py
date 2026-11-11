"""
serfmad.validation.materials_vv
===============================
Verification & Validation for the material layer and the geometry-derived
vehicle magnetic model.

WHY THIS IS A SEPARATE SUITE
----------------------------
The existing suite in `vv_suite.py` reports 17/17 and `stress.py` reports
10/10, and those counts appear in `docs/SR_report.tex` (Tables 3 and 4) and in
the README. Adding tests there would change published numbers. This suite is
therefore standalone and is run separately:

    python -m serfmad.validation.materials_vv

It reuses the `VVResult` record type from `vv_suite` so results tabulate
identically, and follows the same terminology [Oberkampf10]: M-V tests verify
the code solves its equations correctly, M-X tests validate against external
measurement or against the pre-existing model's behaviour.

Test M-10 is a REGRESSION GUARD: it asserts that importing and exercising the
material layer leaves the default parameters of the pre-existing physics
untouched. The material path is opt-in by construction, and that test is what
proves it stayed that way.
"""

from __future__ import annotations

import numpy as np
from typing import List

from .vv_suite import VVResult
from ..constants import (
    OEHLER_RAW_INTERFERENCE_T, EARTH_FIELD_NOMINAL_T,
    SERF_RESIDUAL_FIELD_TOLERANCE_T,
)
from ..materials import (
    MATERIALS, MU_METAL_ANNEALED, MU_METAL_FOIL_FORMED,
    SS316L_ANNEALED, SS316L_COLD_WORKED, NITRONIC_50, TI_6AL_4V,
    Measured, value_of, is_assumed, shielding_factor, assumed_fraction,
)
from ..constants import EngineeringAssumption
from ..physics.dipole import dipole_field
from ..physics import vehicle as veh


# ===========================================================================
# VERIFICATION
# ===========================================================================

def m01_superposition_reduces_to_single_dipole() -> VVResult:
    """
    A vehicle model containing exactly one component must reproduce the bare
    dipole field of `physics.dipole` -- i.e. the superposition machinery adds
    nothing when there is nothing to superpose.
    """
    m_vec = np.array([2.0, -1.0, 0.5])
    pos = (0.3, 0.0, 0.0)
    comp = veh.MagneticComponent(
        label="test", position_m=pos,
        moment_override=EngineeringAssumption(
            float(np.linalg.norm(m_vec)), "A*m^2", "unit test"),
        override_direction=tuple(m_vec / np.linalg.norm(m_vec)))
    v = veh.VehicleMagneticModel((comp,), "single")

    p = np.array([2.5, 0.4, -0.2])
    B_model = v.field_at(p, veh.default_ambient_vector())
    B_ref = dipole_field(p - np.asarray(pos), m_vec)
    rel = float(np.linalg.norm(B_model - B_ref) / np.linalg.norm(B_ref))
    return VVResult("M-01", "verification",
                    "Vehicle superposition reduces to single dipole",
                    "PASS" if rel < 1e-12 else "FAIL",
                    f"rel err {rel:.2e}", "identical to physics.dipole",
                    "rel < 1e-12", "[Jackson98]")


def m02_cmrr_converges_to_analytic() -> VVResult:
    """
    In the far field the derived CMRR must converge to the closed-form
    estimate d/(3L). This is the same style of check as V-02 (analytic
    gradient vs finite difference): an independent expression for the same
    quantity, agreeing where both are valid.

    Convergence is checked at large d/L, where the 3L/d expansion holds. At
    small standoff the expansion is expected to fail and is not asserted.
    """
    L = 0.5
    v = veh.VehicleMagneticModel(
        (veh.MagneticComponent(
            label="single source", position_m=(0.0, 0.0, 0.0),
            moment_override=EngineeringAssumption(10.0, "A*m^2", "unit test"),
            override_direction=(1.0, 0.0, 0.0)),), "single-source")

    errs = []
    for d in (20.0, 50.0, 100.0):
        geom = veh.GradiometerGeometry(standoff_m=d, baseline_m=L)
        num = veh.derived_cmrr(v, geom, channel_mismatch=0.0)["cmrr_linear"]
        ana = veh.analytic_cmrr_estimate(d, L)
        errs.append(abs(num - ana) / ana)

    worst = max(errs)
    improving = errs[-1] < errs[0]
    ok = worst < 0.10 and improving
    return VVResult("M-02", "verification",
                    "Derived CMRR converges to analytic d/(3L)",
                    "PASS" if ok else "FAIL",
                    f"rel err {errs[0]:.3f} -> {errs[-1]:.3f} at d/L = 40 -> 200",
                    "converges to closed form", "rel < 0.10 and decreasing",
                    "[Jackson98]")


def m03_cmrr_monotonic_in_standoff() -> VVResult:
    """
    Derived CMRR must increase monotonically with standoff. This is the
    quantitative content of R-13, and a model that failed it would be
    recommending the sensors be moved closer to the thrusters.
    """
    rows = veh.standoff_sweep([0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0])
    cm = [r["cmrr_db"] for r in rows]
    mono = all(b > a for a, b in zip(cm, cm[1:]))
    return VVResult("M-03", "verification",
                    "Derived CMRR is monotonic in standoff",
                    "PASS" if mono else "FAIL",
                    f"{cm[0]:.1f} dB at 0.8 m -> {cm[-1]:.1f} dB at 12 m",
                    "monotone increasing", "strict monotonicity",
                    "R-13")


def m04_remanence_independent_of_ambient() -> VVResult:
    """
    Induced moment must scale linearly with the ambient field while remanent
    moment must not change at all. This is the physical distinction that makes
    magnetic hygiene a BUILD-time constraint: nulling the ambient field
    removes the induced term and leaves the remanent one.
    """
    B0 = veh.default_ambient_vector()
    V = 1e-3

    comp = veh.MagneticComponent(
        label="induced+remanent", position_m=(0., 0., 0.),
        material=SS316L_ANNEALED, volume_m3=V,
        remanent_direction=(1., 0., 0.))

    # The two terms are NOT parallel -- induced follows the ambient field
    # (inclination 70 deg, mostly +z) while remanence is fixed along +x -- so
    # this must be done in vectors. Subtracting magnitudes would be wrong.
    m_rem = comp.moment(np.zeros(3))
    m_1 = comp.moment(B0)
    m_2 = comp.moment(2.0 * B0)

    # Remanent term: exactly M_r * V along +x, independent of ambient field.
    expect_rem = np.array([SS316L_ANNEALED.M_r * V, 0.0, 0.0])
    rem_err = float(np.linalg.norm(m_rem - expect_rem))

    # Induced term recovered by vector subtraction; must double when the
    # ambient field doubles.
    ind_1 = m_1 - m_rem
    ind_2 = m_2 - m_rem
    ratio = float(np.linalg.norm(ind_2) / np.linalg.norm(ind_1))

    # And it must point along the ambient field, not along the remanent axis.
    align = float(np.dot(ind_1, B0) / (np.linalg.norm(ind_1) * np.linalg.norm(B0)))

    ok = rem_err < 1e-18 and abs(ratio - 2.0) < 1e-12 and abs(align - 1.0) < 1e-12
    return VVResult("M-04", "verification",
                    "Induced moment scales with ambient; remanent does not",
                    "PASS" if ok else "FAIL",
                    f"induced ratio {ratio:.12f} at 2x ambient, aligned to B "
                    f"(cos = {align:.12f}); remanent error {rem_err:.2e} A*m^2",
                    "induced linear in B and parallel to B; remanent constant",
                    "ratio == 2 to 1e-12")


def m05_shielding_factor_reproduces_measurement() -> VVResult:
    """
    The [Dubbers86] closed form S = mu_r*Delta/D, evaluated with the
    permeability [Gohil20] MEASURED for its annealed cylinder, must reproduce
    the shielding factor that paper observed for the same geometry
    (ID 5 cm, wall 1 mm), i.e. of order 1e3.

    This validates the schema against an external measurement rather than
    against itself.
    """
    S = shielding_factor(MU_METAL_ANNEALED, diameter_m=0.05,
                         wall_thickness_m=1.0e-3)
    # [Gohil20] measured |T| ~ 2e-4 to 1e-3 for this cylinder -> S ~ 1e3-5e3
    ok = 5.0e2 <= S <= 5.0e3
    return VVResult("M-05", "validation",
                    "Shielding factor vs measured mu-metal cylinder",
                    "PASS" if ok else "FAIL",
                    f"S = {S:.0f}x ({20*np.log10(S):.1f} dB)",
                    "1e3-class, consistent with measured |T|",
                    "5e2 <= S <= 5e3", "[Gohil20],[Dubbers86]")


def m06_forming_penalty_is_represented() -> VVResult:
    """
    The schema must express the measured penalty for forming mu-metal foil
    without re-annealing: [Gohil20] measured mu_r ~ 4,660 against 55,955 for
    annealed bulk, i.e. more than an order of magnitude.
    """
    ann = value_of(MU_METAL_ANNEALED.mu_r_initial)
    formed = value_of(MU_METAL_FOIL_FORMED.mu_r_initial)
    ratio = ann / formed
    ok = ratio > 10.0
    return VVResult("M-06", "validation",
                    "Forming penalty for mu-metal is represented",
                    "PASS" if ok else "FAIL",
                    f"mu_r {ann:,.0f} annealed vs {formed:,.0f} formed "
                    f"({ratio:.1f}x)",
                    ">10x degradation, per measurement", "ratio > 10",
                    "[Gohil20]")


def m07_calibration_matches_oehler() -> VVResult:
    """
    The calibrated reference vehicle must reproduce the measured raw UUV
    interference of 246.90 nT RMS [Oehler24] at the baseline centre. This is
    the model's single anchor to measured reality.
    """
    cal = veh.calibrate_to_oehler()
    geom = veh.GradiometerGeometry()
    centre = geom.standoff_m * np.array([-1.0, 0.0, 0.0])
    B = float(np.linalg.norm(cal["model"].field_at(centre)))
    rel = abs(B - OEHLER_RAW_INTERFERENCE_T) / OEHLER_RAW_INTERFERENCE_T
    return VVResult("M-07", "validation",
                    "Vehicle model calibrated to measured UUV interference",
                    "PASS" if rel < 1e-9 else "FAIL",
                    f"{B*1e9:.3f} nT at 2 m", f"{OEHLER_RAW_INTERFERENCE_T*1e9:.2f} nT",
                    "rel < 1e-9", "[Oehler24]")


def m08_material_substitution_changes_result() -> VVResult:
    """
    THE POINT OF THE WHOLE EXERCISE. Substituting a material must change a
    computed field. Before this layer existed, no material property could
    influence any result; this test fails if that path is ever broken.

    Cold-worked 316L fasteners are replaced with Nitronic 50, which [BSSA] and
    [Electralloy50] identify as retaining low permeability in the cold-worked
    condition.
    """
    def frame(mat):
        return veh.VehicleMagneticModel((veh.MagneticComponent(
            label="fasteners", position_m=(0.30, 0.0, 0.0),
            material=mat, volume_m3=1.2e-4,
            remanent_direction=(1., 0., 0.)),), mat.label)

    p = np.array([-2.0, 0.0, 0.0])
    B_bad = float(np.linalg.norm(frame(SS316L_COLD_WORKED).field_at(p)))
    B_good = float(np.linalg.norm(frame(NITRONIC_50).field_at(p)))
    ratio = B_bad / B_good if B_good > 0 else np.inf
    ok = ratio > 10.0
    return VVResult("M-08", "verification",
                    "Material substitution propagates to the field",
                    "PASS" if ok else "FAIL",
                    f"cold-worked 316L {B_bad*1e12:.2f} pT vs "
                    f"Nitronic 50 {B_good*1e12:.2f} pT ({ratio:.0f}x)",
                    "material choice changes the answer", "ratio > 10",
                    "[BSSA],[Electralloy50]")


def m09_geometry_dominates_matching() -> VVResult:
    """
    At realistic standoffs the geometric leakage must dominate channel
    mismatch -- the central claim of `physics.vehicle`. If matching bound
    instead, the assumed 60 dB would be defensible and this work would be
    unnecessary.
    """
    rows = veh.standoff_sweep([1.0, 2.0, 3.0, 5.0])
    all_geo = all(r["geometric_limited"] for r in rows)
    worst = max(r["cmrr_db"] for r in rows)
    ok = all_geo and worst < 60.0
    return VVResult("M-09", "validation",
                    "Near-field CMRR is geometry-limited, not matching-limited",
                    "PASS" if ok else "FAIL",
                    f"geometry binds at all standoffs 1-5 m; best CMRR "
                    f"{worst:.1f} dB",
                    "geometry-limited and well below the assumed 60 dB",
                    "geometric > mismatch at all d", "[Oehler24], R-09")


def m10_regression_existing_defaults_untouched() -> VVResult:
    """
    REGRESSION GUARD. Importing and exercising the material layer must not
    perturb the pre-existing model's defaults, because every published number
    in SR_report.tex depends on them.
    """
    from ..physics.platform import NullingSystem
    from ..processing.compensation import GradiometerModel
    from ..mission import MissionConfig

    checks = {
        "NullingSystem.passive_factor": (NullingSystem().passive_factor, 100.0),
        "NullingSystem.active_factor": (NullingSystem().active_factor, 100.0),
        "GradiometerModel.cmrr_db": (GradiometerModel().cmrr_db, 60.0),
        "GradiometerModel.baseline_m": (GradiometerModel().baseline_m, 0.5),
        "MissionConfig.cmrr_db": (MissionConfig().cmrr_db, 60.0),
        "MissionConfig.passive_shield": (MissionConfig().passive_shield, 100.0),
        "SERF tolerance": (SERF_RESIDUAL_FIELD_TOLERANCE_T, 5e-9),
        "Earth nominal": (EARTH_FIELD_NOMINAL_T, 50e-6),
    }
    bad = [k for k, (got, want) in checks.items() if got != want]
    return VVResult("M-10", "verification",
                    "Pre-existing defaults unchanged by the material layer",
                    "PASS" if not bad else "FAIL",
                    f"{len(checks)-len(bad)}/{len(checks)} defaults intact"
                    + (f"; CHANGED: {bad}" if bad else ""),
                    "material layer is strictly opt-in", "exact equality")


def m11_provenance_is_typed() -> VVResult:
    """
    Every property in the schema must be typed either `Measured` or
    `EngineeringAssumption` -- never a bare float. This is the mechanical
    enforcement of the project rule that a modelling choice must never be
    presentable as a measurement.
    """
    bare = []
    for name, mat in MATERIALS.items():
        for f in ("chi_volume", "mu_r_initial", "M_r_A_per_m", "sigma_S_per_m",
                  "B_sat_T", "density_kg_m3", "tensile_MPa",
                  "thermal_conductivity_W_mK", "max_service_temp_C"):
            p = getattr(mat, f)
            if not isinstance(p, (Measured, EngineeringAssumption)):
                bare.append(f"{name}.{f}")
    frac = assumed_fraction()
    return VVResult("M-11", "verification",
                    "All schema values carry a provenance type",
                    "PASS" if not bare else "FAIL",
                    f"0 bare floats in {len(MATERIALS)} materials; "
                    f"{frac*100:.1f}% are assumptions",
                    "every value typed Measured or EngineeringAssumption",
                    "no bare floats")


ALL_TESTS = (
    m01_superposition_reduces_to_single_dipole,
    m02_cmrr_converges_to_analytic,
    m03_cmrr_monotonic_in_standoff,
    m04_remanence_independent_of_ambient,
    m05_shielding_factor_reproduces_measurement,
    m06_forming_penalty_is_represented,
    m07_calibration_matches_oehler,
    m08_material_substitution_changes_result,
    m09_geometry_dominates_matching,
    m10_regression_existing_defaults_untouched,
    m11_provenance_is_typed,
)


def run_all() -> List[VVResult]:
    return [t() for t in ALL_TESTS]


def summary(results: List[VVResult]) -> dict:
    return {
        "total": len(results),
        "passed": sum(r.status == "PASS" for r in results),
        "failed": sum(r.status == "FAIL" for r in results),
        "skipped": sum(r.status == "SKIP" for r in results),
    }


def main() -> int:
    print("=" * 78)
    print("SERF-MAD MATERIAL LAYER -- VERIFICATION & VALIDATION")
    print("=" * 78)
    res = run_all()
    for r in res:
        mark = "[ OK ]" if r.status == "PASS" else f"[{r.status}]"
        print(f"{mark} {r.test_id:6s} {r.name}")
        print(f"         measured: {r.measured}")
        print(f"         expected: {r.expected}  (tol: {r.tolerance})")
        if r.citation:
            print(f"         source  : {r.citation}")
    s = summary(res)
    print("-" * 78)
    print(f"TOTAL {s['total']}  PASS {s['passed']}  FAIL {s['failed']}  "
          f"SKIP {s['skipped']}")
    return 1 if s["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
