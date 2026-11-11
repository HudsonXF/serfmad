"""
serfmad.physics.vehicle
=======================
Geometry-based vehicle magnetic model: turns material properties and component
placement into a field at the gradiometer cells, and from that into a DERIVED
common-mode rejection ratio.

WHAT THIS REPLACES
------------------
`SR_report.tex` Section 9.1(2) records as a limitation that "vehicle
interference is represented by a rejection ratio and a 1/f residual rather than
derived from hull geometry and materials", and Recommendation 3 asks for
exactly this module. Previously `cmrr_db = 60.0` was a free parameter. Here it
is computed from where the sources are and what they are made of.

THE CENTRAL DISTINCTION THIS MODULE MAKES
-----------------------------------------
A two-cell gradiometer rejects common-mode field. How well it does so depends
entirely on WHERE the interfering source is, and the two regimes differ by
orders of magnitude:

  1. DISTANT / UNIFORM SOURCES (the Earth's field). Over a 0.5 m baseline the
     geomagnetic field is uniform to a very good approximation, so the
     differential signal is set purely by how well the two channels are
     matched. With the gain and axis matching of R-09 (<=0.1%) the rejection
     is ~1/1e-3 = 60 dB. The commonly quoted figure is correct HERE.

  2. NEAR-FIELD SOURCES (the vehicle's own thrusters, batteries, electronics).
     These sit metres from the baseline, so their field has a genuine gradient
     across it. That gradient is a real differential signal and NO amount of
     channel matching removes it. For a dipole at standoff d and baseline L,
     the differential-to-common ratio is approximately 3L/d, giving

         CMRR ~ d / (3L)

     At d = 2 m and L = 0.5 m this is a rejection of about 1.3x -- a few dB,
     not 60 dB. Reaching 60 dB against a near-field source by standoff alone
     would require d = 3L/1e-3 = 1500 m, which is not an option on a 1.6 m
     vehicle.

`SR_report.tex` states the mechanism qualitatively in its X-07 discussion
("their field has a genuine gradient across it and cancellation is necessarily
imperfect"). This module quantifies it. The engineering consequence is the
thesis of the accompanying materials paper: because gradiometry cannot be
relied upon to reject the platform, the platform must be made quiet in
ABSOLUTE terms, which is a materials and layout problem rather than a signal
processing one.

CALIBRATION
-----------
The reference vehicle below is calibrated so its total field at the nominal
sensor station reproduces the measured raw UUV interference of 246.90 nT RMS
[Oehler24]. That is the only measured anchor available; component-level
moment splits are engineering assumptions and are typed as such.

References (full citations in docs/REFERENCES.md):
  [Jackson98]  dipole field and gradient tensor -- via `physics.dipole`
  [Oehler24]   246.90 nT raw / 2.23 nT compensated UUV interference benchmark
  [Seidel23]   Girona 500: fluxgates at 2 m from the nose, ~3 m from thrusters
  [Seidel23sheet] rigid four-sensor array; rigid mount preferred over towed
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from ..constants import (
    EngineeringAssumption,
    EARTH_FIELD_NOMINAL_T,
    OEHLER_RAW_INTERFERENCE_T,
)
from ..materials import (
    MagneticMaterial,
    MATERIALS,
    NITRONIC_50,
    SS316L_COLD_WORKED,
    TI_6AL_4V,
    CFRP_QUASI_ISOTROPIC,
    value_of,
    is_assumed,
)
from .dipole import dipole_field, gradient_tensor

# Gain/axis matching of the two gradiometer channels, from R-09 of the
# feasibility report (<=0.1%). This sets the rejection floor for a UNIFORM
# field and is a specification, not a measurement.
CHANNEL_MISMATCH_DEFAULT = EngineeringAssumption(
    1.0e-3, "dimensionless",
    "R-09 specifies gradiometer gain and axis matching <=0.1%; 1e-3 is that "
    "specification expressed as a fractional common-mode leakage")


# ----------------------------------------------------------------------------
# SECTION 1 -- COMPONENTS
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class MagneticComponent:
    """
    One magnetic source on the vehicle.

    Its moment comes from ONE of two routes, and which one is used is recorded
    so a report table can say so:

      (a) MATERIAL-DERIVED. m_induced = chi * V * H along the ambient field,
          plus m_remanent = M_r * V along `remanent_direction`. Appropriate for
          structural items -- housings, brackets, fasteners -- where the
          material property is the whole story.

      (b) EXPLICIT MOMENT OVERRIDE. For a motor or a battery pack, m is NOT
          simply M_r * V: a brushless motor's magnetic circuit is designed to
          close its flux internally through the stator, so the NET EXTERNAL
          moment is far smaller than the sum of its magnets' moments, and a
          battery's moment is dominated by DC current loops rather than by any
          material susceptibility. Pretending otherwise would be a fabricated
          material calculation. Such components carry an explicit
          EngineeringAssumption moment instead.
    """
    label: str
    position_m: Tuple[float, float, float]     # body frame, metres
    material: Optional[MagneticMaterial] = None
    volume_m3: float = 0.0
    remanent_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    moment_override: Optional[EngineeringAssumption] = None
    override_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __post_init__(self):
        if self.moment_override is None and self.material is None:
            raise ValueError(
                f"{self.label}: needs either a material+volume or a "
                f"moment_override")

    @property
    def derivation(self) -> str:
        """'material-derived' or 'assumed-moment' -- for provenance tables."""
        return "assumed-moment" if self.moment_override is not None \
            else "material-derived"

    def moment(self, B_ambient_vec: np.ndarray) -> np.ndarray:
        """
        Total dipole moment vector (A*m^2) in the ambient field B_ambient_vec.

        For the material-derived route the induced part follows the ambient
        field direction (as induced magnetisation must) while the remanent part
        keeps its own fixed body-frame direction -- which is why remanence
        survives when the ambient field is nulled and induced magnetisation
        does not.
        """
        B_ambient_vec = np.asarray(B_ambient_vec, dtype=float)

        if self.moment_override is not None:
            d = np.asarray(self.override_direction, dtype=float)
            d = d / max(np.linalg.norm(d), 1e-30)
            return value_of(self.moment_override) * d

        mu0 = 4.0e-7 * np.pi
        H_vec = B_ambient_vec / mu0
        m_induced = self.material.chi * self.volume_m3 * H_vec

        d = np.asarray(self.remanent_direction, dtype=float)
        d = d / max(np.linalg.norm(d), 1e-30)
        m_remanent = self.material.M_r * self.volume_m3 * d

        return m_induced + m_remanent

    def field_at(self, r_field_m: np.ndarray,
                 B_ambient_vec: np.ndarray) -> np.ndarray:
        """Field this component produces at a point (T), via `physics.dipole`."""
        r_field_m = np.asarray(r_field_m, dtype=float)
        r_rel = r_field_m - np.asarray(self.position_m, dtype=float)
        return dipole_field(r_rel, self.moment(B_ambient_vec))


# ----------------------------------------------------------------------------
# SECTION 2 -- VEHICLE MODEL
# ----------------------------------------------------------------------------

@dataclass
class VehicleMagneticModel:
    """
    A vehicle as a superposition of point-dipole magnetic sources.

    The point-dipole treatment is the same approximation the report already
    uses for targets, and carries the same caveat: it is valid beyond roughly
    one body length of each source. At a 2 m standoff from a ~0.1 m motor that
    is comfortably satisfied.
    """
    components: Tuple[MagneticComponent, ...]
    label: str = "reference REMUS-class AUV"

    def field_at(self, r_field_m: np.ndarray,
                 B_ambient_vec: Optional[np.ndarray] = None) -> np.ndarray:
        """Total vehicle field at a point (T), summed over all components."""
        if B_ambient_vec is None:
            B_ambient_vec = default_ambient_vector()
        r_field_m = np.asarray(r_field_m, dtype=float)
        total = np.zeros(r_field_m.shape)
        for c in self.components:
            total = total + c.field_at(r_field_m, B_ambient_vec)
        return total

    def gradient_at(self, r_field_m: np.ndarray,
                    B_ambient_vec: Optional[np.ndarray] = None) -> np.ndarray:
        """Total vehicle gradient tensor at a point (T/m)."""
        if B_ambient_vec is None:
            B_ambient_vec = default_ambient_vector()
        r_field_m = np.asarray(r_field_m, dtype=float)
        total = np.zeros(r_field_m.shape[:-1] + (3, 3))
        for c in self.components:
            r_rel = r_field_m - np.asarray(c.position_m, dtype=float)
            total = total + gradient_tensor(r_rel, c.moment(B_ambient_vec))
        return total

    def total_moment(self, B_ambient_vec: Optional[np.ndarray] = None) -> np.ndarray:
        """Vector sum of all component moments (A*m^2)."""
        if B_ambient_vec is None:
            B_ambient_vec = default_ambient_vector()
        m = np.zeros(3)
        for c in self.components:
            m = m + c.moment(B_ambient_vec)
        return m

    def scaled(self, factor: float) -> "VehicleMagneticModel":
        """
        Return a copy with every component moment scaled.

        Used to calibrate the reference layout against the [Oehler24] raw
        interference benchmark, and to sweep whole-vehicle cleanliness.
        """
        new = []
        for c in self.components:
            if c.moment_override is not None:
                ov = c.moment_override
                new.append(MagneticComponent(
                    label=c.label, position_m=c.position_m,
                    moment_override=EngineeringAssumption(
                        value_of(ov) * factor, ov.units,
                        ov.rationale + f" [scaled x{factor:.4g}]"),
                    override_direction=c.override_direction))
            else:
                new.append(MagneticComponent(
                    label=c.label, position_m=c.position_m,
                    material=c.material, volume_m3=c.volume_m3 * factor,
                    remanent_direction=c.remanent_direction))
        return VehicleMagneticModel(tuple(new), self.label + f" (x{factor:.4g})")

    def provenance_rows(self) -> Tuple[Tuple[str, str, float], ...]:
        """(component, derivation, |moment|) for report tables."""
        B = default_ambient_vector()
        return tuple((c.label, c.derivation,
                      float(np.linalg.norm(c.moment(B))))
                     for c in self.components)


def default_ambient_vector() -> np.ndarray:
    """
    Nominal geomagnetic vector in the body frame, matching the convention in
    `physics.platform.ambient_field_body`: mid-latitude, inclination 70 deg.
    """
    inc = np.radians(70.0)
    return EARTH_FIELD_NOMINAL_T * np.array([np.cos(inc), 0.0, np.sin(inc)])


# ----------------------------------------------------------------------------
# SECTION 3 -- GRADIOMETER RESPONSE AND DERIVED CMRR
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class GradiometerGeometry:
    """
    Two cells on a rigid baseline, positioned relative to the vehicle origin.

    standoff_m : distance from the vehicle origin to the baseline CENTRE,
        along `direction`. This is the quantity R-13 asks to maximise and that
        a boom or tow buys.
    baseline_m : cell separation (R-09 requires >=0.5 m).
    direction : unit vector from vehicle origin toward the baseline centre.
    baseline_axis : unit vector along which the two cells are separated.
    """
    standoff_m: float = 2.0
    baseline_m: float = 0.5
    direction: Tuple[float, float, float] = (-1.0, 0.0, 0.0)
    baseline_axis: Tuple[float, float, float] = (-1.0, 0.0, 0.0)

    def cell_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        d = np.asarray(self.direction, dtype=float)
        d = d / max(np.linalg.norm(d), 1e-30)
        a = np.asarray(self.baseline_axis, dtype=float)
        a = a / max(np.linalg.norm(a), 1e-30)
        centre = self.standoff_m * d
        return centre + 0.5 * self.baseline_m * a, centre - 0.5 * self.baseline_m * a


def gradiometer_response(vehicle: VehicleMagneticModel,
                         geom: GradiometerGeometry,
                         B_ambient_vec: Optional[np.ndarray] = None,
                         sense_axis: Sequence[float] = (1.0, 0.0, 0.0)) -> dict:
    """
    Common-mode and differential vehicle field seen by the two cells.

    The projection onto `sense_axis` reflects that a SERF cell is a VECTOR
    magnetometer sensitive to one transverse axis (the report's Figure 1b
    convention: pumped along z, probed along x, so a transverse field is sensed
    as P_y). Using the projected component rather than |B| is what a real
    single-axis channel measures.
    """
    if B_ambient_vec is None:
        B_ambient_vec = default_ambient_vector()
    s = np.asarray(sense_axis, dtype=float)
    s = s / max(np.linalg.norm(s), 1e-30)

    p1, p2 = geom.cell_positions()
    B1 = float(np.dot(vehicle.field_at(p1, B_ambient_vec), s))
    B2 = float(np.dot(vehicle.field_at(p2, B_ambient_vec), s))

    common = 0.5 * (B1 + B2)
    differential = B1 - B2
    return {
        "B_cell1_T": B1,
        "B_cell2_T": B2,
        "common_mode_T": common,
        "differential_T": differential,
        "gradient_T_per_m": differential / geom.baseline_m,
        "geometric_leakage": abs(differential) / abs(common) if common else np.inf,
    }


def derived_cmrr(vehicle: VehicleMagneticModel,
                 geom: GradiometerGeometry,
                 B_ambient_vec: Optional[np.ndarray] = None,
                 channel_mismatch: Optional[float] = None,
                 sense_axis: Sequence[float] = (1.0, 0.0, 0.0)) -> dict:
    """
    Common-mode rejection ratio DERIVED from geometry, rather than assumed.

    Two independent leakage mechanisms are combined:

      geometric : |B1 - B2| / |(B1+B2)/2|, the genuine gradient of the
          vehicle's near field across the baseline. Irreducible by matching.
      mismatch  : fractional gain/axis mismatch between channels (R-09).

    They are added because they are independent and both leak common mode into
    the differential channel; adding is the conservative choice against
    root-sum-square, and at realistic standoffs the geometric term dominates so
    completely that the choice does not matter.

    Returns the derived CMRR in dB together with both contributions, so the
    report can show which one binds.
    """
    if channel_mismatch is None:
        channel_mismatch = value_of(CHANNEL_MISMATCH_DEFAULT)

    r = gradiometer_response(vehicle, geom, B_ambient_vec, sense_axis)
    geo = r["geometric_leakage"]
    total = geo + channel_mismatch
    cmrr_lin = 1.0 / total if total > 0 else np.inf

    return {
        "geometric_leakage": geo,
        "mismatch_leakage": channel_mismatch,
        "total_leakage": total,
        "cmrr_linear": cmrr_lin,
        "cmrr_db": 20.0 * np.log10(cmrr_lin) if np.isfinite(cmrr_lin) else np.inf,
        "geometric_limited": geo > channel_mismatch,
        "standoff_m": geom.standoff_m,
        "baseline_m": geom.baseline_m,
        **{k: r[k] for k in ("common_mode_T", "differential_T",
                             "gradient_T_per_m")},
    }


def uniform_field_cmrr_db(channel_mismatch: Optional[float] = None) -> float:
    """
    CMRR against a UNIFORM field (the Earth's), which is matching-limited only.

    Included to make explicit that the widely quoted 60 dB figure is correct
    for this case and is not being disputed -- what this module shows is that
    it does not transfer to near-field platform sources.
    """
    if channel_mismatch is None:
        channel_mismatch = value_of(CHANNEL_MISMATCH_DEFAULT)
    return 20.0 * np.log10(1.0 / channel_mismatch)


def analytic_cmrr_estimate(standoff_m: float, baseline_m: float) -> float:
    """
    Closed-form check on the numerical result: for a dipole at standoff d with
    baseline L along the same axis, B ~ 1/d^3 gives a differential-to-common
    ratio of about 3L/d, so CMRR ~ d/(3L).

    Used by the materials V&V suite as an independent comparison against the
    full vector superposition, in the same spirit as the existing V-02 check
    of the analytic gradient against finite differences.
    """
    if baseline_m <= 0:
        raise ValueError("baseline must be positive")
    return standoff_m / (3.0 * baseline_m)


# ----------------------------------------------------------------------------
# SECTION 4 -- REFERENCE VEHICLE
# ----------------------------------------------------------------------------
#
# A REMUS-100-class hull: ~1.6 m long, ~0.19 m diameter, origin at the vehicle
# centre, +x forward. Sensors are carried FORWARD of the nose (negative x in
# the geometry default) because the dominant sources are aft.
#
# The component split is an engineering assumption; the TOTAL is calibrated
# against the one measured anchor available, [Oehler24]'s 246.90 nT raw
# interference. Every moment below is therefore typed EngineeringAssumption,
# and `calibrate_to_oehler` rescales the set so the total matches measurement.

def reference_vehicle() -> VehicleMagneticModel:
    """
    Reference REMUS-class layout, UNCALIBRATED.

    Use `calibrate_to_oehler()` for the version anchored to the measured
    benchmark. Positions are body-frame metres; +x forward, +z up.
    """
    comps = (
        MagneticComponent(
            label="Stern thruster (BLDC, permanent magnets)",
            position_m=(0.80, 0.0, 0.0),
            moment_override=EngineeringAssumption(
                6.0, "A*m^2",
                "dominant single source on a small UUV. NOT computed as "
                "M_r*V: a brushless motor's magnetic circuit closes its flux "
                "through the stator, so the net EXTERNAL moment is far below "
                "the sum of its magnets'. No measured net moment for a UUV "
                "thruster located in open literature"),
            override_direction=(1.0, 0.0, 0.0)),
        MagneticComponent(
            label="Battery pack (DC current loops, cell tabs)",
            position_m=(-0.20, 0.0, -0.03),
            moment_override=EngineeringAssumption(
                2.5, "A*m^2",
                "dominated by DC current loops rather than material "
                "susceptibility, so a material-derived moment would be "
                "meaningless. Loop area x current, order-of-magnitude"),
            override_direction=(0.0, 1.0, 0.0)),
        MagneticComponent(
            label="Power electronics / motor controller",
            position_m=(0.45, 0.0, 0.02),
            moment_override=EngineeringAssumption(
                1.0, "A*m^2",
                "switching converters and their DC feeds; order-of-magnitude"),
            override_direction=(0.0, 0.0, 1.0)),
        MagneticComponent(
            label="Fin actuators (4x servo, magnets)",
            position_m=(0.70, 0.0, 0.0),
            moment_override=EngineeringAssumption(
                0.6, "A*m^2",
                "four small geared servos with permanent-magnet motors; "
                "order-of-magnitude"),
            override_direction=(0.0, 1.0, 0.0)),
        # Structural items ARE material-derived -- this is the path by which a
        # material substitution changes the answer.
        MagneticComponent(
            label="Pressure housing (Ti-6Al-4V)",
            position_m=(0.0, 0.0, 0.0),
            material=TI_6AL_4V,
            volume_m3=2.5e-3,
            remanent_direction=(1.0, 0.0, 0.0)),
        MagneticComponent(
            label="Fastener set (cold-worked 316L)",
            position_m=(0.30, 0.0, 0.0),
            material=SS316L_COLD_WORKED,
            volume_m3=1.2e-4,
            remanent_direction=(1.0, 0.0, 0.0)),
        MagneticComponent(
            label="Payload frame (CFRP)",
            position_m=(-0.50, 0.0, 0.0),
            material=CFRP_QUASI_ISOTROPIC,
            volume_m3=1.5e-3,
            remanent_direction=(1.0, 0.0, 0.0)),
    )
    return VehicleMagneticModel(comps, "reference REMUS-class AUV")


def calibrate_to_oehler(geom: Optional[GradiometerGeometry] = None,
                        target_T: float = OEHLER_RAW_INTERFERENCE_T) -> dict:
    """
    Scale the reference vehicle so its field at the baseline centre matches the
    measured raw UUV interference of 246.90 nT RMS [Oehler24].

    This is the model's one anchor to measurement. It does NOT validate the
    component split -- only the total. Returns the scaled model and the factor,
    both of which the report states explicitly.
    """
    if geom is None:
        geom = GradiometerGeometry()
    v = reference_vehicle()
    B = default_ambient_vector()
    centre = geom.standoff_m * (np.asarray(geom.direction, dtype=float)
                                / np.linalg.norm(geom.direction))
    raw = float(np.linalg.norm(v.field_at(centre, B)))
    factor = target_T / raw if raw > 0 else 1.0
    return {
        "model": v.scaled(factor),
        "scale_factor": factor,
        "uncalibrated_field_T": raw,
        "target_field_T": target_T,
        "source": "[Oehler24] raw UUV interference 246.90 nT RMS",
    }


def standoff_sweep(standoffs_m: Sequence[float],
                   baseline_m: float = 0.5,
                   calibrated: bool = True) -> list:
    """
    Derived CMRR versus sensor standoff -- the quantitative form of R-13
    ("boom or tow displacement maximising distance from thrusters and
    battery").

    Returns one row per standoff, each carrying the derived CMRR, the field at
    the cells, and which leakage mechanism binds.
    """
    if calibrated:
        cal = calibrate_to_oehler()
        v = cal["model"]
    else:
        v = reference_vehicle()

    rows = []
    for d in standoffs_m:
        geom = GradiometerGeometry(standoff_m=float(d), baseline_m=baseline_m)
        r = derived_cmrr(v, geom)
        r["analytic_cmrr_db"] = 20.0 * np.log10(
            analytic_cmrr_estimate(float(d), baseline_m))
        rows.append(r)
    return rows


def summary() -> str:
    """Text summary, for the CLI and for cross-checking the paper's numbers."""
    cal = calibrate_to_oehler()
    v = cal["model"]
    lines = [
        "GEOMETRY-DERIVED VEHICLE MAGNETIC MODEL",
        "=" * 74,
        f"vehicle            : {v.label}",
        f"calibration        : x{cal['scale_factor']:.4g} to match "
        f"{cal['target_field_T']*1e9:.2f} nT  {cal['source']}",
        f"uniform-field CMRR : {uniform_field_cmrr_db():.1f} dB "
        f"(matching-limited, R-09) -- applies to the EARTH's field",
        "",
        "component moments (|m|, A*m^2):",
    ]
    for label, deriv, mag in v.provenance_rows():
        lines.append(f"  {deriv:16s} {label:44s} {mag:10.4g}")

    lines += ["", "derived CMRR vs standoff (baseline 0.5 m):",
              f"  {'standoff':>9s} {'common':>12s} {'diff':>12s} "
              f"{'CMRR':>9s} {'analytic':>9s}  binds"]
    for r in standoff_sweep([1.0, 2.0, 3.0, 5.0, 10.0]):
        binds = "geometry" if r["geometric_limited"] else "matching"
        lines.append(
            f"  {r['standoff_m']:7.1f} m {r['common_mode_T']*1e9:10.3f} nT "
            f"{r['differential_T']*1e9:10.3f} nT {r['cmrr_db']:7.1f} dB "
            f"{r['analytic_cmrr_db']:7.1f} dB  {binds}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
