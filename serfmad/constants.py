"""
serfmad.constants
=================
Physical constants and literature-anchored parameter sets.

EVERY numeric value in this module carries a citation. Values without a
traceable source are not permitted here -- if a number is a modelling choice
rather than a measured quantity, it is placed in `EngineeringAssumption`
and flagged as such so it can never be mistaken for a validated datum.

References (full citations in docs/REFERENCES.md):
  [Allred02]   Allred, Lyman, Kornack & Romalis, PRL 89, 130801 (2002)
  [Kominis03]  Kominis, Kornack, Allred & Romalis, Nature 422, 596 (2003)
  [Savukov05]  Savukov & Romalis, Phys. Rev. A 71, 023405 (2005)
  [Dang10]     Dang, Maloof & Romalis, Appl. Phys. Lett. 97, 151110 (2010)
  [Griffith10] Griffith, Knappe & Kitching, Opt. Express 18, 27167 (2010)
  [Shah07]     Shah, Knappe, Schwindt & Kitching, Nature Photonics 1, 649 (2007)
  [Borna22]    Borna et al., NeuroImage 247, 118818 (2022)  -- CAPE
  [Seltzer08]  Seltzer, PhD thesis, Princeton (2008)
  [Oehler24]   Oehler et al., Geophysical Prospecting 72 (2024)
  [Wei23]      Wei et al., Front. Phys. 11:1181399 (2023)
  [NCEI]       NOAA National Centers for Environmental Information
"""

# PEP 604 unions ("float | None") are evaluated at class-creation
# time in dataclass fields, which is a TypeError on Python < 3.10.
# This import makes all annotations lazy strings, so the module
# imports cleanly on 3.9 as well.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple
import math

# ----------------------------------------------------------------------------
# SECTION 1 -- FUNDAMENTAL CONSTANTS (SI)
# ----------------------------------------------------------------------------

MU0_OVER_4PI: float = 1.0e-7          # T*m/A, exact by definition of mu_0 (pre-2019 SI)
HBAR: float = 1.054_571_817e-34       # J*s, CODATA
BOHR_MAGNETON: float = 9.274_010_078e-24  # J/T, CODATA
K_BOLTZMANN: float = 1.380_649e-23    # J/K, exact (SI 2019)

# Bare electron gyromagnetic ratio.
# Cross-validated two ways:
#   (a) stated directly as gamma_e/2pi = 28 Hz/nT for Rb            [Wei23]
#   (b) back-calculated from the measured fit gamma_e/q = 2pi*6.7 Hz/nT
#       at q = 4.17 for 39-K, giving gamma_e/2pi = 27.9 Hz/nT       [Savukov05]
# Agreement <1% between independent sources.
GAMMA_E_OVER_2PI: float = 28.0e9      # Hz/T  (== 28 Hz/nT)
GAMMA_E: float = 2.0 * math.pi * GAMMA_E_OVER_2PI   # rad/(s*T)


# ----------------------------------------------------------------------------
# SECTION 2 -- ALKALI SPECIES
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class AlkaliSpecies:
    """Alkali-metal working medium for the vapour cell."""
    name: str
    nuclear_spin_I: float
    q_low_pol: float          # slowing-down factor at low polarisation
    q_high_pol: float         # slowing-down factor at P -> 1
    source: str

    def slowing_down_factor(self, polarisation: float) -> float:
        """
        Nuclear slowing-down factor q(P).

        For I=3/2 alkalis q ranges 6 (low P) -> 4 (P->1) [Savukov05, Seltzer08].
        Linear interpolation in P is a modelling approximation to the true
        (nonlinear) dependence and is flagged as such in the V&V report; the
        true form follows from the hyperfine Bloch equations [Savukov05].
        """
        p = min(max(polarisation, 0.0), 1.0)
        return self.q_low_pol + (self.q_high_pol - self.q_low_pol) * p


RB87 = AlkaliSpecies("Rb-87", 1.5, 6.0, 4.0, "[Savukov05],[Seltzer08]")
K39 = AlkaliSpecies("K-39", 1.5, 6.0, 4.0, "[Savukov05]")
CS133 = AlkaliSpecies("Cs-133", 3.5, 16.0, 8.0, "[Seltzer08] (I=7/2 branch)")

SPECIES: Dict[str, AlkaliSpecies] = {s.name: s for s in (RB87, K39, CS133)}


# ----------------------------------------------------------------------------
# SECTION 3 -- SENSOR PERFORMANCE ENVELOPE (measured devices, for V&V anchoring)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SensorDatum:
    """A single measured sensor performance point from the literature."""
    label: str
    noise_floor_T_rtHz: float
    cell_volume_mm3: float | None
    note: str
    source: str


# These are MEASURED DEVICE RESULTS, used as validation anchors. The simulated
# sensor must fall inside this envelope or the model is wrong.
SENSOR_ENVELOPE: Tuple[SensorDatum, ...] = (
    SensorDatum("Allred 2002 (K, bulk)", 10e-15, None,
                "1.1 Hz linewidth at n=1e14 cm^-3; shield-Johnson-noise limited",
                "[Allred02]"),
    SensorDatum("Kominis 2003 (multichannel)", 1e-15, None,
                "sub-fT multichannel; theoretical limit quoted <0.01 fT/rtHz",
                "[Kominis03]"),
    SensorDatum("Dang 2010 (bulk, best)", 0.16e-15, None,
                "160 aT/rtHz -- best reported atomic magnetometer",
                "[Dang10]"),
    SensorDatum("Griffith 2010 (MEMS)", 5e-15, 1.0,
                "microfabricated 1 mm^3 Si cell, anodically bonded Pyrex windows",
                "[Griffith10]"),
    SensorDatum("Hybrid MEMS cell (2024)", 20e-15, None,
                "single-beam hybrid Cs-Rb optical pumping, microfab cell",
                "Microsystems & Nanoeng. 2024"),
)

# The MEMS cell is the relevant design point for a rugged AUV payload:
# it is the only form factor that survives shock/vibration and fits a UUV.
MEMS_DESIGN_POINT = SENSOR_ENVELOPE[3]   # Griffith 2010, 5 fT/rtHz, 1 mm^3


# ----------------------------------------------------------------------------
# SECTION 4 -- THE BINDING CONSTRAINT: SERF FIELD TOLERANCE
# ----------------------------------------------------------------------------

# This is the single most important number in this entire project.
# SERF operation requires the spin-exchange rate to dominate the Larmor
# precession rate, which caps the tolerable residual STATIC field.
SERF_RESIDUAL_FIELD_TOLERANCE_T: float = 5e-9   # +/-5 nT  [Borna22]
SERF_TOLERANCE_SOURCE = "[Borna22] 'The SERF regime accommodates remnant " \
                        "static magnetic fields up to +/-5 nT'"

# Earth's field is 4 orders of magnitude larger. Any airborne/underwater SERF
# deployment MUST actively null the ambient field into the SERF window.
EARTH_FIELD_RANGE_T: Tuple[float, float] = (25e-6, 65e-6)     # [NCEI], [BGS]
EARTH_FIELD_NOMINAL_T: float = 50e-6                          # mid-latitude

# Required ambient-field suppression factor (dimensionless).
REQUIRED_NULLING_FACTOR: float = EARTH_FIELD_NOMINAL_T / SERF_RESIDUAL_FIELD_TOLERANCE_T
# == 10,000x  -> i.e. 80 dB of combined passive shielding + active nulling.


# ----------------------------------------------------------------------------
# SECTION 5 -- TARGET LIBRARY (hardest-to-detect signatures)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetClass:
    """
    A magnetic target with a literature-traceable dipole moment.

    `difficulty` is an ordinal ranking used only for GUI presentation; the
    physics uses `moment_A_m2` alone.
    """
    label: str
    moment_A_m2: float
    typical_standoff_m: float
    difficulty: int          # 1 = easy ... 5 = hardest
    source: str

    def peak_field_at(self, r_m: float) -> float:
        """On-axis peak |B| of a point dipole at range r (T). B = 2*(mu0/4pi)*m/r^3."""
        return 2.0 * MU0_OVER_4PI * self.moment_A_m2 / (r_m ** 3)


# Ordered easiest -> hardest. The project's stated goal is the bottom entries.
TARGET_LIBRARY: Tuple[TargetClass, ...] = (
    TargetClass(
        "Submarine, undegaussed", 1.0e7, 500.0, 1,
        "~1e7 A*m^2 for a 100 m x 10 m hull (Atlantis Press, differential MAD study)"),
    TargetClass(
        "Submarine, degaussed", 5.0e5, 400.0, 2,
        "simulated MAD studies draw moments from [1,9]x10^5 A*m^2 "
        "(ResearchGate degaussing/anomaly studies); open sources quote "
        "1000-2000 nT residual signature at 200-600 m range"),
    TargetClass(
        "Large UXO / naval mine", 5.0e2, 10.0, 3,
        "BASTA project: AUV magnetometry resolves <10 nT anomalies from "
        "seabed munitions (Geophysical Prospecting 2024, [Oehler24])"),
    TargetClass(
        "Small UXO / ordnance cluster", 5.0e1, 5.0, 4,
        "order-of-magnitude scaling from [Oehler24] sub-10 nT class; "
        "ENGINEERING ESTIMATE, not a directly measured moment"),
    TargetClass(
        "Adversary micro-UUV, low-signature hull", 5.0e0, 5.0, 5,
        "ENGINEERING ESTIMATE. No open measured moment located for "
        "low-signature micro-UUVs; bounded from below by the <10 nT "
        "detection class demonstrated in [Oehler24]. Treat as a "
        "requirements-driving hypothetical, NOT a validated datum."),
)

TARGETS: Dict[str, TargetClass] = {t.label: t for t in TARGET_LIBRARY}


# ----------------------------------------------------------------------------
# SECTION 6 -- PLATFORM / ENVIRONMENT (engineering assumptions, clearly flagged)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class EngineeringAssumption:
    """
    A modelling choice, NOT a literature-measured value.

    Kept in a distinct type so that no downstream code or report table can
    silently present an assumption as a validated constant.
    """
    value: float
    units: str
    rationale: str


PLATFORM_ASSUMPTIONS: Dict[str, EngineeringAssumption] = {
    "auv_speed": EngineeringAssumption(
        2.0, "m/s", "typical REMUS-class survey speed (3-4 kn)"),
    "roll_amplitude": EngineeringAssumption(
        math.radians(3.0), "rad", "moderate sea-state induced roll for a small UUV"),
    "roll_frequency": EngineeringAssumption(
        0.25, "Hz", "wave encounter frequency, 4 s period swell"),
    "pitch_amplitude": EngineeringAssumption(
        math.radians(1.5), "rad", "pitch typically smaller than roll for slender hulls"),
    "pitch_frequency": EngineeringAssumption(
        0.4, "Hz", "distinct from roll so the two are separable in frequency"),
    "gradiometer_baseline": EngineeringAssumption(
        0.5, "m", "practical rigid baseline inside/along a UUV hull"),
    "platform_self_noise": EngineeringAssumption(
        250e-9, "T", "[Oehler24] report raw UUV interference of 246.9 nT RMS "
                     "before compensation, reduced to 2.23 nT after"),
    "platform_noise_residual": EngineeringAssumption(
        2.23e-9, "T", "[Oehler24] post-compensation residual -- a MEASURED "
                      "benchmark our processing chain must match or beat"),
}

# Post-compensation residual from a real UUV magnetic survey. This is the
# number our processing chain is benchmarked against in the V&V suite.
OEHLER_RAW_INTERFERENCE_T: float = 246.90e-9
OEHLER_COMPENSATED_RESIDUAL_T: float = 2.23e-9
OEHLER_SOURCE = "[Oehler24] / UUV vector magnetometer compensation study"


def summary_table() -> str:
    """Human-readable dump of the binding constraints, for the GUI and logs."""
    lines = [
        "SERF-MAD CONSTANT SET (all values literature-anchored)",
        "-" * 62,
        f"gamma_e/2pi              : {GAMMA_E_OVER_2PI/1e9:.1f} Hz/nT   [Wei23],[Savukov05]",
        f"SERF field tolerance     : +/-{SERF_RESIDUAL_FIELD_TOLERANCE_T*1e9:.1f} nT     [Borna22]",
        f"Earth field (nominal)    : {EARTH_FIELD_NOMINAL_T*1e9:.0f} nT   [NCEI]",
        f"=> required nulling      : {REQUIRED_NULLING_FACTOR:,.0f}x  "
        f"({20*math.log10(REQUIRED_NULLING_FACTOR):.0f} dB)",
        f"MEMS design point        : {MEMS_DESIGN_POINT.noise_floor_T_rtHz*1e15:.0f} fT/rtHz "
        f"@ {MEMS_DESIGN_POINT.cell_volume_mm3:.0f} mm^3  [Griffith10]",
        f"UUV interference (raw)   : {OEHLER_RAW_INTERFERENCE_T*1e9:.1f} nT  [Oehler24]",
        f"UUV interference (comp.) : {OEHLER_COMPENSATED_RESIDUAL_T*1e9:.2f} nT  [Oehler24]",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary_table())
