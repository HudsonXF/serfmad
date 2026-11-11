"""
serfmad.materials
=================
Material property layer for structural and magnetic-cleanliness modelling.

WHY THIS MODULE EXISTS
----------------------
Before this module, no material property existed anywhere in `serfmad`. Passive
shielding was a bare scalar multiplier (`NullingSystem.passive_factor = 100.0`)
and the gradiometer's common-mode rejection was a bare parameter
(`cmrr_db = 60.0`). Neither was derived from what anything is actually made of,
so no material choice could influence any result. This module supplies the
property data; `serfmad.physics.vehicle` turns it into a field.

PROVENANCE IS ENFORCED BY THE TYPE SYSTEM
-----------------------------------------
Following the convention established in `serfmad.constants`, every numeric
property here is wrapped in one of exactly two types:

    Measured(value, units, source)          -- traceable to a citation
    EngineeringAssumption(value, units, rationale)   -- a modelling choice

`EngineeringAssumption` is imported from `constants` rather than redefined, so
there is one such type in the codebase and downstream tables cannot confuse the
two. `provenance_report()` prints the split, and `assumed_fraction()` is
asserted in the materials V&V suite, so an uncited value cannot silently
accumulate.

A NOTE ON WHAT IS AND IS NOT KNOWN
----------------------------------
Magnetic susceptibility of engineering alloys is genuinely poorly served by the
open literature. Handbook values are usually quoted for annealed, contaminant-
free material; a real fastener has been cold-headed and thread-rolled, and a
real weld has a different phase balance from the parent plate. Where the
literature gives a number, it is `Measured`. Where it gives a range, a
direction, or nothing at all, it is an `EngineeringAssumption` that says so.
Remanent magnetisation `M_r` is the worst case: for nominally non-magnetic
structural alloys it is dominated by ferromagnetic contamination and handling
history, and NO defensible measured value exists. Every `M_r` below is
therefore an assumption, and the paper says so.

References (full citations in docs/REFERENCES.md):
  [Gohil20]      Gohil et al., arXiv:2009.01519 (2020) -- measured mu-metal
  [Dubbers86]    Dubbers, NIM A 243, 511 (1986) -- shielding factor closed form
  [Sumner87]     Sumner, Pendlebury & Smith, J. Phys. D 20, 1095 (1987)
  [Hoburg96]     Hoburg, IEEE Trans. EMC 38, 93 (1996)
  [Celozzi08]    Celozzi, Araneo & Lovat, Electromagnetic Shielding (2008)
  [ASSDA]        Australian Stainless Steel Dev. Assoc., Technical FAQ 3
  [BSSA]         British Stainless Steel Assoc., cold work and permeability
  [Electralloy50] Electralloy NITRONIC 50 (XM-19) product data sheet
  [MatWebTi64]   MatWeb, Titanium Ti-6Al-4V (Grade 5), Annealed
  [Lee08]        Lee & Romalis, J. Appl. Phys. 103, 084904 (2008)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Union

from .constants import EngineeringAssumption

# Vacuum permeability, for converting flux density to magnetising field.
# MU0_OVER_4PI is exact by definition of mu_0 in the pre-2019 SI (constants.py).
MU0: float = 4.0e-7 * 3.141592653589793


# ----------------------------------------------------------------------------
# SECTION 1 -- PROVENANCE TYPES
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Measured:
    """
    A property value traceable to a literature or standards citation.

    The counterpart to `EngineeringAssumption`. Kept as a distinct type for the
    same reason: so that a report table can never present an assumption as a
    measurement, or vice versa.
    """
    value: float
    units: str
    source: str


PropertyValue = Union[Measured, EngineeringAssumption]


def value_of(p: PropertyValue) -> float:
    """Numeric value of a property, regardless of provenance."""
    return float(p.value)


def is_assumed(p: PropertyValue) -> bool:
    """True if the property is a modelling choice rather than a citation."""
    return isinstance(p, EngineeringAssumption)


def provenance_of(p: PropertyValue) -> str:
    """Short provenance label: the citation, or the assumption's rationale."""
    return p.rationale if isinstance(p, EngineeringAssumption) else p.source


def label_of(p: PropertyValue) -> str:
    """'assumed' or 'measured' -- used directly in report tables."""
    return "assumed" if is_assumed(p) else "measured"


# ----------------------------------------------------------------------------
# SECTION 2 -- MATERIAL SCHEMA
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class MagneticMaterial:
    """
    A structural or shielding material, described by the properties that
    actually matter for a SERF magnetometer payload.

    chi_volume : volume magnetic susceptibility (SI, dimensionless).
        Sets the INDUCED moment m = chi * V * H in an ambient field. For the
        nominally non-magnetic structural alloys this is the property that
        decides whether a bracket is a magnetic source or not.
    mu_r_initial : initial relative permeability. For shield materials this is
        the primary figure of merit and is very large; for structural alloys it
        is simply 1 + chi_volume and is stored for completeness.
    M_r_A_per_m : remanent magnetisation. Sets the PERMANENT moment
        m = M_r * V, which does not vanish when the ambient field is nulled and
        is therefore the harder of the two to design against.
    sigma_S_per_m : electrical conductivity. Drives eddy-current and Johnson
        noise [Lee08]. Carried in the schema and reported, but NOT yet consumed
        by the physics -- see MODULE STATUS below.
    B_sat_T : saturation flux density. A shield that saturates stops shielding,
        so this bounds the useful ambient field for a given geometry.
    density_kg_m3, tensile_MPa : structural properties, for the mass and
        pressure-housing budget.
    thermal_conductivity_W_mK, max_service_temp_C : thermal properties, for the
        150-200 C cell-heater budget (R-07).

    MODULE STATUS
    -------------
    `chi_volume` and `M_r_A_per_m` ARE consumed by `serfmad.physics.vehicle`
    and therefore drive results. `sigma_S_per_m` and `B_sat_T` are carried and
    reported but are NOT yet consumed by any physics path: the eddy-current /
    Johnson-noise term they would drive is not implemented. They are present so
    the schema is complete and so the gap is visible rather than hidden.
    """
    label: str
    category: str                      # 'shield' | 'structural' | 'thermal'
    chi_volume: PropertyValue
    mu_r_initial: PropertyValue
    M_r_A_per_m: PropertyValue
    sigma_S_per_m: PropertyValue
    B_sat_T: PropertyValue
    density_kg_m3: PropertyValue
    tensile_MPa: PropertyValue
    thermal_conductivity_W_mK: PropertyValue
    max_service_temp_C: PropertyValue
    note: str = ""

    # -- convenience accessors -------------------------------------------
    @property
    def chi(self) -> float:
        return value_of(self.chi_volume)

    @property
    def M_r(self) -> float:
        return value_of(self.M_r_A_per_m)

    def induced_moment_magnitude(self, volume_m3: float,
                                 B_ambient_T: float) -> float:
        """
        Induced dipole moment |m| = chi * V * H, with H = B / mu_0.

        This is the term that scales with the ambient field, so it is present
        whenever the vehicle is in the Earth's field and vanishes only inside
        a shield.
        """
        return self.chi * volume_m3 * (B_ambient_T / MU0)

    def remanent_moment_magnitude(self, volume_m3: float) -> float:
        """
        Permanent dipole moment |m| = M_r * V.

        Independent of the ambient field, so nulling the ambient field does
        NOT remove it. This is why magnetic hygiene is a build-time constraint
        rather than something that can be compensated away later.
        """
        return self.M_r * volume_m3

    def provenance_rows(self) -> Tuple[Tuple[str, float, str, str], ...]:
        """(property, value, 'measured'|'assumed', source-or-rationale)."""
        fields = ("chi_volume", "mu_r_initial", "M_r_A_per_m", "sigma_S_per_m",
                  "B_sat_T", "density_kg_m3", "tensile_MPa",
                  "thermal_conductivity_W_mK", "max_service_temp_C")
        out = []
        for f in fields:
            p = getattr(self, f)
            out.append((f, value_of(p), label_of(p), provenance_of(p)))
        return tuple(out)


# ----------------------------------------------------------------------------
# SECTION 3 -- SHIELD MATERIALS
# ----------------------------------------------------------------------------
#
# The two mu-metal entries below are the same alloy in two conditions, and the
# difference between them is one of the most consequential numbers in this
# work: forming mu-metal foil into a shield and NOT re-annealing it cost
# [Gohil20] more than an order of magnitude of permeability. Re-annealing
# requires >1000 C, which is not something that can be done to an assembled
# payload. A shield deformed during handling, integration, or pressure cycling
# does not fail gracefully -- it quietly stops shielding.

MU_METAL_ANNEALED = MagneticMaterial(
    label="Mu-metal, annealed in final form",
    category="shield",
    chi_volume=Measured(
        55954.0, "dimensionless (SI)",
        "[Gohil20] chi = mu_r - 1 from the measured mu_i = 55,955 +/- 7"),
    mu_r_initial=Measured(
        55955.0, "dimensionless",
        "[Gohil20] measured on an annealed cylinder, ID 5 cm, wall 1 mm, "
        "length 50 cm; above the advertised 50,000"),
    M_r_A_per_m=EngineeringAssumption(
        0.0, "A/m",
        "mu-metal is a soft magnetic material selected for low coercivity; "
        "remanence treated as negligible relative to structural contamination. "
        "NOT a measured value"),
    sigma_S_per_m=EngineeringAssumption(
        1.7e6, "S/m",
        "typical for 80Ni-15Fe-4.5Mo permalloy; [Gohil20] gives the magnetic "
        "but not the electrical properties of its sample"),
    B_sat_T=Measured(
        0.74, "T",
        "[Gohil20] Table 1, advertised saturation induction for the mu-metal "
        "used (80Ni-15Fe-4.5Mo-0.4Mn-0.1Si)"),
    density_kg_m3=EngineeringAssumption(
        8700.0, "kg/m^3", "nominal for 80% Ni permalloy"),
    tensile_MPa=EngineeringAssumption(
        550.0, "MPa",
        "nominal annealed permalloy; mu-metal is selected magnetically, not "
        "structurally, and should not carry primary load"),
    thermal_conductivity_W_mK=EngineeringAssumption(
        19.0, "W/(m*K)", "nominal for Ni-Fe permalloy"),
    max_service_temp_C=EngineeringAssumption(
        400.0, "degC",
        "well above the 150-200 C cell-heater range (R-07); the binding "
        "thermal limit is the >1000 C re-anneal, not service temperature"),
    note="Design-point shield material. Requires annealing in FINAL form.",
)

MU_METAL_FOIL_FORMED = MagneticMaterial(
    label="Mu-metal foil, formed, not re-annealed",
    category="shield",
    chi_volume=Measured(
        4659.0, "dimensionless (SI)",
        "[Gohil20] chi = mu_r - 1 from the best of three formed foil shields"),
    mu_r_initial=Measured(
        4660.0, "dimensionless",
        "[Gohil20] measured mu_r for a shield rolled from 0.1 mm foil, "
        "D = 4.5 cm, not re-annealed; companion shields measured 3,670 and "
        "3,602. All are >10x below the annealed bulk value"),
    M_r_A_per_m=EngineeringAssumption(
        0.0, "A/m", "as MU_METAL_ANNEALED; soft magnetic material"),
    sigma_S_per_m=EngineeringAssumption(
        1.7e6, "S/m", "same alloy as MU_METAL_ANNEALED"),
    B_sat_T=Measured(
        0.74, "T", "[Gohil20] same alloy; saturation is not degraded by "
                   "cold work, only permeability is"),
    density_kg_m3=EngineeringAssumption(8700.0, "kg/m^3", "same alloy"),
    tensile_MPa=EngineeringAssumption(
        700.0, "MPa", "cold-worked foil is stronger and less permeable than "
                      "annealed strip -- the trade runs the wrong way here"),
    thermal_conductivity_W_mK=EngineeringAssumption(19.0, "W/(m*K)", "same alloy"),
    max_service_temp_C=EngineeringAssumption(400.0, "degC", "same alloy"),
    note="CAUTIONARY ENTRY. Same alloy as MU_METAL_ANNEALED, degraded by "
         "forming. Present so the schema can express the penalty.",
)


# ----------------------------------------------------------------------------
# SECTION 4 -- STRUCTURAL MATERIALS
# ----------------------------------------------------------------------------

SS316L_ANNEALED = MagneticMaterial(
    label="316L austenitic stainless, annealed",
    category="structural",
    chi_volume=Measured(
        4.0e-3, "dimensionless (SI)",
        "[ASSDA] DC permeability 1.003-1.005 at 200 Oe for annealed 316/317L; "
        "chi = mu_r - 1, midpoint of the quoted range"),
    mu_r_initial=Measured(
        1.004, "dimensionless",
        "[ASSDA] annealed 316/317L, generally < 1.02 at 200 Oe"),
    M_r_A_per_m=EngineeringAssumption(
        5.0, "A/m",
        "no measured remanence for annealed 316L located. Nominally zero for "
        "a fully austenitic structure; a small non-zero value is carried so "
        "the model does not silently assume perfection"),
    sigma_S_per_m=EngineeringAssumption(
        1.35e6, "S/m", "nominal 316L, ~74 microhm*cm resistivity"),
    B_sat_T=EngineeringAssumption(
        0.0, "T", "paramagnetic; does not saturate in any relevant field"),
    density_kg_m3=EngineeringAssumption(8000.0, "kg/m^3", "nominal 316L"),
    tensile_MPa=EngineeringAssumption(
        485.0, "MPa", "nominal annealed 316L UTS"),
    thermal_conductivity_W_mK=EngineeringAssumption(16.3, "W/(m*K)", "nominal 316L"),
    max_service_temp_C=EngineeringAssumption(870.0, "degC", "nominal 316L"),
    note="Acceptable ONLY in the annealed condition. See SS316L_COLD_WORKED.",
)

SS316L_COLD_WORKED = MagneticMaterial(
    label="316L austenitic stainless, cold worked",
    category="structural",
    chi_volume=EngineeringAssumption(
        5.0e-1, "dimensionless (SI)",
        "[BSSA] states cold work transforms austenite to strain-induced "
        "martensite and RAISES permeability, but publishes no numerical "
        "cold-worked value. A two-order-of-magnitude increase over annealed "
        "is assumed to represent a cold-headed or thread-rolled fastener. "
        "THIS IS AN ASSUMPTION, NOT A MEASUREMENT, and it is the single "
        "largest uncertainty in the vehicle model"),
    mu_r_initial=EngineeringAssumption(
        1.5, "dimensionless", "1 + chi_volume, consistent with the above"),
    M_r_A_per_m=EngineeringAssumption(
        500.0, "A/m",
        "martensite is ferromagnetic and therefore carries remanence. No "
        "measured value for a worked 316L fastener located. Assumed two "
        "orders above annealed to represent the qualitative hazard"),
    sigma_S_per_m=EngineeringAssumption(1.35e6, "S/m", "as annealed 316L"),
    B_sat_T=EngineeringAssumption(
        0.3, "T",
        "partial martensite fraction; saturation of the ferromagnetic phase "
        "only. Assumed"),
    density_kg_m3=EngineeringAssumption(8000.0, "kg/m^3", "as annealed 316L"),
    tensile_MPa=EngineeringAssumption(
        860.0, "MPa", "nominal cold-worked 316L -- stronger and magnetically "
                      "worse, the trade that makes fastener choice hard"),
    thermal_conductivity_W_mK=EngineeringAssumption(16.3, "W/(m*K)", "as annealed"),
    max_service_temp_C=EngineeringAssumption(870.0, "degC", "as annealed"),
    note="THE FASTENER TRAP. A 316L bolt is specified as non-magnetic but is "
         "cold-worked by heading and thread rolling. Prefer NITRONIC_50.",
)

NITRONIC_50 = MagneticMaterial(
    label="Nitronic 50 (XM-19, UNS S20910)",
    category="structural",
    chi_volume=Measured(
        4.0e-3, "dimensionless (SI)",
        "[Electralloy50] chi = mu_r - 1 from the quoted annealed mu_r = 1.004"),
    mu_r_initial=Measured(
        1.004, "dimensionless",
        "[Electralloy50] typical annealed permeability at 50, 100 and 200 Oe"),
    M_r_A_per_m=EngineeringAssumption(
        2.0, "A/m",
        "nitrogen-strengthened austenite does not form martensite on cold "
        "work [BSSA], so remanence is assumed lower than 316L. No measured "
        "value located"),
    sigma_S_per_m=EngineeringAssumption(
        1.2e6, "S/m", "nominal for a high-alloy austenitic stainless"),
    B_sat_T=EngineeringAssumption(0.0, "T", "paramagnetic"),
    density_kg_m3=EngineeringAssumption(7880.0, "kg/m^3", "nominal XM-19"),
    tensile_MPa=EngineeringAssumption(
        690.0, "MPa", "nominal annealed XM-19; roughly 1.4x annealed 316L"),
    thermal_conductivity_W_mK=EngineeringAssumption(11.6, "W/(m*K)", "nominal XM-19"),
    max_service_temp_C=EngineeringAssumption(870.0, "degC", "nominal XM-19"),
    note="PREFERRED FASTENER MATERIAL. [BSSA]/[Electralloy50]: retains low "
         "permeability in the COLD-WORKED condition, which 316L does not.",
)

TI_6AL_4V = MagneticMaterial(
    label="Ti-6Al-4V (Grade 5), annealed",
    category="structural",
    chi_volume=EngineeringAssumption(
        1.8e-4, "dimensionless (SI)",
        "open sources CONFLICT: values from chi ~ -1.7e-6 (diamagnetic) to "
        "~ +1.8e-4 (paramagnetic) are quoted for Ti-6Al-4V. The paramagnetic "
        "value is adopted as the conservative (worse) case. Flagged as an "
        "assumption precisely because the literature does not agree"),
    mu_r_initial=EngineeringAssumption(
        1.00018, "dimensionless", "1 + chi_volume, per the above"),
    M_r_A_per_m=EngineeringAssumption(
        0.5, "A/m",
        "titanium carries no ferromagnetic phase; remanence assumed to arise "
        "only from iron contamination (Gr5 permits 0.25% Fe max). No measured "
        "value located"),
    sigma_S_per_m=EngineeringAssumption(
        5.8e5, "S/m", "nominal Ti-6Al-4V, ~1.7 microhm*m resistivity"),
    B_sat_T=EngineeringAssumption(0.0, "T", "paramagnetic"),
    density_kg_m3=Measured(
        4430.0, "kg/m^3", "[MatWebTi64] Ti-6Al-4V Grade 5, annealed"),
    tensile_MPa=Measured(
        950.0, "MPa", "[MatWebTi64] Ti-6Al-4V Grade 5, annealed UTS"),
    thermal_conductivity_W_mK=Measured(
        6.7, "W/(m*K)", "[MatWebTi64] Ti-6Al-4V Grade 5, annealed"),
    max_service_temp_C=EngineeringAssumption(400.0, "degC", "nominal Gr5 limit"),
    note="Best structural strength-to-susceptibility ratio, and the low "
         "thermal conductivity helps the heater budget (R-07).",
)

CFRP_QUASI_ISOTROPIC = MagneticMaterial(
    label="CFRP laminate, quasi-isotropic",
    category="structural",
    chi_volume=EngineeringAssumption(
        -1.0e-5, "dimensionless (SI)",
        "carbon and epoxy are both diamagnetic; no measured laminate "
        "susceptibility located. Magnitude assumed at the diamagnetic scale. "
        "Sign is negative, so CFRP opposes rather than reinforces the "
        "ambient field"),
    mu_r_initial=EngineeringAssumption(
        0.99999, "dimensionless", "1 + chi_volume, per the above"),
    M_r_A_per_m=EngineeringAssumption(
        0.0, "A/m",
        "no ferromagnetic phase and no remanence mechanism. The strongest "
        "magnetic-cleanliness case of any structural material here"),
    sigma_S_per_m=EngineeringAssumption(
        1.0e4, "S/m",
        "HIGHLY ANISOTROPIC: in-plane (fibre-direction) conductivity is "
        "~1e4 S/m while through-thickness is ~1e-3 S/m -- seven orders of "
        "difference. A single scalar is a poor description and is retained "
        "only because no conductivity-driven physics consumes it yet"),
    B_sat_T=EngineeringAssumption(0.0, "T", "diamagnetic"),
    density_kg_m3=EngineeringAssumption(1600.0, "kg/m^3", "nominal CFRP laminate"),
    tensile_MPa=EngineeringAssumption(
        600.0, "MPa", "nominal quasi-isotropic laminate; strongly layup-dependent"),
    thermal_conductivity_W_mK=EngineeringAssumption(
        1.0, "W/(m*K)", "nominal through-thickness; also strongly anisotropic"),
    max_service_temp_C=EngineeringAssumption(
        150.0, "degC",
        "epoxy matrix glass transition. NOTE this is AT the bottom of the "
        "150-200 C cell operating range (R-07), so CFRP must not be in "
        "direct contact with the cell oven"),
    note="Magnetically the cleanest structural option. Anisotropic "
         "conductivity is a real limitation of this scalar schema.",
)

NIAL_BRONZE = MagneticMaterial(
    label="Nickel-aluminium bronze (C63000)",
    category="structural",
    chi_volume=EngineeringAssumption(
        1.0e-4, "dimensionless (SI)",
        "no measured value located. Nickel-bearing bronzes are weakly "
        "paramagnetic and the nickel content makes them measurably less "
        "clean than tin bronzes; assumed at the weak-paramagnetic scale"),
    mu_r_initial=EngineeringAssumption(
        1.0001, "dimensionless", "1 + chi_volume, per the above"),
    M_r_A_per_m=EngineeringAssumption(
        1.0, "A/m", "no ferromagnetic phase expected; assumed small"),
    sigma_S_per_m=EngineeringAssumption(
        4.3e6, "S/m",
        "nominal C63000. HIGH conductivity -- this is the material most "
        "likely to be an eddy-current and Johnson-noise problem once that "
        "term is implemented"),
    B_sat_T=EngineeringAssumption(0.0, "T", "paramagnetic"),
    density_kg_m3=EngineeringAssumption(7580.0, "kg/m^3", "nominal C63000"),
    tensile_MPa=EngineeringAssumption(620.0, "MPa", "nominal C63000"),
    thermal_conductivity_W_mK=EngineeringAssumption(42.0, "W/(m*K)", "nominal C63000"),
    max_service_temp_C=EngineeringAssumption(300.0, "degC", "nominal C63000"),
    note="Traditional non-magnetic marine bearing/fitting alloy. Its high "
         "conductivity is a liability near the cell.",
)


# ----------------------------------------------------------------------------
# SECTION 5 -- THERMAL MATERIALS (R-07: 150-200 C at +/-5 mK, non-magnetic)
# ----------------------------------------------------------------------------

SILICA_AEROGEL = MagneticMaterial(
    label="Silica aerogel blanket",
    category="thermal",
    chi_volume=EngineeringAssumption(
        -1.0e-6, "dimensionless (SI)",
        "amorphous silica is diamagnetic and the blanket is mostly air; "
        "assumed negligible. No measured value located"),
    mu_r_initial=EngineeringAssumption(
        0.999999, "dimensionless", "1 + chi_volume, per the above"),
    M_r_A_per_m=EngineeringAssumption(
        0.0, "A/m", "no ferromagnetic phase"),
    sigma_S_per_m=EngineeringAssumption(
        1.0e-12, "S/m",
        "electrical insulator -- which is exactly why it is attractive next "
        "to the cell: it cannot support eddy currents or Johnson noise"),
    B_sat_T=EngineeringAssumption(0.0, "T", "diamagnetic"),
    density_kg_m3=EngineeringAssumption(150.0, "kg/m^3", "nominal aerogel blanket"),
    tensile_MPa=EngineeringAssumption(
        0.5, "MPa", "aerogel blanket is not a structural material"),
    thermal_conductivity_W_mK=EngineeringAssumption(
        0.015, "W/(m*K)",
        "nominal for silica aerogel blanket at ambient pressure -- roughly "
        "half the conductivity of still air, which is what makes the 150-200 C "
        "budget tractable in cold seawater"),
    max_service_temp_C=EngineeringAssumption(
        650.0, "degC", "nominal silica aerogel; far above the R-07 range"),
    note="Preferred cell insulation: lowest thermal conductivity AND an "
         "electrical insulator, so it adds no Johnson noise.",
)


MATERIALS: Dict[str, MagneticMaterial] = {
    m.label: m for m in (
        MU_METAL_ANNEALED, MU_METAL_FOIL_FORMED,
        SS316L_ANNEALED, SS316L_COLD_WORKED, NITRONIC_50,
        TI_6AL_4V, CFRP_QUASI_ISOTROPIC, NIAL_BRONZE,
        SILICA_AEROGEL,
    )
}

STRUCTURAL_CANDIDATES: Tuple[MagneticMaterial, ...] = (
    SS316L_ANNEALED, SS316L_COLD_WORKED, NITRONIC_50,
    TI_6AL_4V, CFRP_QUASI_ISOTROPIC, NIAL_BRONZE,
)


# ----------------------------------------------------------------------------
# SECTION 6 -- SHIELDING FACTOR (literature closed form, [Dubbers86])
# ----------------------------------------------------------------------------

def shielding_factor(material: MagneticMaterial, diameter_m: float,
                     wall_thickness_m: float) -> float:
    """
    Transverse shielding factor of a single-layer cylindrical shield.

    [Dubbers86] gives the amplitude response of a single layer as
    T = D / (mu_r * Delta), so the shielding factor is

        S = mu_r * Delta / D

    for shield diameter D and wall thickness Delta. [Gohil20] fits exactly
    this form to measured transfer functions, and [Hoburg96] gives the fuller
    treatment; [Sumner87] covers the multi-shell case, which is NOT modelled
    here.

    This is a LITERATURE CLOSED FORM evaluated with schema properties. It is
    not a simulation result, and it is reported as such.
    """
    if diameter_m <= 0.0 or wall_thickness_m <= 0.0:
        raise ValueError("diameter and wall thickness must be positive")
    mu_r = value_of(material.mu_r_initial)
    return mu_r * wall_thickness_m / diameter_m


def required_passive_factor(total_required: float, active_factor: float) -> float:
    """
    Passive shielding needed to reach a total suppression, given the active
    nulling factor. Trivial arithmetic, kept here so the paper and the code
    cannot disagree about it.
    """
    if active_factor <= 0.0:
        raise ValueError("active factor must be positive")
    return total_required / active_factor


# ----------------------------------------------------------------------------
# SECTION 7 -- PROVENANCE REPORTING
# ----------------------------------------------------------------------------

def assumed_fraction() -> float:
    """Fraction of all schema property values that are assumptions."""
    total = assumed = 0
    for m in MATERIALS.values():
        for _, _, lab, _ in m.provenance_rows():
            total += 1
            assumed += (lab == "assumed")
    return assumed / total if total else 0.0


def provenance_report() -> str:
    """
    Human-readable provenance dump. Every value is labelled measured or
    assumed, per the project rule that a modelling choice must never be
    presentable as a measurement.
    """
    lines = ["SERF-MAD MATERIAL SCHEMA -- PROVENANCE",
             "=" * 78]
    for m in MATERIALS.values():
        rows = m.provenance_rows()
        n_assumed = sum(1 for r in rows if r[2] == "assumed")
        lines.append("")
        lines.append(f"{m.label}  [{m.category}]")
        lines.append(f"  {n_assumed}/{len(rows)} values assumed")
        for name, val, lab, src in rows:
            flag = "ASSUMED " if lab == "assumed" else "measured"
            lines.append(f"    {flag} {name:26s} = {val:<12.6g} {src[:60]}")
    lines.append("")
    lines.append("-" * 78)
    lines.append(f"OVERALL: {assumed_fraction()*100:.1f}% of property values "
                 f"are engineering assumptions, not measurements.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(provenance_report())
