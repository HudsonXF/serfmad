"""
serfmad.materials_sweep
=======================
Parameter sweeps over the material layer, and the figures for the companion
materials paper.

WRITES ONLY TO `outputs_materials/`. The existing `outputs/` directory is the
published figure set for `docs/SR_report.tex` and is never touched by this
module.

    python -m serfmad.materials_sweep

WHAT IS SWEPT, AND WHAT KIND OF NUMBER EACH SWEEP PRODUCES
----------------------------------------------------------
  1. Derived CMRR vs sensor standoff        -- SIMULATED (new physics path)
  2. Fastener material substitution         -- SIMULATED (new physics path)
  3. Passive shielding vs mu_r and geometry -- LITERATURE CLOSED FORM
                                               ([Dubbers86] evaluated with
                                               schema properties; NOT a
                                               simulation result)
  4. Detection range under derived vs assumed CMRR
                                            -- SIMULATED, via the EXISTING
                                               and unmodified
                                               `mission.monte_carlo_detection`,
                                               called with a different CMRR.

Sweep 4 deliberately changes nothing but the CMRR argument, so any difference
is attributable to the derived value alone.
"""

from __future__ import annotations

import os
import numpy as np

from .constants import (
    EARTH_FIELD_NOMINAL_T, SERF_RESIDUAL_FIELD_TOLERANCE_T,
    OEHLER_COMPENSATED_RESIDUAL_T, GAMMA_E,
)
from .materials import (
    MU_METAL_ANNEALED, MU_METAL_FOIL_FORMED, STRUCTURAL_CANDIDATES,
    SS316L_COLD_WORKED, NITRONIC_50, TI_6AL_4V, CFRP_QUASI_ISOTROPIC,
    SS316L_ANNEALED, NIAL_BRONZE,
    shielding_factor, value_of, is_assumed, required_passive_factor,
)
from .physics import vehicle as veh

OUTDIR_DEFAULT = "outputs_materials"

# The report's derived requirement: <=0.94 nT residual for <=5% CAPE gain loss.
CAPE_REQUIREMENT_T = 0.94e-9
REQUIRED_TOTAL_SUPPRESSION = EARTH_FIELD_NOMINAL_T / CAPE_REQUIREMENT_T
ACTIVE_NULLING_FACTOR = 100.0     # the report's baseline active stage


# ---------------------------------------------------------------------------
# SWEEP 1 -- derived CMRR vs standoff
# ---------------------------------------------------------------------------

def sweep_standoff(standoffs=None, baseline_m=0.5):
    """Derived CMRR versus sensor standoff. SIMULATED."""
    if standoffs is None:
        standoffs = np.concatenate([np.linspace(0.5, 3.0, 26),
                                    np.linspace(3.2, 12.0, 45)])
    return veh.standoff_sweep(list(standoffs), baseline_m=baseline_m)


# ---------------------------------------------------------------------------
# SWEEP 2 -- fastener material substitution
# ---------------------------------------------------------------------------

def sweep_fastener_material(volume_m3=1.2e-4, standoff_m=2.0):
    """
    Vehicle field contribution from an identical fastener set in each candidate
    material. SIMULATED via the new physics path.

    Everything except the material is held fixed, so the spread is entirely
    attributable to material choice -- which is the claim the material layer
    exists to support.
    """
    p = np.array([-standoff_m, 0.0, 0.0])
    rows = []
    for mat in STRUCTURAL_CANDIDATES:
        v = veh.VehicleMagneticModel((veh.MagneticComponent(
            label="fasteners", position_m=(0.30, 0.0, 0.0),
            material=mat, volume_m3=volume_m3,
            remanent_direction=(1.0, 0.0, 0.0)),), mat.label)
        B = float(np.linalg.norm(v.field_at(p)))
        m = float(np.linalg.norm(v.total_moment()))
        rows.append({
            "material": mat.label,
            "chi": mat.chi,
            "M_r": mat.M_r,
            "moment_A_m2": m,
            "field_at_sensor_T": B,
            "chi_assumed": is_assumed(mat.chi_volume),
            "M_r_assumed": is_assumed(mat.M_r_A_per_m),
        })
    return rows


# ---------------------------------------------------------------------------
# SWEEP 3 -- passive shielding against the 94 dB requirement
# ---------------------------------------------------------------------------

def sweep_shielding(diameters_m=None, thicknesses_m=None):
    """
    Shielding factor from the [Dubbers86] closed form, evaluated with schema
    permeabilities, against the passive factor the 0.94 nT CAPE requirement
    implies.

    THIS IS A LITERATURE CLOSED FORM, NOT A SIMULATION RESULT. It is reported
    as such in the paper.
    """
    if diameters_m is None:
        diameters_m = np.linspace(0.04, 0.30, 60)
    if thicknesses_m is None:
        thicknesses_m = (0.5e-3, 1.0e-3, 2.0e-3)

    need = required_passive_factor(REQUIRED_TOTAL_SUPPRESSION,
                                   ACTIVE_NULLING_FACTOR)
    out = {"required_passive_factor": need,
           "required_total": REQUIRED_TOTAL_SUPPRESSION,
           "curves": []}
    for mat in (MU_METAL_ANNEALED, MU_METAL_FOIL_FORMED):
        for t in thicknesses_m:
            S = np.array([shielding_factor(mat, float(d), float(t))
                          for d in diameters_m])
            out["curves"].append({
                "material": mat.label,
                "thickness_m": float(t),
                "diameter_m": np.asarray(diameters_m, dtype=float),
                "shielding_factor": S,
                "meets_requirement": S >= need,
            })
    return out


# ---------------------------------------------------------------------------
# SWEEP 4 -- detection impact of the derived CMRR
# ---------------------------------------------------------------------------

def sweep_detection_cmrr(target="Large UXO / naval mine",
                         standoffs=(1.0, 2.0, 3.0, 5.0, 10.0),
                         assumed_cmrr_db=60.0,
                         n_trials=16384, seed=0,
                         range_min_m=2.0, range_max_m=300.0):
    """
    Detection range under the DERIVED CMRR at each standoff, against the
    assumed 60 dB baseline.

    Calls the EXISTING, UNMODIFIED `mission.monte_carlo_detection`. The only
    argument that changes between rows is `cmrr_db`, so the difference is
    attributable to the derived value alone. GPU-accelerated; gated on
    `main.py --gpu-verify` passing.
    """
    from .mission import monte_carlo_detection

    # Range bounds match those used by the existing stress suite (S-1/S-2), so
    # the assumed-CMRR row is directly comparable to the R50 figures published
    # in SR_report.tex Table 6.
    mc = dict(target_label=target, n_trials=n_trials, seed=seed,
              range_min_m=range_min_m, range_max_m=range_max_m)

    base = monte_carlo_detection(cmrr_db=assumed_cmrr_db, **mc)
    rows = [{
        "standoff_m": None,
        "cmrr_db": assumed_cmrr_db,
        "r50_m": float(base["r50_m"]),
        "kind": "assumed (report baseline)",
    }]
    for d in standoffs:
        geom = veh.GradiometerGeometry(standoff_m=float(d), baseline_m=0.5)
        cal = veh.calibrate_to_oehler()
        c = veh.derived_cmrr(cal["model"], geom)
        r = monte_carlo_detection(cmrr_db=float(c["cmrr_db"]), **mc)
        rows.append({
            "standoff_m": float(d),
            "cmrr_db": float(c["cmrr_db"]),
            "r50_m": float(r["r50_m"]),
            "kind": "derived from geometry",
        })
    return rows


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------

def make_figures(outdir=OUTDIR_DEFAULT):
    """Generate the materials-paper figures into `outdir` (never `outputs/`)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .gui import style

    if os.path.abspath(outdir).rstrip("\\/").endswith(os.sep + "outputs"):
        raise RuntimeError("refusing to write into the published outputs/ dir")
    os.makedirs(outdir, exist_ok=True)
    style.use_style()

    # ---------------- Figure 1: CMRR vs standoff -----------------------
    rows = sweep_standoff()
    d = np.array([r["standoff_m"] for r in rows])
    cm = np.array([r["cmrr_db"] for r in rows])
    an = np.array([r["analytic_cmrr_db"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
    ax = axes[0]
    ax.plot(d, cm, color=style.SIGNAL, lw=2.0,
            label="derived (vector superposition)")
    ax.plot(d, an, color=style.MUTED, lw=1.2, ls="--",
            label=r"analytic $d/(3L)$")
    ax.axhline(60.0, color=style.CLUTTER, lw=1.6, ls=":",
               label="assumed 60 dB (R-09)")
    ax.set_xlabel("sensor standoff from vehicle centre, $d$ (m)")
    ax.set_ylabel("common-mode rejection (dB)")
    ax.set_title("Derived vs assumed CMRR", loc="left", fontsize=10.5)
    ax.legend(fontsize=7.6, frameon=False, loc="lower right")
    ax.grid(True, color=style.GRID, lw=0.5)
    ax.annotate(f"{cm[np.argmin(np.abs(d-2.0))]:.1f} dB at 2 m",
                xy=(2.0, cm[np.argmin(np.abs(d - 2.0))]),
                xytext=(3.4, 26.0), fontsize=8, color=style.INK,
                arrowprops=dict(arrowstyle="->", color=style.MUTED, lw=0.9))

    # panel b: which mechanism binds
    ax = axes[1]
    geo = np.array([r["geometric_leakage"] for r in rows])
    mis = np.array([r["mismatch_leakage"] for r in rows])
    ax.loglog(d, geo, color=style.CLUTTER, lw=2.0, label="geometric leakage")
    ax.loglog(d, mis, color=style.GOOD, lw=1.6, ls="--",
              label="channel mismatch (0.1%)")
    ax.set_xlabel("sensor standoff, $d$ (m)")
    ax.set_ylabel("common-mode leakage fraction")
    ax.set_title("Which mechanism binds", loc="left", fontsize=10.5)
    ax.legend(fontsize=7.6, frameon=False)
    ax.grid(True, which="both", color=style.GRID, lw=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_mat_cmrr_standoff.png"), dpi=200)
    plt.close(fig)

    # ---------------- Figure 2: material substitution ------------------
    mrows = sweep_fastener_material()
    labels = [r["material"].replace(" austenitic stainless", "")
                            .replace(" laminate, quasi-isotropic", "")
                            .replace(" (Grade 5), annealed", " Gr5")
                            .replace("Nickel-aluminium bronze (C63000)",
                                     "Ni-Al bronze")
              for r in mrows]
    fld = np.array([r["field_at_sensor_T"] for r in mrows]) * 1e12
    assumed = [r["M_r_assumed"] for r in mrows]

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    cols = [style.CLUTTER if a else style.SIGNAL for a in assumed]
    bars = ax.barh(range(len(labels)), fld, color=cols, alpha=0.88, height=0.62)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"field at sensor from an identical fastener set (pT)")
    ax.set_title("Fastener material substitution, all else held fixed",
                 loc="left", fontsize=10.5)
    ax.grid(True, axis="x", which="both", color=style.GRID, lw=0.4)
    for i, (b, v) in enumerate(zip(bars, fld)):
        ax.text(v * 1.25, i, f"{v:,.3g} pT", va="center", fontsize=7.6,
                color=style.INK)
    ax.text(0.99, -0.19,
            "all bars red: every $M_r$ here is an ENGINEERING ASSUMPTION, "
            "not a measured value",
            transform=ax.transAxes, ha="right", fontsize=7.2,
            color=style.CLUTTER)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_mat_fastener.png"), dpi=200)
    plt.close(fig)

    # ---------------- Figure 3: shielding vs requirement ----------------
    sh = sweep_shielding()
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    styles = {0.5e-3: ":", 1.0e-3: "-", 2.0e-3: "--"}
    for c in sh["curves"]:
        ann = "annealed" in c["material"]
        ax.plot(c["diameter_m"] * 100.0, c["shielding_factor"],
                color=style.SIGNAL if ann else style.CLUTTER,
                ls=styles[c["thickness_m"]], lw=1.7,
                label=f"{'annealed' if ann else 'formed foil'}, "
                      f"{c['thickness_m']*1e3:.1f} mm")
    ax.axhline(sh["required_passive_factor"], color=style.ACCENT, lw=2.0)
    ax.text(0.02, sh["required_passive_factor"] * 1.25,
            f"required passive stage: {sh['required_passive_factor']:.0f}x "
            f"(0.94 nT CAPE requirement with 100x active)",
            transform=ax.get_yaxis_transform(), fontsize=7.8,
            color=style.ACCENT)
    ax.set_yscale("log")
    ax.set_xlabel("shield diameter (cm)")
    ax.set_ylabel(r"transverse shielding factor $S=\mu_r\Delta/D$")
    ax.set_title(r"Passive shielding against the 94 dB requirement "
                 r"(literature closed form, [Dubbers86])",
                 loc="left", fontsize=10.0)
    ax.legend(fontsize=7.4, frameon=False, ncol=2)
    ax.grid(True, which="both", color=style.GRID, lw=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_mat_shielding.png"), dpi=200)
    plt.close(fig)

    # ---------------- Figure 4: detection impact ------------------------
    det = sweep_detection_cmrr()
    base = det[0]
    pts = det[1:]
    dd = np.array([r["standoff_m"] for r in pts])
    rr = np.array([r["r50_m"] for r in pts])
    cc = np.array([r["cmrr_db"] for r in pts])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
    ax = axes[0]
    ax.plot(dd, rr, "o-", color=style.SIGNAL, lw=2.0, ms=5.5,
            label="derived CMRR from geometry")
    ax.axhline(base["r50_m"], color=style.CLUTTER, lw=1.8, ls=":",
               label=f"assumed 60 dB: $R_{{50}}$ = {base['r50_m']:.1f} m")
    ax.set_xlabel("sensor standoff, $d$ (m)")
    ax.set_ylabel(r"$R_{50}$ (m)")
    ax.set_title("Detection range under derived CMRR", loc="left", fontsize=10.5)
    ax.legend(fontsize=7.6, frameon=False, loc="center right")
    ax.grid(True, color=style.GRID, lw=0.5)
    ax.set_ylim(0, base["r50_m"] * 1.12)
    for x, y in zip(dd, rr):
        ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=7.2,
                    color=style.INK)

    ax = axes[1]
    ax.plot(cc, rr, "o-", color=style.ACCENT, lw=2.0, ms=5.5)
    ax.axhline(base["r50_m"], color=style.CLUTTER, lw=1.8, ls=":")
    ax.axvline(60.0, color=style.CLUTTER, lw=1.2, ls=":")
    ax.set_xlabel("common-mode rejection (dB)")
    ax.set_ylabel(r"$R_{50}$ (m)")
    ax.set_title("Range vs rejection, naval-mine class", loc="left",
                 fontsize=10.5)
    ax.grid(True, color=style.GRID, lw=0.5)
    ax.set_xlim(0, 66)
    ax.set_ylim(0, base["r50_m"] * 1.12)
    ax.annotate("assumed", xy=(60, base["r50_m"]), xytext=(-46, -22),
                textcoords="offset points", fontsize=7.6,
                color=style.CLUTTER,
                arrowprops=dict(arrowstyle="->", color=style.CLUTTER, lw=0.9))
    ax.annotate("derived, 1-10 m standoff", xy=(cc.mean(), rr.mean()),
                xytext=(10, 34), textcoords="offset points", fontsize=7.6,
                color=style.ACCENT)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_mat_detection.png"), dpi=200)
    plt.close(fig)

    return outdir


def summary(outdir=OUTDIR_DEFAULT) -> str:
    """Text summary of every sweep, with provenance labels."""
    L = ["SERF-MAD MATERIAL LAYER -- SWEEP RESULTS", "=" * 78, ""]

    L.append("1. DERIVED CMRR vs STANDOFF  [SIMULATED]")
    for r in veh.standoff_sweep([1.0, 2.0, 3.0, 5.0, 10.0]):
        L.append(f"   d = {r['standoff_m']:5.1f} m   "
                 f"CMRR = {r['cmrr_db']:6.1f} dB   "
                 f"(assumed 60 dB; shortfall "
                 f"{60.0 - r['cmrr_db']:5.1f} dB)")

    L += ["", "2. FASTENER MATERIAL SUBSTITUTION  [SIMULATED]"]
    for r in sweep_fastener_material():
        flag = "ASSUMED M_r" if r["M_r_assumed"] else "measured M_r"
        L.append(f"   {r['material'][:44]:44s} "
                 f"{r['field_at_sensor_T']*1e12:10.4g} pT   {flag}")

    sh = sweep_shielding()
    L += ["", "3. PASSIVE SHIELDING  [LITERATURE CLOSED FORM, not simulated]",
          f"   required total suppression : {sh['required_total']:,.0f}x "
          f"({20*np.log10(sh['required_total']):.1f} dB)",
          f"   required passive stage     : {sh['required_passive_factor']:,.0f}x "
          f"(with {ACTIVE_NULLING_FACTOR:.0f}x active)"]
    for mat, D, t in ((MU_METAL_ANNEALED, 0.05, 1e-3),
                      (MU_METAL_ANNEALED, 0.10, 1e-3),
                      (MU_METAL_FOIL_FORMED, 0.05, 1e-3),
                      (MU_METAL_FOIL_FORMED, 0.10, 1e-3)):
        S = shielding_factor(mat, D, t)
        ok = "MEETS" if S >= sh["required_passive_factor"] else "FAILS"
        L.append(f"   {mat.label[:38]:38s} D={D*100:4.0f} cm t={t*1e3:.1f} mm "
                 f"-> S={S:9.1f}x  {ok}")
    return "\n".join(L)


def main():
    print(summary())
    out = make_figures()
    print(f"\nfigures written to {out}/")


if __name__ == "__main__":
    main()
