"""
serfmad.gui.hero
================
"Hero" figures: matte-shaded 3D surfaces for quantities that genuinely live on
a two-dimensional parameter plane.

The rule applied throughout: 3D is used ONLY where the third dimension carries
real information. None of these could be honestly reduced to a line plot
without discarding a variable.

  H1  Sensor response surface  P_y(Bx, By)        -- the SERF transfer function
  H2  Target anomaly field     B_z(x, y)          -- dipole over a survey plane
  H3  CAPE penalty surface     gain(B_res, R_tot) -- the design constraint
  H4  Detection surface        Pd(range, moment)  -- operational performance

Run:  python -m serfmad.gui.hero --outdir outputs
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .style import (use_style, matte_surface, clean_3d, panel_label,
                    annotate_corner, QUALITY, EMBER, TIDE, SIGNAL, CLUTTER,
                    ACCENT, GOOD, WARN, INK, MUTED, latex_available)
from ..quantum.bloch import BlochConfig
from ..physics.dipole import dipole_field
from ..physics.platform import cape_error
from ..constants import GAMMA_E, SERF_RESIDUAL_FIELD_TOLERANCE_T, TARGETS


def _bloch_steady_2d(Bx, By, cfg: BlochConfig):
    """
    Exact steady-state transverse response for an arbitrary transverse field.

    Solves gamma_e (P x B) + R_op*s_hat - R*P = 0 with B = (Bx, By, 0) and
    pumping along z, componentwise, on the grid. Written out explicitly rather
    than looped so the whole surface evaluates as array arithmetic.
    """
    g, Rop = cfg.gamma_e, cfg.R_op
    R1, R2 = cfg.R_1, cfg.R_2
    wx, wy = g * Bx, g * By
    # Steady state of the 3-component linear system.
    denom = R1 * R2 ** 2 + R2 * (wx ** 2 + wy ** 2)
    Px = Rop * (wy * R2) / denom
    Py = Rop * (-wx * R2) / denom
    Pz = Rop * R2 ** 2 / denom
    return Px, Py, Pz


def h1_sensor_response(fig, cfg: BlochConfig | None = None, n=260, quality='publication', **kw):
    """H1: the SERF transfer function over the transverse field plane."""
    cfg = cfg or BlochConfig()
    lim = 6e-9
    b = np.linspace(-lim, lim, n)
    BX, BY = np.meshgrid(b, b)
    _, Py, _ = _bloch_steady_2d(BX, BY, cfg)

    ax = fig.add_subplot(111, projection="3d")
    # 'overlay' retains far more chroma than 'soft' on a diverging map, which
    # otherwise desaturates to near-grey across the whole mid-range.
    matte_surface(ax, BX * 1e9, BY * 1e9, Py, cmap=TIDE, vert_exag=0.7,
                  azdeg=310, altdeg=55, blend_mode="overlay",
                  max_cells=QUALITY[quality])
    clean_3d(ax, elev=30, azim=-128, box_aspect=(1, 1, 0.55))
    ax.set_xlabel(r"$B_x$ (nT)", labelpad=8)
    ax.set_ylabel(r"$B_y$ (nT)", labelpad=8)
    ax.set_zlabel(r"transverse polarisation $P_y$", labelpad=10, rotation=90)
    fig.suptitle(r"SERF transfer function: dispersive response over the "
                 r"transverse field plane", y=0.97)
    annotate_corner(ax, f"HWHM $= {cfg.B_hwhm*1e9:.2f}$ nT\n"
                        f"$R_{{op}}={cfg.R_op:.0f}$ s$^{{-1}}$, "
                        f"$q={cfg.q:.0f}$", loc="upper right")
    return fig


def h2_anomaly_field(fig, target="Large UXO / naval mine", depth=12.0, n=300, quality='publication', **kw):
    """H2: dipole anomaly over the survey plane the AUV flies."""
    tgt = TARGETS[target]
    x = np.linspace(-40, 40, n)
    y = np.linspace(-30, 30, n)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    r = np.stack([X, Y, Z + depth], axis=-1)      # sensor plane above target
    m = np.array([0.0, 0.0, tgt.moment_A_m2])
    B = dipole_field(r, m)
    Bz = B[..., 2] * 1e9                           # nT

    # A dipole spans orders of magnitude across the plane: on a linear z-axis
    # the peak crushes the negative side-lobes to invisibility, which are
    # physically the informative part of a MAD signature. Signed cube-root
    # scaling preserves sign and relative structure while compressing range.
    Bz_s = np.sign(Bz) * np.abs(Bz) ** (1.0 / 3.0)

    ax = fig.add_subplot(111, projection="3d")
    matte_surface(ax, X, Y, Bz_s, cmap=EMBER, vert_exag=1.0,
                  azdeg=320, altdeg=46, blend_mode="overlay",
                  max_cells=QUALITY[quality])
    ticks = np.array([-20, -5, 0, 5, 20, 50])
    ax.set_zticks(np.sign(ticks) * np.abs(ticks) ** (1 / 3))
    ax.set_zticklabels([f"{t:g}" for t in ticks])
    clean_3d(ax, elev=38, azim=-120, box_aspect=(1.25, 1, 0.5))
    ax.set_xlabel(r"along-track $x$ (m)", labelpad=8)
    ax.set_ylabel(r"cross-track $y$ (m)", labelpad=8)
    ax.set_zlabel(r"anomaly $B_z$ (nT, signed cube-root scale)",
                  labelpad=14, rotation=90)
    fig.suptitle(rf"Magnetic anomaly of a {tgt.moment_A_m2:.0f}"
                 r"$\,$A$\cdot$m$^2$ dipole at "
                 rf"{depth:.0f}$\,$m standoff", y=0.97)
    annotate_corner(ax, rf"peak $|B_z| = {np.abs(Bz).max():.1f}$ nT",
                    loc="upper right")
    return fig


def h3_cape_surface(fig, n=220, quality='publication', **kw):
    """
    H3: CAPE gain penalty over (residual field, total relaxation rate).

    This is the report's central design constraint rendered as a surface: the
    ridge line where gain = 0.95 IS the requirement.
    """
    Bres = np.logspace(-11, -8, n)
    Rtot = np.linspace(100.0, 1500.0, n)
    BB, RR = np.meshgrid(Bres, Rtot)
    alpha = np.arctan2(GAMMA_E * BB, RR)
    gain = np.cos(alpha)

    ax = fig.add_subplot(111, projection="3d")
    matte_surface(ax, np.log10(BB * 1e9), RR, gain, cmap=TIDE,
                  vert_exag=1.0, azdeg=300, altdeg=50,
                  max_cells=QUALITY[quality])
    clean_3d(ax, elev=28, azim=-58, box_aspect=(1.15, 1, 0.55))
    ax.set_xlabel(r"$\log_{10}\,B_{\mathrm{res}}$ (nT)", labelpad=8)
    ax.set_ylabel(r"$R_{\mathrm{tot}}$ (s$^{-1}$)", labelpad=8)
    ax.set_zlabel(r"CAPE gain factor", labelpad=10, rotation=90)
    fig.suptitle(r"Cross-axis projection error: gain penalty surface", y=0.97)

    # The 0.95 iso-contour is the requirement; draw it on the surface.
    Rline = np.linspace(100.0, 1500.0, 200)
    Breq = np.tan(np.arccos(0.95)) * Rline / GAMMA_E
    ax.plot(np.log10(Breq * 1e9), Rline, np.full_like(Rline, 0.95),
            color=GOOD, lw=2.2, zorder=10,
            label=r"requirement: $5\%$ gain loss")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.92))
    annotate_corner(ax, r"nominal $\pm 5$ nT window $\Rightarrow$ gain $\approx 0.5$",
                    loc="upper right")
    return fig


def h4_detection_surface(fig, n_r=260, n_m=240, quality='publication', **kw):
    """
    H4: detection probability over (range, dipole moment).

    Computed from the closed-form gradiometric signal against the clutter
    floor, so the surface is physics, not an interpolation of scattered runs.
    """
    from ..constants import MU0_OVER_4PI
    ranges = np.linspace(3.0, 220.0, n_r)
    _ = kw.get("target")   # target enters only through the moment axis here
    moments = np.logspace(0.3, 5.0, n_m)
    RG, MM = np.meshgrid(ranges, moments)

    baseline, speed = 0.5, 2.0
    # peak along-track gradient of an on-axis dipole ~ 3*mu0/4pi*m/r^4
    grad = 3.0 * MU0_OVER_4PI * MM / RG ** 4
    clutter = (2.23e-9 / 1000.0) / baseline          # 60 dB CMRR
    sensor = 5e-15 * np.sqrt(10.0) * np.sqrt(2) / baseline
    noise = np.hypot(clutter, sensor)
    # matched-filter processing gain over a 512-sample transit
    d = grad / noise * np.sqrt(512) / 40.0
    # Logistic stand-in for the CFAR detector curve. The transition is sharp,
    # so the grid must be fine (260x240) -- a coarse grid aliases the cliff
    # into visible striping rather than a clean edge.
    Pd = 1.0 / (1.0 + np.exp(-(d - 4.0)))

    ax = fig.add_subplot(111, projection="3d")
    matte_surface(ax, RG, np.log10(MM), Pd, cmap=EMBER, vert_exag=0.8,
                  azdeg=325, altdeg=52, blend_mode="overlay",
                  max_cells=QUALITY[quality])
    clean_3d(ax, elev=30, azim=-132, box_aspect=(1.2, 1, 0.55))
    ax.set_xlabel(r"target range (m)", labelpad=8)
    ax.set_ylabel(r"$\log_{10}\,m$ (A$\cdot$m$^2$)", labelpad=8)
    ax.set_zlabel(r"detection probability $P_d$", labelpad=10, rotation=90)
    fig.suptitle(r"Detection performance surface (clutter-limited, "
                 r"$P_{fa}=10^{-4}$)", y=0.97)
    annotate_corner(ax, r"$R_{50}\propto m^{1/4}$ (gradiometer, $r^{-4}$)",
                    loc="upper right")
    return fig


HEROES = {
    "hero_sensor_response": h1_sensor_response,
    "hero_anomaly_field": h2_anomaly_field,
    "hero_cape_surface": h3_cape_surface,
    "hero_detection_surface": h4_detection_surface,
}


def generate_all(outdir="outputs", formats=("png",), figsize=(8.6, 6.0),
                 quality="publication"):
    import os
    os.makedirs(outdir, exist_ok=True)
    usetex = use_style()
    made = []
    for name, fn in HEROES.items():
        fig = plt.figure(figsize=figsize)
        fn(fig, quality=quality)
        # 3D axes leave large dead margins by default; pull them in hard.
        # 3D axes leave large dead margins by default; pull them in hard, but
        # leave headroom for the suptitle (top=1.06 clips it entirely).
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.05, top=1.0)
        for f in formats:
            p = os.path.join(outdir, f"{name}.{f}")
            fig.savefig(p, format=f)
            made.append(p)
            print(f"  wrote {p}")
        plt.close(fig)
    print(f"  (LaTeX typesetting: {'ON' if usetex else 'OFF -- mathtext fallback'})")
    return made


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--pdf", action="store_true", help="also write vector PDFs")
    a = ap.parse_args()
    fmts = ("png", "pdf") if a.pdf else ("png",)
    print("Generating hero figures...")
    generate_all(a.outdir, formats=fmts)
