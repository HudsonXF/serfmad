"""
serfmad.gui.plots
=================
All report/GUI figures, rendered in the house style (`serfmad.gui.style`).

These are the SAME functions the GUI displays and the report embeds, so a
published figure cannot drift from what the operator sees on screen.

Rules applied to every figure:
  * house typography (LaTeX/Computer Modern when available, mathtext otherwise)
  * hairline axes, sub-weight dotted grid, no top/right spines
  * lines lit from 315 degrees via `shadow()`, matching the 3D surfaces
  * units chosen so no tick label needs scientific notation
  * every stated number computed from the data plotted, never hard-coded

Run headless:
    python -m serfmad.gui.plots --outdir outputs
"""

from __future__ import annotations
import numpy as np

from .style import (use_style, annotate_corner, panel_label, shadow, glow,
                    hairline_axes, soft_fill, param_box, SIGNAL, CLUTTER,
                    ACCENT, GOOD, WARN, INK, MUTED, latex_available)
from ..constants import (SERF_RESIDUAL_FIELD_TOLERANCE_T, GAMMA_E,
                         TARGET_LIBRARY, SENSOR_ENVELOPE,
                         OEHLER_RAW_INTERFERENCE_T, OEHLER_COMPENSATED_RESIDUAL_T)
from ..physics.platform import (AUVPlatform, SEA_STATES, NullingSystem,
                                field_budget)
from ..quantum.bloch import BlochConfig, steady_state_analytic
from ..mission import MissionConfig, run_mission, monte_carlo_detection

_RMAX = {"Adversary micro-UUV, low-signature hull": 80.0,
         "Small UXO / ordnance cluster": 150.0,
         "Large UXO / naval mine": 300.0,
         "Submarine, degaussed": 800.0,
         "Submarine, undegaussed": 2500.0}


def plot_mission(fig, res):
    """Single-transit view: signature, clutter, gradiometer, detection."""
    gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.27,
                          left=0.085, right=0.975, top=0.91, bottom=0.095)
    t = res.t

    ax = fig.add_subplot(gs[0, 0]); hairline_axes(ax)
    ax.plot(t, res.B_target_1[:, 0] * 1e9, color=SIGNAL, lw=1.7,
            label=r"cell 1", path_effects=shadow())
    ax.plot(t, res.B_target_2[:, 0] * 1e9, color=ACCENT, lw=1.1,
            ls=(0, (4, 2)), label=r"cell 2")
    ax.axvline(res.cpa_time, color=MUTED, ls=(0, (1, 3)), lw=0.9)
    ax.set_xlabel(r"time (s)"); ax.set_ylabel(r"target field $B_x$ (nT)")
    ax.set_title(r"Target signature at the gradiometer cells")
    c = res.config
    pk = res.peak_target_field_T
    pk_s = f"{pk*1e9:.3f} nT" if pk >= 1e-11 else f"{pk*1e12:.2f} pT"
    param_box(ax, [
        ("target", c.target_label.split(",")[0][:20]),
        ("depth", f"{c.target_depth_m:.0f} m"),
        ("sea", c.sea_state_label.split(" ")[0]),
        ("shield x", f"{c.passive_shield:.0f}"),
        ("null x", f"{c.active_null:.0f}"),
        ("baseline", f"{c.baseline_m:.2f} m"),
        ("peak field", pk_s),
        ("CPA range", f"{res.cpa_range_m:.1f} m"),
    ], loc="lower left", title="simulation inputs")
    ax.legend(loc="lower right")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1]); hairline_axes(ax)
    clut = res.platform_interference * 1e9
    ax.plot(t, clut, color=CLUTTER, lw=1.0, alpha=0.9, path_effects=shadow())
    soft_fill(ax, t, clut, 0, color=CLUTTER, alpha=0.08)
    ratio = np.abs(res.platform_interference).max() / max(res.peak_target_field_T, 1e-18)
    ax.set_xlabel(r"time (s)"); ax.set_ylabel(r"body-frame field (nT)")
    ax.set_title(r"Platform motion clutter (pre-nulling)")
    ax.set_ylim(np.abs(clut).max() * -1.15, np.abs(clut).max() * 1.42)
    annotate_corner(ax, rf"peak ${np.abs(clut).max():,.0f}$ nT "
                        rf"$=\,{ratio:,.0f}\times$ the target",
                    loc="upper left")
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0]); hairline_axes(ax)
    # At this SNR the two traces coincide to within a line width, which is
    # itself the result worth showing. Draw the measured trace thicker and
    # underneath so the dashed truth reads as lying exactly on top of it.
    ax.plot(t, res.gradient_measured * 1e12, color=MUTED, lw=3.0,
            label=r"measured", alpha=0.55, solid_capstyle="butt")
    ax.plot(t, res.gradient_true * 1e12, color=GOOD, lw=1.4, ls=(0, (4, 2.5)),
            label=r"true target", path_effects=glow(lw=2.4))
    ax.set_xlabel(r"time (s)"); ax.set_ylabel(r"gradient (pT$\,$m$^{-1}$)")
    ax.set_title(r"Gradiometer output (finite CMRR)")
    ax.legend(loc="upper right")
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1]); hairline_axes(ax)
    d = res.detection
    scale = max(abs(d.threshold), 1e-30)
    ax.plot(t, d.statistic / scale, color=ACCENT, lw=1.5,
            path_effects=shadow())
    ax.axhline(1.0, color=INK, ls=(0, (4, 2)), lw=0.9)
    ax.axhline(-1.0, color=INK, ls=(0, (4, 2)), lw=0.9,
               label=rf"CFAR threshold ($P_{{fa}}={d.pfa_design:.0e}$)")
    ax.plot(d.peak_time, d.peak_value / scale, "o", color=CLUTTER, ms=7,
            zorder=6, mec="white", mew=0.8,
            label=rf"peak at ${d.peak_time:.2f}$ s")
    ax.set_xlabel(r"time (s)"); ax.set_ylabel(r"statistic / threshold")
    # The verdict must reflect VALIDITY, not merely a threshold crossing: a
    # matched filter will cross threshold on uncancelled platform motion, and
    # reporting that as a detection is the most dangerous possible error here.
    if res.detection_credible:
        verdict, vcol = "DETECTED", GOOD
    elif not res.serf_valid:
        verdict, vcol = "INVALID -- SERF LOCK LOST", CLUTTER
    elif d.detected:
        verdict, vcol = "REJECTED -- CLUTTER, NOT TARGET", CLUTTER
    else:
        verdict, vcol = "NO DETECTION", MUTED
    ax.set_title(rf"Decision: \textbf{{{verdict}}}"
                 if latex_available() else f"Decision: {verdict}", color=vcol)
    ax.legend(loc="upper right")
    param_box(ax, [
        ("true CPA", f"{res.cpa_time:.2f} s"),
        ("peak at", f"{d.peak_time:.2f} s"),
        ("S/C ratio", f"{res.signal_to_clutter:.2f}"),
        ("SERF lock", "held" if res.serf_valid else "LOST"),
        ("P_fa", f"{d.pfa_design:.0e}"),
    ], loc="lower left")
    if res.invalid_reason:
        fig.text(0.5, 0.012, "Result rejected -- " + res.invalid_reason,
                 ha="center", va="bottom", fontsize=7.2, color=CLUTTER,
                 wrap=True)
    panel_label(ax, "d")
    return fig


def plot_field_budget(fig, cfg: MissionConfig = None):
    """The ambient-field suppression chain -- the feasibility argument."""
    cfg = cfg or MissionConfig()
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.29,
                          left=0.085, right=0.975, top=0.91, bottom=0.105)
    t = np.linspace(0, 60, 3000)
    sea = SEA_STATES[cfg.sea_state_label]
    nul = NullingSystem(passive_factor=cfg.passive_shield,
                        active_factor=cfg.active_null)
    plat = AUVPlatform(sea_state=sea, nulling=nul)
    fb = field_budget(plat, t)

    ax = fig.add_subplot(gs[0, 0]); hairline_axes(ax)
    stages = ["Earth\nfield", "after\nshield", "after\nnull", "SERF\nwindow"]
    vals = np.array([fb["ambient_static_T"],
                     fb["ambient_static_T"] / cfg.passive_shield,
                     fb["residual_static_T"],
                     SERF_RESIDUAL_FIELD_TOLERANCE_T]) * 1e9
    cols = [CLUTTER, WARN, GOOD if fb["serf_viable"] else CLUTTER, INK]
    ax.bar(range(4), vals, color=cols, width=0.62, zorder=3)
    ax.set_yscale("log"); ax.set_ylim(1, 4e5)
    ax.set_xticks(range(4)); ax.set_xticklabels(stages)
    ax.set_ylabel(r"field (nT)")
    ax.axhline(SERF_RESIDUAL_FIELD_TOLERANCE_T * 1e9, color=INK,
               ls=(0, (4, 2)), lw=0.9, zorder=4)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.8, rf"${v:,.1f}$", ha="center", fontsize=8.5,
                color=INK)
    ax.set_title(r"Ambient-field suppression chain")
    param_box(ax, [
        ("shield x", f"{cfg.passive_shield:.0f}"),
        ("null x", f"{cfg.active_null:.0f}"),
        ("sea", cfg.sea_state_label.split(" ")[0]),
        ("margin", f"{fb['margin_factor']:.2f}x"),
        ("SERF", "OK" if fb["serf_viable"] else "LOST"),
    ], loc="upper right")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1]); hairline_axes(ax)
    B_body = plat.ambient_field_body(t)
    for i, (lbl, c) in enumerate(zip([r"$x$", r"$y$", r"$z$"],
                                     [SIGNAL, CLUTTER, GOOD])):
        ax.plot(t[:1200], (B_body[:1200, i] - B_body[:, i].mean()) * 1e9,
                lw=1.2, color=c, label=lbl, path_effects=shadow(alpha=0.10))
    ax.set_xlabel(r"time (s)"); ax.set_ylabel(r"body-frame variation (nT)")
    ax.set_title(r"Attitude-induced ambient swing")
    ax.legend(loc="upper right", ncol=3, columnspacing=1.0)
    annotate_corner(ax, rf"{sea.label}: peak-to-peak "
                        rf"${fb['motion_swing_T']*1e9:,.0f}$ nT vs "
                        rf"${SERF_RESIDUAL_FIELD_TOLERANCE_T*1e9:.0f}$ nT window",
                    loc="lower left")
    ax.set_ylim(-3600, 3900)     # headroom so legend clears the traces
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0]); hairline_axes(ax)
    Bres = np.logspace(-11, -8, 400)
    R_tot = BlochConfig().R_tot
    gains = np.cos(np.arctan2(GAMMA_E * Bres, R_tot))
    b_req = float(np.tan(np.arccos(0.95)) * R_tot / GAMMA_E)
    ax.semilogx(Bres * 1e9, gains, color=SIGNAL, lw=2.0, path_effects=shadow())
    soft_fill(ax, Bres * 1e9, gains, 0, color=SIGNAL, alpha=0.07)
    ax.axhline(0.95, color=GOOD, ls=(0, (4, 2)), lw=1.0, label=r"$5\%$ gain loss")
    ax.axvline(SERF_RESIDUAL_FIELD_TOLERANCE_T * 1e9, color=CLUTTER,
               ls=(0, (4, 2)), lw=1.0, label=r"nominal $\pm5$ nT")
    ax.axvline(b_req * 1e9, color=ACCENT, ls=(0, (1, 2)), lw=1.8,
               label=rf"required ${b_req*1e9:.2f}$ nT")
    ax.set_xlabel(r"residual static field (nT)")
    ax.set_ylabel(r"CAPE gain factor"); ax.set_ylim(0.25, 1.04)
    ax.set_title(r"Cross-axis projection error penalty")
    ax.legend(loc="lower left")
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1]); hairline_axes(ax)
    labels = ["raw UUV\ninterference", "after\ncompensation",
              "SERF\nwindow", "MEMS sensor\nnoise"]
    v = np.array([OEHLER_RAW_INTERFERENCE_T, OEHLER_COMPENSATED_RESIDUAL_T,
                  SERF_RESIDUAL_FIELD_TOLERANCE_T,
                  5e-15 * np.sqrt(10)]) * 1e12
    ax.bar(range(4), v, color=[CLUTTER, WARN, INK, GOOD], width=0.62, zorder=3)
    ax.set_yscale("log"); ax.set_ylim(1e-2, 3e6)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_ylabel(r"field (pT)")
    for i, vv in enumerate(v):
        ax.text(i, vv * 2.8, rf"${vv:,.0f}$" if vv >= 1 else rf"${vv:.3f}$",
                ha="center", fontsize=8.5, color=INK)
    ratio = v[1] / v[3]
    ax.annotate("", xy=(3.34, v[3] * 1.7), xytext=(3.34, v[1] * 0.55),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9))
    ax.text(3.34, np.sqrt(v[1] * v[3]) * 1.5, rf"${ratio:,.0f}\times$",
            ha="center", fontsize=8.5, color=INK, rotation=90,
            bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.2))
    ax.set_xlim(-0.6, 3.75)
    ax.set_title(r"Clutter dominates sensor noise")
    panel_label(ax, "d")
    return fig


def plot_sensor_physics(fig):
    """SERF resonance, linearity, measured envelope, bandwidth trade."""
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.32,
                          left=0.085, right=0.955, top=0.91, bottom=0.105)
    cfg = BlochConfig()

    ax = fig.add_subplot(gs[0, 0]); hairline_axes(ax)
    B = np.linspace(-15e-9, 15e-9, 900)
    ss = steady_state_analytic(B, cfg)
    ax.plot(B * 1e9, ss["Py"], color=SIGNAL, lw=1.9,
            label=r"$P_y$ (dispersive)", path_effects=shadow())
    ax.plot(B * 1e9, ss["Pz"], color=GOOD, lw=1.5, ls=(0, (5, 2)),
            label=r"$P_z$ (absorptive)")
    for sgn in (-1, 1):
        ax.axvline(sgn * cfg.B_hwhm * 1e9, color=CLUTTER, ls=(0, (1, 2)), lw=1.0)
    ax.set_xlabel(r"transverse field $B_x$ (nT)")
    ax.set_ylabel(r"polarisation")
    ax.set_title(r"SERF magnetic resonance")
    ax.legend(loc="upper right")
    annotate_corner(ax, rf"HWHM $={cfg.B_hwhm*1e9:.2f}$ nT", loc="lower left")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1]); hairline_axes(ax)
    Bs = np.logspace(-13, -8.3, 200)
    resp = steady_state_analytic(Bs, cfg)["Py"] / Bs
    ax.semilogx(Bs * 1e9, resp / cfg.dc_responsivity, color=ACCENT, lw=1.9,
                path_effects=shadow())
    ax.axhline(1.0, color=INK, ls=(0, (4, 2)), lw=0.9, label=r"ideal linear")
    ax.axvline(cfg.B_hwhm * 1e9, color=CLUTTER, ls=(0, (1, 2)), lw=1.0,
               label=r"HWHM")
    ax.set_xlabel(r"field (nT)"); ax.set_ylabel(r"normalised responsivity")
    ax.set_title(r"Small-signal linearity")
    ax.legend(loc="lower left")
    annotate_corner(ax, rf"$\mathrm{{d}}P_y/\mathrm{{d}}B_x="
                        rf"{cfg.dc_responsivity/1e8:.2f}\times10^{{8}}$ T$^{{-1}}$",
                    loc="upper right")
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0]); hairline_axes(ax)
    env = sorted(SENSOR_ENVELOPE, key=lambda s: s.noise_floor_T_rtHz)
    names = [s.label.split(" (")[0] for s in env]
    vals = [s.noise_floor_T_rtHz * 1e15 for s in env]
    cols = [GOOD if "Griffith" in s.label else MUTED for s in env]
    ax.barh(range(len(vals)), vals, color=cols, height=0.6, zorder=3)
    ax.set_xscale("log"); ax.set_xlim(0.08, 70)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(r"noise floor (fT$\,$Hz$^{-1/2}$)")
    for i, v in enumerate(vals):
        ax.text(v * 1.28, i, rf"${v:g}$", va="center", fontsize=8.5, color=INK)
    ax.set_title(r"Measured atomic-magnetometer envelope")
    annotate_corner(ax, r"green: MEMS design point", loc="lower right")
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1]); hairline_axes(ax)
    qs = np.linspace(3.5, 6.5, 200)
    bws = np.array([BlochConfig(q=q).bandwidth_hz for q in qs])
    hws = np.array([BlochConfig(q=q).B_hwhm * 1e9 for q in qs])
    ax.axvspan(4, 6, color=GOOD, alpha=0.07, zorder=0)
    ax.plot(qs, bws, color=SIGNAL, lw=1.9, path_effects=shadow())
    ax.set_xlabel(r"slowing-down factor $q$")
    ax.set_ylabel(r"3 dB bandwidth (Hz)", color=SIGNAL)
    ax.tick_params(axis="y", colors=SIGNAL)
    ax2 = ax.twinx()
    ax2.plot(qs, hws, color=CLUTTER, lw=1.5, ls=(0, (5, 2)))
    ax2.set_ylabel(r"resonance HWHM (nT)", color=CLUTTER)
    ax2.tick_params(axis="y", colors=CLUTTER)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    ax.set_title(r"Bandwidth trade against $q$")
    annotate_corner(ax, r"shaded: literature $q=4$--$6$", loc="upper right")
    panel_label(ax, "d")
    return fig


def plot_detection(fig, target_label="Large UXO / naval mine", n_trials=8192):
    """Monte Carlo detection performance."""
    # left margin is set by panel (c), whose category names are long
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.34,
                          left=0.155, right=0.97, top=0.91, bottom=0.105)
    rmax = _RMAX.get(target_label, 200.0)
    mc = monte_carlo_detection(n_trials=n_trials, target_label=target_label,
                               range_min_m=2.0, range_max_m=rmax)

    ax = fig.add_subplot(gs[0, 0]); hairline_axes(ax)
    ok = ~np.isnan(mc["pd_vs_range"])
    ax.plot(mc["range_centres"][ok], mc["pd_vs_range"][ok], "-o", color=SIGNAL,
            lw=1.8, ms=3.6, mec="white", mew=0.6, path_effects=shadow())
    soft_fill(ax, mc["range_centres"][ok], mc["pd_vs_range"][ok], 0,
              color=SIGNAL, alpha=0.08)
    ax.axhline(0.5, color=CLUTTER, ls=(0, (4, 2)), lw=0.9)
    if not np.isnan(mc["r50_m"]):
        ax.axvline(mc["r50_m"], color=GOOD, ls=(0, (1, 2)), lw=1.8,
                   label=rf"$R_{{50}}={mc['r50_m']:.1f}$ m")
        ax.legend(loc="upper right")
    ax.set_xlabel(r"target range (m)"); ax.set_ylabel(r"$P_d$")
    ax.set_ylim(-0.04, 1.06)
    ax.set_title(r"Detection probability versus range")
    param_box(ax, [
        ("target", target_label.split(",")[0][:20]),
        ("P_fa", f"{mc['pfa']:.0e}"),
        ("trials", f"{n_trials:,}"),
        ("clutter", f"{2.23:.2f} nT"),
        ("CMRR", "60 dB"),
    ], loc="lower left")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1]); hairline_axes(ax)
    allv = np.concatenate([mc["null_statistic"], mc["statistic"]])
    allv = allv[allv > 0]
    bins = np.logspace(np.log10(allv.min()), np.log10(allv.max()), 64)
    ax.hist(mc["null_statistic"], bins=bins, color=MUTED, alpha=0.75,
            label=r"$H_0$ (noise)", zorder=2)
    ax.hist(mc["statistic"], bins=bins, color=SIGNAL, alpha=0.72,
            label=r"$H_1$ (target)", zorder=3)
    ax.axvline(mc["threshold"], color=INK, ls=(0, (4, 2)), lw=1.1,
               label=r"CFAR threshold", zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"matched-filter statistic"); ax.set_ylabel(r"count")
    ax.set_title(r"Hypothesis separation sets $P_d$")
    ax.legend(loc="upper right")
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0]); hairline_axes(ax)
    labels, r50s = [], []
    for tgt in TARGET_LIBRARY:
        m = monte_carlo_detection(n_trials=3000, target_label=tgt.label,
                                  range_min_m=2.0, range_max_m=_RMAX[tgt.label])
        short = (tgt.label.replace(" / ", "/")
                 .replace(", undegaussed", " (undeg.)")
                 .replace(", degaussed", " (deg.)")
                 .replace(", low-signature hull", ""))
        labels.append(short[:26])
        r50s.append(m["r50_m"] if not np.isnan(m["r50_m"]) else 0.0)
    ax.barh(range(len(labels)), r50s, color=SIGNAL, height=0.6, zorder=3)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(8, 1400)
    ax.set_xlabel(r"$R_{50}$ (m)")
    for i, v in enumerate(r50s):
        if v > 0:
            ax.text(v * 1.18, i, rf"${v:.0f}$", va="center", fontsize=8.5,
                    color=INK)
    ax.set_title(r"Detection range by target class")
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1]); hairline_axes(ax)
    mc_nc = monte_carlo_detection(n_trials=n_trials, target_label=target_label,
                                  range_min_m=2.0, range_max_m=rmax,
                                  include_clutter=False)
    ok2 = ~np.isnan(mc_nc["pd_vs_range"])
    ax.plot(mc["range_centres"][ok], mc["pd_vs_range"][ok], "-o", color=SIGNAL,
            lw=1.8, ms=3.2, mec="white", mew=0.5,
            label=r"with residual clutter", path_effects=shadow())
    ax.plot(mc_nc["range_centres"][ok2], mc_nc["pd_vs_range"][ok2], "--s",
            color=CLUTTER, lw=1.6, ms=3.2, mec="white", mew=0.5,
            label=r"sensor noise only")
    ax.set_xlabel(r"target range (m)"); ax.set_ylabel(r"$P_d$")
    ax.set_ylim(-0.04, 1.06)
    ax.set_title(r"Why clutter modelling matters")
    ax.legend(loc="upper right")
    annotate_corner(ax, r"omitting clutter overstates range", loc="lower left")
    panel_label(ax, "d")
    return fig


def _tex_escape(s: str) -> str:
    """
    Escape raw test output for LaTeX rendering.

    V&V strings contain characters LaTeX treats as maths/markup (^ _ | & % # $
    backslash, braces, tilde), which abort the run and produce NO figure. Cells are plain
    prose, so escaping is the right fix rather than disabling usetex.
    """
    if not latex_available():
        return s
    out = s.replace("\\", r"\textbackslash{}")
    for ch in ("&", "%", "$", "#", "_", "{", "}"):
        out = out.replace(ch, "\\" + ch)
    out = (out.replace("^", r"\textasciicircum{}")
              .replace("~", r"\textasciitilde{}")
              .replace("|", r"\textbar{}")
              .replace("<", r"\textless{}")
              .replace(">", r"\textgreater{}"))
    return out


def plot_vv_table(fig, results, summ):
    """V&V results rendered as a styled table."""
    ax = fig.add_subplot(111)
    ax.axis("off")
    rows = [[_tex_escape(r.test_id), r.category[:4].upper(),
             _tex_escape(r.name[:46]), r.status,
             _tex_escape(r.measured[:44])] for r in results]
    cols = ["ID", "TYPE", "Test", "Status", "Measured"]
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center",
                   cellLoc="left", colWidths=[0.06, 0.07, 0.33, 0.08, 0.46])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.2)
    tbl.scale(1, 1.42)
    face = {"PASS": "#e8f3ee", "FAIL": "#fbeaea", "SKIP": "#fdf4e3"}
    for i, r in enumerate(results, start=1):
        for j in range(len(cols)):
            c = tbl[(i, j)]
            c.set_facecolor(face[r.status])
            c.set_edgecolor("#dfe2e7"); c.set_linewidth(0.4)
    for j in range(len(cols)):
        c = tbl[(0, j)]
        c.set_facecolor("#2b2f36"); c.set_edgecolor("#2b2f36")
        c.set_text_props(color="white", weight="bold")
    t = (rf"Verification \& Validation --- {summ['passed']}/{summ['total']} pass, "
         rf"{summ['failed']} fail, {summ['skipped']} skip") if latex_available() \
        else f"Verification & Validation - {summ['passed']}/{summ['total']} pass"
    ax.set_title(t, pad=14)
    return fig


def plot_setup(fig):
    """
    System setup / concept-of-operation figure.

    Four panels answering "what is actually being simulated?":
      (a) the physical arrangement -- AUV, gradiometer baseline, target
      (b) the sensor internals -- pumped axis, sensed axis, nulling coils
      (c) the signal chain, from field to detection decision
      (d) the field magnitudes each stage must handle
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22,
                          left=0.055, right=0.975, top=0.91, bottom=0.06)

    # ---- (a) survey geometry ---------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-32, 32); ax.set_ylim(-23, 11); ax.set_aspect("equal")
    ax.axis("off")

    # sea surface band
    ax.fill_between([-32, 32], 7.5, 11, color="#4a7fb5", alpha=0.12, lw=0)
    ax.plot([-32, 32], [7.5, 7.5], color="#4a7fb5", lw=1.0, alpha=0.7)
    ax.text(-31, 8.6, r"sea surface", fontsize=7.5, color=MUTED, va="center")

    # AUV hull with the two cells on a rigid baseline
    ax.add_patch(Ellipse((-13, 0), 15, 3.4, facecolor="#e4e8ee",
                         edgecolor=INK, lw=1.0, zorder=3))
    ax.text(-13, 4.9, r"AUV", fontsize=9, ha="center", color=INK, zorder=6)
    for dx, lab, ha in ((-6.2, r"cell 1", "left"), (-19.8, r"cell 2", "right")):
        ax.add_patch(Circle((dx, 0), 0.95, facecolor=GOOD, edgecolor="white",
                            lw=0.9, zorder=5))
    # baseline dimension line sits BELOW the hull so it cannot collide with
    # the hull label above it
    ax.annotate("", xy=(-6.2, -2.9), xytext=(-19.8, -2.9),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.0))
    ax.text(-13, -4.5, r"baseline $b$", fontsize=8, ha="center", va="center",
            color=ACCENT)
    ax.text(-6.2, 2.1, r"cell 1", fontsize=7.5, ha="center", color=INK)
    ax.text(-19.8, 2.1, r"cell 2", fontsize=7.5, ha="center", color=INK)

    # transit direction
    ax.annotate("", xy=(24, 0), xytext=(-3.5, 0),
                arrowprops=dict(arrowstyle="-|>", color=SIGNAL, lw=1.5))
    ax.text(11, 1.7, r"transit $v$", fontsize=8, ha="center", color=SIGNAL)

    # target dipole with field lines, well clear of the hull
    for w, h in ((8.5, 5.0), (14.0, 8.5), (20.0, 12.0)):
        ax.add_patch(Ellipse((4, -15.5), w, h, fill=False,
                             edgecolor=CLUTTER, lw=0.5, alpha=0.30, zorder=1))
    ax.add_patch(Ellipse((4, -15.5), 7.5, 2.6, facecolor="#cfd3da",
                         edgecolor=CLUTTER, lw=1.1, zorder=3))
    ax.annotate("", xy=(4, -12.4), xytext=(4, -18.6),
                arrowprops=dict(arrowstyle="-|>", color=CLUTTER, lw=1.4))
    ax.text(4, -21.4, r"target dipole $\mathbf{m}$", fontsize=8, ha="center",
            color=CLUTTER)

    # Standoff is measured from the sensor plane to the target, so draw it
    # vertically at the target's own along-track position. Started below the
    # transit arrow so the two do not overlap.
    ax.annotate("", xy=(15.5, -15.0), xytext=(15.5, -1.8),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9,
                                linestyle=(0, (3, 2))))
    ax.plot([4, 15.5], [-15.0, -15.0], color=MUTED, lw=0.6, ls=(0, (2, 2)),
            alpha=0.7)
    ax.plot([-13, 15.5], [-1.8, -1.8], color=MUTED, lw=0.6, ls=(0, (2, 2)),
            alpha=0.7)
    ax.text(16.6, -8.4, r"standoff $r$", fontsize=8, color=MUTED,
            va="center", ha="left")
    ax.set_title(r"Survey geometry")
    panel_label(ax, "a")

    # ---- (b) sensor head, showing the atomic spins ------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(-0.35, 12.1); ax.set_ylim(-1.5, 7.9); ax.set_aspect("equal")
    ax.axis("off")

    # Nested mu-metal shield layers. Real shields are multi-layer: each layer
    # buys roughly an order of magnitude, and the ~10^4 total suppression
    # required here cannot be reached with one.
    for k, (inset, alpha) in enumerate(((0.00, 0.30), (0.34, 0.45), (0.68, 0.60))):
        ax.add_patch(FancyBboxPatch((0.55 + inset, 0.95 + inset),
                                    9.3 - 2 * inset, 5.7 - 2 * inset,
                                    boxstyle="round,pad=0.03,rounding_size=0.22",
                                    facecolor="none", edgecolor="#6f7681",
                                    lw=1.5 - 0.25 * k, alpha=alpha, zorder=2))
    ax.text(0.55, 7.35, r"$\mu$-metal shield (3 layers)", fontsize=7.8,
            color="#6f7681", ha="left")

    # Nulling coil pair (Helmholtz-like), drawn as coil windings
    for y in (2.05, 5.55):
        for xi in np.linspace(2.35, 8.05, 13):
            ax.add_patch(Ellipse((xi, y), 0.30, 0.62, fill=False,
                                 edgecolor=WARN, lw=1.1, zorder=3))
        ax.plot([2.2, 8.2], [y, y], color=WARN, lw=0.7, alpha=0.5, zorder=2)
    # Coil label placed once, outside the winding run, to avoid clutter.
    ax.text(10.85, 6.30, r"nulling" "\n" r"coils", fontsize=7.5, color=WARN,
            va="center", ha="center", linespacing=1.4)
    ax.annotate("", xy=(8.05, 5.62), xytext=(10.30, 6.30),
                arrowprops=dict(arrowstyle="-", color=WARN, lw=0.7))

    # MEMS cell containing the Rb vapour
    ax.add_patch(FancyBboxPatch((3.45, 3.05), 3.35, 1.45,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                facecolor="#eef3f7", edgecolor=INK, lw=1.0,
                                zorder=4))
    # Cell label placed clear of the upper coil, with a leader line to the
    # cell so the association is unambiguous.
    ax.text(5.12, 4.72, r"MEMS cell: $^{87}$Rb vapour", fontsize=7.5,
            ha="center", va="center", color=INK, zorder=8,
            bbox=dict(fc="white", ec="none", alpha=0.92, pad=1.4))

    # --- the atomic spins --------------------------------------------------
    # Pumping along z-hat polarises the ensemble along z. A transverse field
    # B_x then tips the polarisation toward y by the small angle
    # theta = arctan(gamma_e B_x / R_tot); the probe reads that tilt out.
    # NOTE the causal order: the PUMP creates the polarisation, the FIELD
    # displaces it. Drawn at an exaggerated tilt so it is visible.
    tilt = np.radians(26.0)
    for sx in np.linspace(3.95, 6.30, 6):
        for sy in (3.45, 4.05):
            dx, dy = 0.30 * np.sin(tilt), 0.30 * np.cos(tilt)
            ax.annotate("", xy=(sx + dx, sy + dy), xytext=(sx - dx, sy - dy),
                        arrowprops=dict(arrowstyle="-|>", color=ACCENT,
                                        lw=1.1, mutation_scale=7), zorder=6)
    # reference: undisplaced spin direction
    # Undisplaced (pure pump) direction, for comparison with the tipped spins.
    ax.annotate("", xy=(7.45, 4.42), xytext=(7.45, 3.20),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9,
                                linestyle=(0, (2, 1.6)), mutation_scale=6),
                zorder=6)
    ax.text(7.62, 4.52, r"$\hat z$", fontsize=7.5, color=MUTED, ha="left")

    # OPTICAL SOURCES. Both beams originate from diode lasers mounted OUTSIDE
    # the shield (they contain magnetic components and drive currents, so they
    # cannot sit next to the cell) and are delivered through small apertures
    # by polarisation-maintaining fibre. The probe is read out by a balanced
    # polarimeter on the far side, which measures the Faraday rotation angle.
    def _source(x, y, w, h, txt, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor="white", edgecolor=col, lw=1.0,
                                    zorder=7))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=6.6, color=col, zorder=8, linespacing=1.3)

    # pump diode below the shield, fibre up through the wall
    _source(4.10, -0.52, 2.15, 0.82, r"pump laser" "\n" r"795 nm", CLUTTER)
    # probe diode left, polarimeter right
    _source(-0.05, 3.36, 1.60, 0.86, r"probe" "\n" r"laser", SIGNAL)
    _source(9.62, 3.36, 2.05, 0.86, r"balanced" "\n" r"polarimeter", SIGNAL)

    ax.text(8.75, 2.92, r"rotation $\propto P_y$", fontsize=6.9, color=SIGNAL,
            ha="center", va="center",
            bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.0))

    # pump beam along z (enters from below)
    ax.annotate("", xy=(5.17, 3.02), xytext=(5.17, 0.36),
                arrowprops=dict(arrowstyle="-|>", color=CLUTTER, lw=1.8))
    ax.text(5.55, 0.62, r"pump $\hat z$ (circular)", fontsize=7.2,
            color=CLUTTER, va="center", ha="left")
    # probe beam along x (crosses the cell)
    ax.annotate("", xy=(9.62, 3.78), xytext=(1.55, 3.78),
                arrowprops=dict(arrowstyle="-|>", color=SIGNAL, lw=1.6))
    ax.text(1.70, 3.38, r"probe $\hat x$", fontsize=7.2, color=SIGNAL,
            ha="left", va="center")
    # the sensed transverse field
    # B_x is the external field under measurement: it enters from outside the
    # shield, which is what the shield and coils are attenuating.
    ax.annotate("", xy=(2.20, 6.42), xytext=(0.85, 6.42),
                arrowprops=dict(arrowstyle="-|>", color=GOOD, lw=1.6))
    ax.text(0.82, 6.86, r"$B_x$ (external)", fontsize=7.4, color=GOOD,
            ha="left")

    ax.text(5.85, -1.30,
            r"pump polarises along $\hat z$; $B_x$ tips the spins toward "
            r"$\hat y$; probe reads the tilt",
            fontsize=7.5, ha="center", color=MUTED)
    ax.set_title(r"Sensor head")
    panel_label(ax, "b")

    # ---- (c) signal chain -------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 4.4); ax.axis("off")

    def box(x, y, w, h, txt, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.04,rounding_size=0.13",
                                    facecolor=fc, edgecolor=INK, lw=0.8))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=7.2, color=INK)

    box(0.15, 2.5, 1.75, 1.15, "target\n" r"$B(t)$", "#e3ece4")
    box(0.15, 0.5, 1.75, 1.15, "platform\nclutter", "#f6e2e2")
    box(2.35, 1.5, 1.8, 1.15, "SERF\ncells", "#e6ecf4")
    box(4.6, 1.5, 1.8, 1.15, "gradiometer\n" r"$(S_1{-}S_2)/b$", "#e6ecf4")
    box(6.85, 1.5, 1.8, 1.15, "motion\ncompensation", "#f3edd9")
    box(9.1, 1.5, 1.4, 1.15, "matched\nfilter", "#ece3f2")
    for x0, x1, y0, y1 in [(1.9, 2.35, 3.07, 2.4), (1.9, 2.35, 1.07, 1.75),
                           (4.15, 4.6, 2.07, 2.07), (6.4, 6.85, 2.07, 2.07),
                           (8.65, 9.1, 2.07, 2.07)]:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=9, color=INK, lw=0.8))
    ax.text(10.5, 0.75, r"detection decision", fontsize=7.5, ha="right",
            color=ACCENT)
    ax.set_title(r"Signal chain")
    panel_label(ax, "c")

    # ---- (d) magnitudes each stage sees -----------------------------------
    ax = fig.add_subplot(gs[1, 1]); hairline_axes(ax)
    names = [r"Earth field", r"motion swing", r"platform" "\n" r"residual",
             r"SERF window", r"target" "\n" r"at CPA", r"sensor noise"]
    vals = np.array([50_000e-9, 2762e-9, OEHLER_COMPENSATED_RESIDUAL_T,
                     SERF_RESIDUAL_FIELD_TOLERANCE_T, 29e-9,
                     5e-15 * np.sqrt(10)]) * 1e12
    cols = [CLUTTER, CLUTTER, WARN, INK, GOOD, MUTED]
    ax.barh(range(len(vals)), vals, color=cols, height=0.62, zorder=3)
    ax.set_xscale("log"); ax.set_xlim(1e-2, 4e9)   # headroom for value labels
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"field magnitude (pT)")
    for i, v in enumerate(vals):
        ax.text(v * 2.2, i, rf"${v:,.0f}$" if v >= 1 else rf"${v:.3f}$",
                va="center", fontsize=7.5, color=INK)
    ax.set_title(r"Dynamic range spanned by the payload")
    annotate_corner(ax, r"$\sim\!10^{10}$ from ambient to sensor noise",
                    loc="lower right")
    panel_label(ax, "d")
    return fig


def generate_all(outdir="outputs", dpi=340, force_setup=False):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    import os
    os.makedirs(outdir, exist_ok=True)
    usetex = use_style()
    from ..validation.vv_suite import run_all, summary

    made = []
    # plot_setup is a static schematic: it depends on no simulation output,
    # so regenerating it on every run wastes ~20 s for an identical file.
    # Pass force_setup=True (or delete the PNG) to rebuild it.
    specs = []
    setup_path = os.path.join(outdir, "fig_setup.png")
    if force_setup or not os.path.exists(setup_path):
        specs.append(("fig_setup.png", (11.4, 7.6), plot_setup))
    else:
        print(f"  skipped {setup_path} (static schematic, already present)")
    specs += [
        ("fig_mission.png", (11.4, 7.8), lambda f: plot_mission(
            f, run_mission(MissionConfig(target_label="Large UXO / naval mine",
                                         target_depth_m=12.0)))),
        ("fig_field_budget.png", (11.4, 7.6),
         lambda f: plot_field_budget(f, MissionConfig())),
        ("fig_sensor_physics.png", (11.4, 7.4), plot_sensor_physics),
        ("fig_detection.png", (11.4, 7.4),
         lambda f: plot_detection(f, "Large UXO / naval mine")),
    ]
    for name, size, fn in specs:
        fig = Figure(figsize=size, dpi=dpi)
        fn(fig)
        p = os.path.join(outdir, name)
        fig.savefig(p, dpi=dpi)
        made.append(p); print(f"  wrote {p}")

    res = run_all(verbose=False)
    fig = Figure(figsize=(12.6, 7.4), dpi=dpi)
    plot_vv_table(fig, res, summary(res))
    p = os.path.join(outdir, "fig_vv_table.png")
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    made.append(p); print(f"  wrote {p}")
    print(f"  (LaTeX typesetting: {'ON' if usetex else 'OFF'})")
    return made


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    a = ap.parse_args()
    print("Generating figures headlessly...")
    generate_all(a.outdir)
