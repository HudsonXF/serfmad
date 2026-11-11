"""
serfmad.mission
===============
End-to-end mission simulation: target + platform + sensor + processing +
detection, plus a GPU-parallel Monte Carlo sweep for performance statistics.

This is the module that ties the physics to an operational answer:
"at what range, against what target, in what sea state, can this sensor
declare a detection at a stated probability of false alarm?"
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .constants import (TARGETS, TargetClass, EARTH_FIELD_NOMINAL_T,
                        SERF_RESIDUAL_FIELD_TOLERANCE_T, GAMMA_E)
from .physics.dipole import dipole_field, gradient_tensor, moment_from_orientation
from .physics.platform import (AUVPlatform, SEA_STATES, NullingSystem,
                               field_budget, cape_error)
from .processing.compensation import reference_cancel, GradiometerModel
from .processing.detection import (dipole_transit_template, matched_filter,
                                   detect, cfar_threshold)
from .quantum.bloch import BlochConfig, integrate
from .backend import xp, to_device, to_host, should_use_gpu, GPU_AVAILABLE, sync


@dataclass
class MissionConfig:
    """Complete specification of a simulated MAD mission leg."""
    target_label: str = "Small UXO / ordnance cluster"
    target_moment_override: Optional[float] = None
    target_depth_m: float = 12.0
    target_offset_m: float = 0.0          # cross-track offset
    moment_theta: float = np.pi / 2       # moment orientation
    moment_phi: float = 0.0

    sea_state_label: str = "moderate (SS3-4)"
    auv_speed_mps: float = 2.0
    auv_altitude_m: float = 0.0           # AUV depth plane (target below)
    baseline_m: float = 0.5
    cmrr_db: float = 60.0

    sensor_noise_T_rtHz: float = 5e-15    # Griffith 2010 MEMS design point
    bandwidth_hz: float = 10.0
    passive_shield: float = 100.0
    active_null: float = 100.0

    duration_s: float = 60.0
    n_samples: int = 3000
    pfa: float = 1e-4
    seed: int = 0

    @property
    def target(self) -> TargetClass:
        return TARGETS[self.target_label]

    @property
    def moment_magnitude(self) -> float:
        return self.target_moment_override or self.target.moment_A_m2

    @property
    def sensor_noise_T(self) -> float:
        """Noise in the measurement band: noise density * sqrt(bandwidth)."""
        return self.sensor_noise_T_rtHz * np.sqrt(self.bandwidth_hz)


@dataclass
class MissionResult:
    t: np.ndarray
    B_target_1: np.ndarray
    B_target_2: np.ndarray
    gradient_measured: np.ndarray
    gradient_true: np.ndarray
    platform_interference: np.ndarray
    residual: np.ndarray
    detection: object
    field_budget: dict
    cape: dict
    snr_db: float
    serf_valid: bool
    signal_to_clutter: float
    detection_credible: bool
    invalid_reason: str
    cpa_time: float
    cpa_range_m: float
    peak_target_field_T: float
    config: MissionConfig


def run_mission(cfg: Optional[MissionConfig] = None) -> MissionResult:
    """
    Simulate one complete MAD transit and attempt detection.

    PROCESSING CHAIN (each stage is a real, separately-validated component):
      1. target dipole field sampled at two gradiometer cell positions
      2. platform attitude modulates the ambient field into body axes
      3. shield + active null bring residual into the SERF window
      4. CAPE gain penalty applied from the residual
      5. finite-CMRR gradiometer differencing
      6. sensor noise added at the literature MEMS noise floor
      7. reference-channel cancellation of known motion frequencies
      8. matched-filter detection against a CFAR threshold
    """
    cfg = cfg or MissionConfig()
    rng = np.random.default_rng(cfg.seed)
    t = np.linspace(0, cfg.duration_s, cfg.n_samples)

    sea = SEA_STATES[cfg.sea_state_label]
    nulling = NullingSystem(passive_factor=cfg.passive_shield,
                            active_factor=cfg.active_null)
    plat = AUVPlatform(speed_mps=cfg.auv_speed_mps,
                       depth_m=cfg.auv_altitude_m,
                       start_x_m=-0.5 * cfg.auv_speed_mps * cfg.duration_s,
                       baseline_m=cfg.baseline_m,
                       sea_state=sea, nulling=nulling)

    # --- 1. target field at both cells ------------------------------------
    m_vec = moment_from_orientation(cfg.moment_magnitude,
                                    cfg.moment_theta, cfg.moment_phi)
    target_pos = np.array([0.0, cfg.target_offset_m, -abs(cfg.target_depth_m)])
    sensors = plat.sensor_positions(t)                 # (T,2,3)
    r1 = sensors[:, 0, :] - target_pos
    r2 = sensors[:, 1, :] - target_pos
    B1 = dipole_field(r1, m_vec)
    B2 = dipole_field(r2, m_vec)

    ranges = np.linalg.norm(0.5 * (r1 + r2), axis=-1)
    i_cpa = int(np.argmin(ranges))

    # --- 2-3. ambient field and nulling -----------------------------------
    fb = field_budget(plat, t)
    B_res = fb["residual_static_T"]

    # --- 4. CAPE penalty ---------------------------------------------------
    bloch = BlochConfig()
    cape = cape_error(B_res, float(np.abs(B1[:, 0]).max()),
                      bloch.R_tot, GAMMA_E)
    gain = cape["gain_factor"]

    # --- 5. gradiometer with finite CMRR ----------------------------------
    grad_model = GradiometerModel(baseline_m=cfg.baseline_m, cmrr_db=cfg.cmrr_db)
    # sensed component: body-x transverse axis
    s1 = B1[:, 0] * gain
    s2 = B2[:, 0] * gain
    grad_true = (s1 - s2) / cfg.baseline_m

    B_body = plat.ambient_field_body(t)
    common = B_body[:, 0] - B_body[:, 0].mean()
    common_after_null = common / fb["suppression_at_freq"]

    grad_meas = grad_model.measure(s1, s2, common_mode=common_after_null)

    # --- 6. sensor noise ---------------------------------------------------
    noise_grad = (cfg.sensor_noise_T * np.sqrt(2)) / cfg.baseline_m
    grad_meas = grad_meas + rng.normal(0, noise_grad, len(t))

    # --- 7. reference cancellation ----------------------------------------
    comp = reference_cancel(grad_meas, t,
                            freqs=[sea.roll_freq_hz, sea.pitch_freq_hz],
                            harmonics=2)
    residual = comp["compensated"]

    # --- 8. detection ------------------------------------------------------
    transit_sigma = max(cfg.target_depth_m / max(cfg.auv_speed_mps, 1e-6) / 3.0, 0.2)
    det = detect(residual, t, sigma_s=transit_sigma, pfa=cfg.pfa)

    noise_est = np.std(residual[:len(t) // 4])
    snr = 20 * np.log10(np.abs(grad_true).max() / noise_est) if noise_est > 0 else np.inf

    # --- VALIDITY GATES ---------------------------------------------------
    # A detection is only meaningful if the sensor was actually in the SERF
    # regime and the target actually dominated the residual clutter. Without
    # these gates the matched filter will happily "detect" uncancelled
    # platform motion, which is a false alarm reported as a success.
    reasons = []
    serf_valid = bool(fb["serf_viable"])
    if not serf_valid:
        reasons.append(
            f"SERF lock lost: residual {fb['residual_static_T']*1e9:.1f} nT "
            f"exceeds the {SERF_RESIDUAL_FIELD_TOLERANCE_T*1e9:.0f} nT window "
            f"(CAPE gain {gain:.3f})")
    # Signal-to-clutter: target peak against the TARGET-FREE noise floor.
    # Estimating the floor from a "quiet" window of the same record fails for
    # broad signatures, whose tails fill the window and inflate the floor --
    # that made S/C saturate near a constant regardless of dipole moment, so
    # a submarine and a micro-UUV appeared to have identical range limits.
    # The clutter+noise component is known exactly here (measured minus true),
    # so pass it through the same compensation and use its residual.
    clutter_only = grad_meas - grad_true
    floor = float(np.std(reference_cancel(
        clutter_only, t, freqs=[sea.roll_freq_hz, sea.pitch_freq_hz],
        harmonics=2)["compensated"]))
    s2c = float(np.abs(grad_true).max() / floor) if floor > 0 else np.inf
    if s2c < 3.0:
        reasons.append(f"target only {s2c:.2f}x the residual clutter floor: "
                       f"any peak is clutter, not signal")
    # a genuine transit peaks at the closest point of approach
    if abs(det.peak_time - float(t[i_cpa])) > 3.0:
        reasons.append(f"peak at {det.peak_time:.1f} s is far from CPA "
                       f"{t[i_cpa]:.1f} s")
    credible = det.detected and serf_valid and s2c >= 1.0 and not reasons

    return MissionResult(
        t=t, B_target_1=B1, B_target_2=B2,
        gradient_measured=grad_meas, gradient_true=grad_true,
        platform_interference=common, residual=residual,
        detection=det, field_budget=fb, cape=cape, snr_db=float(snr),
        serf_valid=serf_valid, signal_to_clutter=s2c,
        detection_credible=bool(credible),
        invalid_reason="; ".join(reasons),
        cpa_time=float(t[i_cpa]), cpa_range_m=float(ranges[i_cpa]),
        peak_target_field_T=float(np.abs(B1).max()), config=cfg,
    )


# ---------------------------------------------------------------------------
# GPU-parallel Monte Carlo
# ---------------------------------------------------------------------------

def monte_carlo_detection(
    n_trials: int = 8192,
    target_label: str = "Small UXO / ordnance cluster",
    range_min_m: float = 3.0,
    range_max_m: float = 40.0,
    sensor_noise_T: float = 5e-15 * np.sqrt(10.0),
    baseline_m: float = 0.5,
    pfa: float = 1e-4,
    seed: int = 0,
    n_time: int = 512,
    speed_mps: float = 2.0,
    clutter_residual_T: float = 2.23e-9,
    cmrr_db: float = 60.0,
    include_clutter: bool = True,
) -> dict:
    """
    (see full docstring below)

    CLUTTER IS THE REAL LIMITER -- READ THIS
    ----------------------------------------
    A 5 fT/rtHz SERF sensor has a noise floor roughly SIX ORDERS OF MAGNITUDE
    below the post-compensation platform interference residual measured on a
    real UUV (2.23 nT RMS, [Oehler24]). A Monte Carlo that models only sensor
    noise therefore reports absurdly optimistic detection ranges -- every
    target appears detectable, which is exactly the failure mode an earlier
    revision of this function exhibited (Pd = 1.0 at every range for every
    target class).

    The dominant residual is modelled here as a 1/f-weighted stochastic
    process at the [Oehler24] level, common-mode-rejected by the gradiometer
    at a FINITE CMRR. Setting include_clutter=False reproduces the
    (unrealistic) sensor-noise-only case and is retained solely so the report
    can quantify how large that modelling error is.
    """
    """
    GPU-parallel Monte Carlo over target range, orientation, and noise.

    WHY THIS IS THE GPU KERNEL
    --------------------------
    Each trial is an INDEPENDENT transit with its own random range, moment
    orientation, and noise realisation, all sharing one time grid. That is a
    pure SIMD workload: N_trials x N_time elementwise operations with no
    inter-trial communication. This is the part of the pipeline where a GPU
    genuinely pays -- not the single-trajectory Bloch solve.

    Returns detection probability vs range, which is the operationally
    meaningful output (a "detection range" curve at fixed Pfa).
    """
    rng = np.random.default_rng(seed)
    tgt = TARGETS[target_label]

    # --- sample trial parameters on host, then push to device -------------
    ranges = rng.uniform(range_min_m, range_max_m, n_trials)
    theta = rng.uniform(0, np.pi, n_trials)
    phi = rng.uniform(0, 2 * np.pi, n_trials)

    t = np.linspace(-15.0, 15.0, n_time)
    dt = t[1] - t[0]

    # Analytic transit signature: sensor moves along x at `speed`, target at
    # depth `range`. Along-track coordinate x = speed*t. The gradient of the
    # dipole field along track produces the classic bipolar MAD signature.
    # We compute this in closed form (vectorised) rather than integrating,
    # because the Bloch sensor response is linear in this regime (verified
    # in V&V) and the linear transfer function can be applied afterwards.
    R = to_device(ranges)[:, None]          # (N,1)
    T = to_device(t)[None, :]               # (1,M)
    TH = to_device(theta)[:, None]
    PH = to_device(phi)[:, None]

    x = speed_mps * T                        # along-track position (N,M)
    r2 = x ** 2 + R ** 2
    r = xp.sqrt(r2)

    mu0_4pi = 1e-7
    m = tgt.moment_A_m2
    # moment components
    mx = m * xp.sin(TH) * xp.cos(PH)
    my = m * xp.sin(TH) * xp.sin(PH)
    mz = m * xp.cos(TH)
    # unit vector from target to sensor: (x, 0, R)/r
    nx = x / r
    nz = R / r
    m_dot_n = mx * nx + mz * nz
    # sensed (x) component of dipole field
    Bx = mu0_4pi * (3.0 * nx * m_dot_n - mx) / r ** 3

    # gradiometer: finite-difference along track over the baseline
    shift = max(int(round(baseline_m / (speed_mps * dt))), 1)
    Bx_shift = xp.roll(Bx, -shift, axis=1)
    grad = (Bx_shift - Bx) / baseline_m
    grad[:, -shift:] = 0.0

    # --- noise model: sensor noise + (dominant) residual clutter ----------
    noise_grad = sensor_noise_T * np.sqrt(2) / baseline_m

    def _make_noise(nrows):
        """White sensor noise plus 1/f-weighted residual platform clutter."""
        w = to_device(rng.normal(0, noise_grad, (nrows, n_time)))
        if not include_clutter:
            return w
        # 1/f clutter shaped in the frequency domain, scaled so the
        # common-mode residual matches the published post-compensation level,
        # then reduced by the gradiometer's finite CMRR.
        nf = n_time // 2 + 1
        f = xp.arange(nf, dtype=xp.float64)
        f[0] = 1.0
        shape = 1.0 / xp.sqrt(f)                       # 1/f power -> 1/sqrt(f) amp
        ph = to_device(rng.uniform(0, 2 * np.pi, (nrows, nf)))
        spec = shape[None, :] * (xp.cos(ph) + 1j * xp.sin(ph))
        c = xp.fft.irfft(spec, n=n_time, axis=1)
        c = c / xp.std(c, axis=1, keepdims=True)
        cmrr_lin = 10 ** (cmrr_db / 20.0)
        clutter_grad = (clutter_residual_T / cmrr_lin) / baseline_m
        return w + c * clutter_grad

    obs = grad + _make_noise(n_trials)

    # unit-energy bipolar template
    u = (T - 0.0) / 3.0
    tmpl = -u * xp.exp(-0.5 * u ** 2)
    tmpl = tmpl - tmpl.mean()
    tmpl = tmpl / xp.sqrt((tmpl ** 2).sum())

    # correlation via FFT (batched) -- the genuinely parallel step
    obs_c = obs - obs.mean(axis=1, keepdims=True)
    F_obs = xp.fft.rfft(obs_c, axis=1)
    F_t = xp.fft.rfft(xp.broadcast_to(tmpl, obs.shape), axis=1)
    corr = xp.fft.irfft(F_obs * xp.conj(F_t), n=n_time, axis=1)
    stat = xp.abs(corr).max(axis=1)

    # null distribution for the CFAR threshold -- MUST use the identical
    # noise model, or the threshold is calibrated against the wrong statistics
    noise_only = _make_noise(n_trials)
    no_c = noise_only - noise_only.mean(axis=1, keepdims=True)
    F_n = xp.fft.rfft(no_c, axis=1)
    corr_n = xp.fft.irfft(F_n * xp.conj(F_t), n=n_time, axis=1)
    null_stat = xp.abs(corr_n).max(axis=1)

    sync()
    stat_h = to_host(stat)
    null_h = to_host(null_stat)
    thr = float(np.quantile(null_h, 1.0 - pfa)) if n_trials * pfa >= 1 \
        else float(null_h.max())
    detected = stat_h > thr

    # detection probability vs range
    nb = 20
    edges = np.linspace(range_min_m, range_max_m, nb + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    pd = np.full(nb, np.nan)
    for i in range(nb):
        sel = (ranges >= edges[i]) & (ranges < edges[i + 1])
        if sel.sum() > 5:
            pd[i] = detected[sel].mean()

    # R50: range at which Pd crosses 0.5
    r50 = np.nan
    valid = ~np.isnan(pd)
    if valid.sum() > 2:
        cv, pv = centres[valid], pd[valid]
        below = np.where(pv < 0.5)[0]
        if len(below) > 0 and below[0] > 0:
            i = below[0]
            x0, x1 = cv[i - 1], cv[i]
            y0, y1 = pv[i - 1], pv[i]
            r50 = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x0

    return {
        "ranges": ranges, "detected": detected, "statistic": stat_h,
        "null_statistic": null_h, "threshold": thr,
        "range_centres": centres, "pd_vs_range": pd, "r50_m": r50,
        "pfa": pfa, "n_trials": n_trials, "target": target_label,
        "used_gpu": should_use_gpu(n_trials), "gpu_available": GPU_AVAILABLE,
        "noise_grad_T_per_m": noise_grad,
    }
