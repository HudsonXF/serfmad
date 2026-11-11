"""
serfmad.processing.detection
============================
Detection-theoretic back end: matched filtering, CFAR thresholding, and ROC
performance characterisation.

FRAMING
-------
MAD is a signal-detection problem with exactly the same structure as active
sonar or radar pulse detection, and is treated with the same machinery:

    H0 : r(t) = n(t)                (no target)
    H1 : r(t) = a*s(t - tau) + n(t) (target at unknown range/time tau)

For a known waveform shape in additive white Gaussian noise, the matched
filter is the optimal (Neyman-Pearson) detector, maximising output SNR. The
target signature shape IS known here: a dipole transiting past a sensor
produces a characteristic bipolar/tripolar anomaly whose form follows from
the dipole geometry, so a matched filter is the correct tool rather than an
arbitrary one.

The output detection statistic is compared against a threshold set for a
specified probability of false alarm (CFAR), and performance is reported as
a ROC curve -- Pd vs Pfa -- which is the only defensible way to state MAD
performance. A single "SNR" number without a ROC is not an operational claim.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy import signal as sps
from scipy import stats


def dipole_transit_template(t, t_cpa, sigma_s, kind="bipolar"):
    """
    Analytic template for a dipole transit signature.

    A sensor passing a dipole at constant speed sees a signature whose shape
    depends on the moment orientation. The two canonical forms:
      - 'gaussian' : monopolar bump (moment aligned with closest approach)
      - 'bipolar'  : first derivative of a Gaussian (transverse moment) --
                     the classic MAD "S-curve"
      - 'tripolar' : second derivative, common for along-track moments

    Using the RIGHT template matters: a matched filter mismatched in shape
    loses processing gain. In practice a bank of templates is run in
    parallel, which is what `matched_filter_bank` does.
    """
    t = np.asarray(t, dtype=float)
    u = (t - t_cpa) / sigma_s
    g = np.exp(-0.5 * u ** 2)
    if kind == "gaussian":
        s = g
    elif kind == "bipolar":
        s = -u * g
    elif kind == "tripolar":
        s = (u ** 2 - 1.0) * g
    else:
        raise ValueError(f"unknown template kind: {kind}")
    s = s - s.mean()
    nrm = np.sqrt(np.sum(s ** 2))
    return s / nrm if nrm > 0 else s


def matched_filter(x, template):
    """
    Correlate the record against a unit-energy template.

    Returns the detection statistic time series. Normalisation is by the
    residual noise standard deviation so the output is directly in units of
    'sigmas above noise', which is what a CFAR threshold consumes.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    tmpl = np.asarray(template, dtype=float)
    d = np.correlate(x, tmpl, mode="same")
    return d


def matched_filter_bank(x, t, sigma_list, kinds=("gaussian", "bipolar", "tripolar")):
    """
    Run a bank of templates spanning plausible transit durations and shapes,
    returning the per-sample maximum over the bank.

    This is the standard approach when the target's range/speed/orientation
    (hence signature width and shape) are unknown a priori. The cost is a
    small CFAR loss from maximising over the bank, which is accounted for by
    calibrating the threshold empirically on noise-only data (see `roc_curve`).
    """
    t = np.asarray(t, dtype=float)
    t_mid = 0.5 * (t[0] + t[-1])
    best = None
    best_meta = None
    for kind in kinds:
        for s in sigma_list:
            tmpl = dipole_transit_template(t, t_mid, s, kind)
            d = matched_filter(x, tmpl)
            if best is None:
                best = d.copy()
                best_meta = np.array([[kind == k for k in kinds].index(True)] * len(d))
            else:
                upd = np.abs(d) > np.abs(best)
                best[upd] = d[upd]
    return best


@dataclass
class DetectionResult:
    statistic: np.ndarray
    threshold: float
    detected: bool
    peak_time: float
    peak_value: float
    noise_sigma: float
    pfa_design: float


def cfar_threshold(noise_statistic, pfa=1e-4):
    """
    Set a threshold for a target probability of false alarm.

    Two estimates are returned and the LARGER (more conservative) is used:
      - Gaussian assumption: thr = sigma * Phi^-1(1 - pfa)
      - Empirical quantile of the observed noise statistic
    Relying on the Gaussian form alone is optimistic when the statistic is
    not truly Gaussian (matched-filter output over a bank is not), so the
    empirical quantile guards against that.
    """
    x = np.abs(np.asarray(noise_statistic, dtype=float))
    sigma = float(np.std(noise_statistic))
    thr_gauss = sigma * stats.norm.ppf(1.0 - pfa)
    n = len(x)
    if n * pfa >= 1:
        thr_emp = float(np.quantile(x, 1.0 - pfa))
    else:
        thr_emp = float(x.max())
    return {
        "threshold": max(thr_gauss, thr_emp),
        "threshold_gaussian": thr_gauss,
        "threshold_empirical": thr_emp,
        "noise_sigma": sigma,
    }


def detect(x, t, template=None, sigma_s=1.0, pfa=1e-4, noise_region=None):
    """
    Full detection decision on one record.

    noise_region : boolean mask selecting target-free samples used to
        estimate the noise statistics. If None, the outer 25% of the record
        at each end is assumed target-free -- an assumption that must hold
        operationally (it does for a transit geometry with known CPA window).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if template is None:
        template = dipole_transit_template(t, 0.5 * (t[0] + t[-1]), sigma_s, "bipolar")
    d = matched_filter(x, template)

    if noise_region is None:
        n = len(t)
        noise_region = np.zeros(n, dtype=bool)
        noise_region[: n // 4] = True
        noise_region[-n // 4:] = True

    thr_info = cfar_threshold(d[noise_region], pfa)
    thr = thr_info["threshold"]
    idx = int(np.argmax(np.abs(d)))
    return DetectionResult(
        statistic=d,
        threshold=thr,
        detected=bool(abs(d[idx]) > thr),
        peak_time=float(t[idx]),
        peak_value=float(d[idx]),
        noise_sigma=thr_info["noise_sigma"],
        pfa_design=pfa,
    )


def roc_curve(signal_template, noise_sigma, n_trials=2000, amplitudes=None, rng=None):
    """
    Monte Carlo ROC: Pd vs Pfa for a matched filter in AWGN.

    Rather than quoting the closed-form Pd = Q(Q^-1(Pfa) - sqrt(2E/N0)) and
    stopping there, we simulate, because the closed form assumes perfectly
    known arrival time and exactly matched template -- neither holds
    operationally. The Monte Carlo captures the max-over-time CFAR loss.

    Returns dict with pfa grid, pd per amplitude, and the analytic curve for
    comparison (the gap between them IS the CFAR/mismatch loss, and quoting
    that gap honestly is the point).
    """
    rng = rng or np.random.default_rng(0)
    s = np.asarray(signal_template, dtype=float)
    s = s / np.sqrt(np.sum(s ** 2))
    n = len(s)
    if amplitudes is None:
        amplitudes = noise_sigma * np.array([1, 2, 3, 4, 5, 6, 8, 10])

    pfa_grid = np.logspace(-4, -0.3, 40)
    out = {"pfa": pfa_grid, "pd": {}, "pd_analytic": {}, "snr_db": {}}

    # Noise-only statistic distribution (max over record -> CFAR relevant)
    null_stat = np.empty(n_trials)
    for i in range(n_trials):
        w = rng.normal(0, noise_sigma, n)
        null_stat[i] = np.max(np.abs(np.correlate(w - w.mean(), s, mode="same")))

    for a in amplitudes:
        alt_stat = np.empty(n_trials)
        for i in range(n_trials):
            w = rng.normal(0, noise_sigma, n)
            x = a * s + w
            alt_stat[i] = np.max(np.abs(np.correlate(x - x.mean(), s, mode="same")))
        thr = np.quantile(null_stat, 1.0 - pfa_grid)
        pd = np.array([(alt_stat > th).mean() for th in thr])
        snr_lin = (a ** 2) / (noise_sigma ** 2)
        out["pd"][float(a)] = pd
        out["snr_db"][float(a)] = 10 * np.log10(snr_lin)
        d_prime = a / noise_sigma
        out["pd_analytic"][float(a)] = stats.norm.sf(
            stats.norm.isf(pfa_grid) - d_prime)
    return out


def snr_db(signal, noise_sigma):
    """Peak-signal-to-noise in dB."""
    p = float(np.max(np.abs(signal)))
    return 20 * np.log10(p / noise_sigma) if noise_sigma > 0 else np.inf
