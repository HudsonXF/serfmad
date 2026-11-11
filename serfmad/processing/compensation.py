"""
serfmad.processing.compensation
===============================
Platform interference compensation.

Two complementary mechanisms are implemented, mirroring operational practice:

1. TOLLES-LAWSON-STYLE REGRESSION (`tolles_lawson_compensate`)
   The classical aeromagnetic compensation model. Platform interference is
   decomposed into permanent, induced, and eddy-current terms, each a known
   function of vehicle attitude direction cosines. A linear least-squares fit
   against measured attitude removes the modelled interference.
   [Oehler24] report exactly this class of method reducing raw UUV
   interference from 246.90 nT RMS to 2.23 nT RMS -- the benchmark our V&V
   suite tests against.

2. REFERENCE-CHANNEL / SYNCHRONOUS CANCELLATION (`reference_cancel`)
   Where the disturbance is narrowband and its frequency is known from an
   IMU (e.g. wave-driven roll at 0.25 Hz), a least-squares fit of that
   reference (plus harmonics) is subtracted directly. Cheaper than full
   Tolles-Lawson and effective for the dominant periodic motion term.

3. GRADIOMETRY (`gradiometer_difference`)
   Common-mode rejection by differencing two cells on a short baseline.
   IMPORTANT HONESTY NOTE: gradiometry cancels only the SPATIALLY UNIFORM
   part of the interference. The platform's OWN field has a strong gradient
   across a 0.5 m baseline (the sources are metres away, not hundreds), so
   gradiometry does NOT remove vehicle self-noise -- a common and serious
   modelling error. We model a finite common-mode rejection ratio (CMRR)
   rather than assuming perfect cancellation.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


def tolles_lawson_basis(roll, pitch, yaw, B_earth_body=None, include_eddy=True):
    """
    Build the Tolles-Lawson design matrix from vehicle attitude.

    The classical model expresses platform interference as
        B_int = sum_i c_i * f_i(direction cosines)
    with 16 terms (3 permanent + 6 induced + 9 eddy, reduced by symmetry
    to the standard 16-term or 18-term forms).

    Here we use direction cosines of the Earth field in the body frame,
    which is the standard formulation.
    """
    roll = np.asarray(roll); pitch = np.asarray(pitch); yaw = np.asarray(yaw)
    from ..physics.platform import rotation_matrix
    from ..constants import EARTH_FIELD_NOMINAL_T

    if B_earth_body is None:
        inc = np.radians(70.0)
        Bw = EARTH_FIELD_NOMINAL_T * np.array([np.cos(inc), 0.0, np.sin(inc)])
        R = rotation_matrix(roll, pitch, yaw)
        B_earth_body = np.einsum('...ij,j->...i', R, Bw)

    B = np.asarray(B_earth_body)
    Bt = np.linalg.norm(B, axis=-1, keepdims=True)
    Bt = np.maximum(Bt, 1e-18)
    u = B / Bt                    # direction cosines
    u1, u2, u3 = u[..., 0], u[..., 1], u[..., 2]

    cols = [u1, u2, u3]                                    # permanent (3)
    cols += [u1 * u1, u1 * u2, u1 * u3,
             u2 * u2, u2 * u3, u3 * u3]                    # induced (6)

    if include_eddy:
        du1 = np.gradient(u1)
        du2 = np.gradient(u2)
        du3 = np.gradient(u3)
        cols += [u1 * du1, u1 * du2, u1 * du3,
                 u2 * du1, u2 * du2, u2 * du3,
                 u3 * du1, u3 * du2, u3 * du3]             # eddy (9)

    cols.append(np.ones_like(u1))                          # DC offset
    return np.stack(cols, axis=-1)


def tolles_lawson_compensate(signal, roll, pitch, yaw, ridge=1e-12):
    """
    Fit and remove platform interference using the Tolles-Lawson basis.

    `ridge` applies Tikhonov regularisation: the design matrix is often
    ill-conditioned when the vehicle does not exercise all attitudes (a real
    operational problem -- compensation manoeuvres exist precisely to
    condition this fit). Ridge regularisation keeps the solution bounded
    instead of blowing up on near-collinear columns.

    Returns dict with compensated signal, fitted interference, and metrics.
    """
    y = np.asarray(signal, dtype=float)
    A = tolles_lawson_basis(roll, pitch, yaw)
    A = A.reshape(len(y), -1)

    # Ridge-regularised normal equations
    AtA = A.T @ A
    n = AtA.shape[0]
    coeffs = np.linalg.solve(AtA + ridge * np.trace(AtA) / n * np.eye(n), A.T @ y)
    fitted = A @ coeffs
    resid = y - fitted

    return {
        "compensated": resid,
        "interference_estimate": fitted,
        "coefficients": coeffs,
        "rms_before": float(np.std(y)),
        "rms_after": float(np.std(resid)),
        "improvement_ratio": float(np.std(y) / np.std(resid)) if np.std(resid) > 0 else np.inf,
        "condition_number": float(np.linalg.cond(A)),
    }


def reference_cancel(signal, t, freqs, harmonics=2):
    """
    Least-squares cancellation of known narrowband disturbances.

    Builds a sin/cos basis at each supplied frequency and its harmonics,
    fits, and subtracts. Equivalent to an ideal adaptive notch with perfect
    frequency knowledge -- appropriate when an IMU supplies the wave
    encounter frequency directly.

    Preferred over an IIR notch for short records: with only a few cycles in
    the window, an IIR notch rings badly at the record edges, whereas the
    least-squares fit does not.
    """
    y = np.asarray(signal, dtype=float)
    t = np.asarray(t, dtype=float)
    cols = [np.ones_like(t)]
    for f in np.atleast_1d(freqs):
        for h in range(1, harmonics + 1):
            w = 2 * np.pi * f * h
            cols += [np.sin(w * t), np.cos(w * t)]
    A = np.stack(cols, axis=-1)
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    fitted = A @ coeffs
    resid = y - fitted
    return {
        "compensated": resid,
        "interference_estimate": fitted,
        "rms_before": float(np.std(y)),
        "rms_after": float(np.std(resid)),
        "improvement_ratio": float(np.std(y) / np.std(resid)) if np.std(resid) > 0 else np.inf,
    }


@dataclass
class GradiometerModel:
    """
    Two-cell gradiometer with FINITE common-mode rejection.

    cmrr_db : common-mode rejection ratio. Perfect cancellation (infinite
        CMRR) is physically wrong for a platform whose own magnetic sources
        sit within metres of the baseline. Real gradiometers are limited by
        (a) sensor gain/axis mismatch and (b) the genuine gradient of the
        interferer across the baseline. 60 dB (1000x) is a demanding but
        achievable engineering target.
    """
    baseline_m: float = 0.5
    cmrr_db: float = 60.0

    @property
    def cmrr_linear(self) -> float:
        return 10 ** (self.cmrr_db / 20.0)

    def measure(self, B1, B2, common_mode=None, rng=None):
        """
        Form the gradient estimate, including leakage of common-mode
        interference at the finite CMRR.
        """
        B1 = np.asarray(B1, dtype=float)
        B2 = np.asarray(B2, dtype=float)
        grad = (B1 - B2) / self.baseline_m
        if common_mode is not None:
            leak = np.asarray(common_mode, dtype=float) / self.cmrr_linear
            grad = grad + leak / self.baseline_m
        return grad
