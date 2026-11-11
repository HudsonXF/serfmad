"""
serfmad.quantum.bloch
=====================
GPU-parallel semiclassical Bloch solver for the SERF spin ensemble.

ROLE IN THE ARCHITECTURE
------------------------
This is the PRODUCTION solver. It integrates the SERF Bloch equation for
N independent spin ensembles simultaneously (Monte Carlo over target moment,
range, orientation, sea state, sensor imperfections), which is what makes
the GPU worthwhile. It is validated against the QuTiP Lindblad reference in
`serfmad.validation.vv_suite`.

GOVERNING EQUATION
------------------
The standard SERF Bloch equation for the electron spin polarisation vector
P = 2<S> (Seltzer 2008 thesis; reproduced in Borna et al. 2021 Eq. 1):

    dP/dt = (1/q) [ gamma_e (P x B) + R_op (s_hat - P) - R_rel P ]

where
    q       nuclear slowing-down factor (4-6 for I=3/2 alkali)
    gamma_e bare electron gyromagnetic ratio
    s_hat   optical pumping direction (unit vector, |s|=1 for circular pol.)
    R_op    optical pumping rate
    R_rel   total relaxation rate (spin destruction + wall + probe + ...)

CONVENTION NOTE (important, and a known source of confusion in the field):
in THIS form the factor 1/q multiplies the ENTIRE right-hand side, so q sets
the response BANDWIDTH but cancels from the DC steady state. Some papers
instead write gamma_e/q only on the precession term, in which case q survives
in the DC linewidth. We adopt the Seltzer/Borna convention above and VERIFY
the resulting linewidth numerically in the V&V suite rather than asserting a
formula. See docs/PHYSICS_NOTES.md.

WHY SEMICLASSICAL IS LEGITIMATE HERE
------------------------------------
A SERF vapour cell contains ~10^13-10^14 atoms. The ensemble-averaged spin
obeys the Bloch equation to excellent approximation; quantum correlations
between atoms are negligible for field sensing at these densities (they
matter only for spin-squeezing/entanglement-enhanced schemes, which are out
of scope here and flagged as future work). The Lindblad model is retained as
the single-spin reference precisely to demonstrate this equivalence rather
than assume it.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..backend import (xp, to_host, to_device, sync, GPU_AVAILABLE,
                       should_use_gpu)
from ..constants import GAMMA_E


@dataclass
class BlochConfig:
    """
    Physical configuration of the SERF spin ensemble.

    RELAXATION ANISOTROPY
    ---------------------
    The textbook SERF Bloch equation uses a single isotropic relaxation rate,
    which implicitly sets R_1 = R_2 = R_op + R_rel. That is an approximation:
    optical pumping is an inherently LONGITUDINAL process, so a full
    open-quantum-system treatment gives different longitudinal and transverse
    rates. Diagonalising the Lindblad superoperator for the collapse set used
    in `serfmad.quantum.lindblad` gives exactly

        R_1 = R_op + Gamma_sd                       (longitudinal)
        R_2 = (R_op + Gamma_sd)/2 + Gamma_sd        (transverse)

    (verified numerically: eigenvalues -505 and -257.5 for R_op=500,
    Gamma_sd=5). This class therefore exposes R_1 and R_2 independently.

    `isotropic=True` (default) reproduces the standard SERF literature form
    and is what the production pipeline uses, so results remain comparable
    with published SERF modelling. `isotropic=False` uses the exact
    Lindblad-derived rates and is what the cross-model V&V test exercises.
    Keeping BOTH, and being explicit about which is in use, is preferable to
    silently picking one and calling it "the" model.
    """
    R_op: float = 500.0        # optical pumping rate (s^-1)
    R_rel: float = 5.0         # spin-destruction / relaxation rate (s^-1)
    q: float = 5.0             # nuclear slowing-down factor (I=3/2, mid-range)
    gamma_e: float = GAMMA_E   # rad/(s*T)
    pump_axis: tuple = (0.0, 0.0, 1.0)   # s_hat, circular polarisation along z
    isotropic: bool = True     # True => textbook SERF form (R_1 = R_2)

    @property
    def R_tot(self) -> float:
        return self.R_op + self.R_rel

    @property
    def R_1(self) -> float:
        """Longitudinal relaxation rate (s^-1)."""
        return self.R_op + self.R_rel

    @property
    def R_2(self) -> float:
        """Transverse relaxation rate (s^-1)."""
        if self.isotropic:
            return self.R_op + self.R_rel
        return 0.5 * (self.R_op + self.R_rel) + self.R_rel

    @property
    def P_equilibrium(self) -> float:
        """Steady-state polarisation with zero field: P0 = R_op/R_1."""
        return self.R_op / self.R_1

    @property
    def bandwidth_hz(self) -> float:
        """
        Sensor 3 dB bandwidth. The 1/q prefactor slows the response, so the
        characteristic rate is R_tot/q  =>  f_3dB = R_tot/(2*pi*q).
        """
        return self.R_tot / (2.0 * np.pi * self.q)

    @property
    def B_hwhm(self) -> float:
        """
        Magnetic resonance half-width in field units.
        Setting the field term equal to the relaxation term in the exact
        steady state gives gamma_e^2*B^2/R_2 = R_1, hence
            B_HWHM = sqrt(R_1*R_2)/gamma_e
        which reduces to R_tot/gamma_e in the isotropic case.
        (q cancels in this convention -- verified numerically in V&V.)
        """
        return np.sqrt(self.R_1 * self.R_2) / self.gamma_e

    @property
    def dc_responsivity(self) -> float:
        """
        Small-signal DC responsivity dP_y/dB_x (units 1/T).
        Linearising the steady state: P_y = gamma_e * P0 * B_x / R_2.
        """
        return self.gamma_e * self.P_equilibrium / self.R_2


def _mod(a):
    """Array module that owns `a` -- NumPy for host arrays, xp otherwise."""
    return np if isinstance(a, np.ndarray) else xp


def _as(module, a):
    """Coerce `a` to `module` without copying when it already belongs to it."""
    if module is np:
        return a if isinstance(a, np.ndarray) else to_host(a)
    return to_device(a) if isinstance(a, np.ndarray) else a


def bloch_rhs(P, B, cfg: BlochConfig):
    """
    Right-hand side of the SERF Bloch equation, vectorised over the ensemble.

    Parameters
    ----------
    P : (N,3) array -- spin polarisation vectors
    B : (N,3) array -- local magnetic field at each ensemble member (T)

    Returns (N,3) dP/dt.

    Implemented with explicit component arithmetic rather than xp.cross()
    because the explicit form avoids a temporary allocation per call inside
    the RK4 inner loop, which measurably matters on GPU.
    """
    Px, Py, Pz = P[:, 0], P[:, 1], P[:, 2]
    Bx, By, Bz = B[:, 0], B[:, 1], B[:, 2]

    # P x B
    cx = Py * Bz - Pz * By
    cy = Pz * Bx - Px * Bz
    cz = Px * By - Py * Bx

    sx, sy, sz = cfg.pump_axis
    inv_q = 1.0 / cfg.q
    g = cfg.gamma_e
    Rop = cfg.R_op
    R1, R2 = cfg.R_1, cfg.R_2

    # Pump drives the polarisation toward s_hat at rate R_op; relaxation acts
    # at R_2 transverse to the pump axis and R_1 along it. With the default
    # pump axis (0,0,1) this reduces to the familiar component form.
    out = _mod(P).empty_like(P)
    out[:, 0] = inv_q * (g * cx + Rop * sx - R2 * Px)
    out[:, 1] = inv_q * (g * cy + Rop * sy - R2 * Py)
    out[:, 2] = inv_q * (g * cz + Rop * sz - R1 * Pz)
    return out


def integrate(
    t: np.ndarray,
    B_field_fn: Callable[[float], "xp.ndarray"],
    cfg: Optional[BlochConfig] = None,
    n_ensemble: int = 1,
    P0=None,
    record_every: int = 1,
):
    """
    RK4-integrate the Bloch equation for an ensemble over a time grid.

    Parameters
    ----------
    t : (T,) uniform time grid (s)
    B_field_fn : callable t -> (N,3) device array of local fields (T)
    cfg : BlochConfig
    n_ensemble : N, number of parallel ensemble members
    P0 : optional (N,3) initial polarisation; defaults to fully pumped along s_hat
    record_every : store every k-th sample (memory control for large N*T)

    Returns dict with 't' and 'P' of shape (T_rec, N, 3), on the HOST.

    NUMERICAL METHOD JUSTIFICATION
    ------------------------------
    Fixed-step RK4 is chosen over an adaptive solver because (a) every
    ensemble member must share a time grid for vectorisation -- adaptive
    per-member stepping would serialise the batch, and (b) the stiffness here
    is bounded and known a priori (fastest rate = R_tot/q), so the stability
    condition dt < ~2.8/(R_tot/q) can be checked up front rather than
    discovered adaptively. `check_stability()` enforces this.
    """
    cfg = cfg or BlochConfig()
    t = np.asarray(t, dtype=float)
    dt = float(t[1] - t[0])
    n_steps = len(t)

    # A single trajectory -- which is what every caller of this function
    # actually passes -- is not a GPU workload. Each RK4 stage would launch
    # ~10 CUDA kernels over 3 floats, where launch latency dwarfs the
    # arithmetic entirely; profiling put bloch_rhs at 15.8 s of a 21 s stress
    # run for exactly this reason. Selecting the array module by ensemble size
    # honours the contract should_use_gpu() already documents. NumPy and CuPy
    # are bit-identical on this path (every Python operator is a separate
    # kernel, so no fused-multiply-add contraction can diverge), which is
    # verified against stored digests rather than assumed.
    am = xp if should_use_gpu(n_ensemble) else np

    if P0 is None:
        P = am.zeros((n_ensemble, 3), dtype=am.float64)
        P[:, 0] = cfg.pump_axis[0] * cfg.P_equilibrium
        P[:, 1] = cfg.pump_axis[1] * cfg.P_equilibrium
        P[:, 2] = cfg.pump_axis[2] * cfg.P_equilibrium
    else:
        P = _as(am, np.asarray(P0, dtype=float).reshape(n_ensemble, 3)).copy()

    n_rec = (n_steps + record_every - 1) // record_every
    out = am.zeros((n_rec, n_ensemble, 3), dtype=am.float64)
    t_rec = np.zeros(n_rec)

    rec = 0
    for i in range(n_steps):
        if i % record_every == 0:
            out[rec] = P
            t_rec[rec] = t[i]
            rec += 1
        ti = t[i]
        # Classical RK4. B is evaluated at the stage times.
        k1 = bloch_rhs(P, _as(am, B_field_fn(ti)), cfg)
        k2 = bloch_rhs(P + 0.5 * dt * k1, _as(am, B_field_fn(ti + 0.5 * dt)), cfg)
        k3 = bloch_rhs(P + 0.5 * dt * k2, _as(am, B_field_fn(ti + 0.5 * dt)), cfg)
        k4 = bloch_rhs(P + dt * k3, _as(am, B_field_fn(ti + dt)), cfg)
        P = P + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    sync()
    return {"t": t_rec[:rec], "P": to_host(out[:rec]), "config": cfg}


def steady_state_analytic(Bx, cfg: Optional[BlochConfig] = None):
    """
    Closed-form steady state for a static transverse field B_x (pump along z).

    Solving 0 = gamma_e(P x B) + R_op*s - R_tot*P with B = (Bx,0,0), s = z:

        P_y = gamma_e*Bx*R_op*R_tot / (R_tot*(R_tot^2 + gamma_e^2*Bx^2))
        P_z = R_op*R_tot / (R_tot^2 + gamma_e^2*Bx^2)  ... (see derivation below)

    Derivation: with B along x, the y and z components couple:
        0 = gamma_e*(Pz*Bx) - R_tot*Py            (y-component, P x B -> Pz*Bx)
        0 = -gamma_e*(Py*Bx) + R_op - R_tot*Pz    (z-component)
    =>  Py = gamma_e*Bx*Pz/R_tot
    =>  Pz = R_op / (R_tot + gamma_e^2*Bx^2/R_tot)
    Both are exact (no small-field approximation).

    This analytic result is a second, independent check on the numerical
    integrator AND on the QuTiP model -- three-way agreement is required by
    the V&V suite.
    """
    cfg = cfg or BlochConfig()
    Bx = np.asarray(Bx, dtype=float)
    g, Rop = cfg.gamma_e, cfg.R_op
    R1, R2 = cfg.R_1, cfg.R_2
    # With B = (Bx,0,0) and pump along z, the coupled equations are
    #   0 = g*Pz*Bx - R2*Py        (y)
    #   0 = -g*Py*Bx + Rop - R1*Pz (z)
    # =>  Py = g*Bx*Pz/R2 ;  Pz = Rop / (R1 + g^2*Bx^2/R2)
    Pz = Rop / (R1 + (g ** 2) * (Bx ** 2) / R2)
    Py = g * Bx * Pz / R2
    Px = np.zeros_like(Bx)
    return {"Px": Px, "Py": Py, "Pz": Pz, "Bx": Bx}


def check_stability(dt: float, cfg: Optional[BlochConfig] = None,
                    B_max: float = 0.0) -> dict:
    """
    Verify the RK4 step size against the fastest timescale in the system.

    The two competing rates are the relaxation rate R_tot/q and the Larmor
    rate gamma_e*|B|/q. RK4 is stable for real eigenvalues up to
    |lambda|*dt < 2.785.
    """
    cfg = cfg or BlochConfig()
    lam_relax = max(cfg.R_1, cfg.R_2) / cfg.q
    lam_larmor = cfg.gamma_e * abs(B_max) / cfg.q
    lam = max(lam_relax, lam_larmor)
    return {
        "dt": dt,
        "fastest_rate_per_s": lam,
        "lambda_dt": lam * dt,
        "rk4_limit": 2.785,
        "stable": lam * dt < 2.785,
        "recommended_max_dt": 2.785 / lam if lam > 0 else np.inf,
        "limiting_process": "Larmor" if lam_larmor > lam_relax else "relaxation",
    }
