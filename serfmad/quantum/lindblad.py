"""
serfmad.quantum.lindblad
========================
QuTiP Lindblad master-equation model of the SERF sensor spin.

ROLE IN THE ARCHITECTURE
------------------------
This module is the REFERENCE TRUTH. It is a first-principles open-quantum-
system treatment of a single effective spin-1/2, solved with QuTiP's `mesolve`.
It is deliberately NOT the production solver: it is slow, single-trajectory,
and cannot be batched across a Monte Carlo ensemble.

Its job is to validate the fast semiclassical Bloch solver
(`serfmad.quantum.bloch`) that the GPU pipeline actually uses. If the two
disagree beyond tolerance, the Bloch solver is wrong and the V&V suite fails.
This is the standard "expensive reference vs cheap production model"
verification pattern.

PHYSICS
-------
    rho_dot = -i[H, rho] + sum_k D[L_k] rho

with H = -gamma_e * B . S  (Zeeman), S = sigma/2, and collapse operators

    L_pump = sqrt(R_op) * S_+          population transfer INTO |0> (pumped state)
    L_sd   = sqrt(Gamma_sd/2) * S_+    isotropic spin destruction
    L_sd   = sqrt(Gamma_sd/2) * S_-
    L_deph = sqrt(2*Gamma_sd) * S_z    dephasing

CRITICAL CORRECTNESS NOTES (each of these was a real bug found in V&V):
  1. The pump operator MUST be a population-transfer operator S_+ = |0><1|.
     Using the PROJECTOR 0.5*(I + 2*Sz) = |0><0| produces D[L]rho == 0 on
     all diagonal rho -- it dephases but pumps nothing, and the spin decays
     to the maximally mixed state regardless of R_op.
  2. Ladder operators MUST be built from the normalised S_x, S_y (i.e.
     0.5*sigma), NOT from the bare Pauli matrices. sigma_x + i*sigma_y is
     2x too large and inflates the associated rate by 4x.
  3. A field applied along the pump axis z cannot rotate <S_z> ([S_z,S_z]=0).
     The SENSED field must be transverse. Pump along z, apply B_x, read <S_y>.
  4. QuTiP >= 5 makes `e_ops` keyword-only in `mesolve`.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Callable

# QuTiP is imported LAZILY. Importing it pulls qutip.bloch -> Matplotlib,
# whose first-run font-cache build takes minutes on Windows. Only V-08 needs
# QuTiP, so the other 16 tests should not pay that cost.
QUTIP_AVAILABLE = None          # None = not yet probed
_QUTIP_ERR = ""
sigmax = sigmay = sigmaz = qeye = basis = mesolve = steadystate = Qobj = None
SX = SY = SZ = IDENT = SP = SM = None


def _ensure_qutip():
    """Import QuTiP and build the operator set on first actual use."""
    global QUTIP_AVAILABLE, _QUTIP_ERR
    global sigmax, sigmay, sigmaz, qeye, basis, mesolve, steadystate, Qobj
    global SX, SY, SZ, IDENT, SP, SM
    if QUTIP_AVAILABLE is not None:
        return QUTIP_AVAILABLE
    try:
        from qutip import (sigmax as _sx, sigmay as _sy, sigmaz as _sz,
                           qeye as _qe, basis as _b, mesolve as _m,
                           steadystate as _ss, Qobj as _Q)
        sigmax, sigmay, sigmaz, qeye = _sx, _sy, _sz, _qe
        basis, mesolve, steadystate, Qobj = _b, _m, _ss, _Q
        SX = 0.5 * sigmax(); SY = 0.5 * sigmay(); SZ = 0.5 * sigmaz()
        IDENT = qeye(2); SP = SX + 1j * SY; SM = SX - 1j * SY
        QUTIP_AVAILABLE = True
    except Exception as exc:
        QUTIP_AVAILABLE = False
        _QUTIP_ERR = f"{type(exc).__name__}: {exc}"
    return QUTIP_AVAILABLE

from ..constants import GAMMA_E





@dataclass
class LindbladConfig:
    """Configuration for a Lindblad run."""
    R_op: float = 500.0            # optical pumping rate (s^-1)
    Gamma_sd: float = 5.0          # spin-destruction rate (s^-1)
    gamma_e: float = GAMMA_E       # rad/(s*T)
    q: float = 1.0                 # slowing-down factor (1 => bare electron)

    @property
    def gamma_eff(self) -> float:
        """Effective gyromagnetic ratio gamma_e/q seen by the coupled spin."""
        return self.gamma_e / self.q

    @property
    def total_relaxation(self) -> float:
        return self.R_op + self.Gamma_sd


def collapse_operators(cfg: LindbladConfig) -> list:
    _ensure_qutip()
    """
    Build the Lindblad collapse operator set.

    NOTE the pump term: sqrt(R_op)*SP transfers population into the pumped
    state. See module docstring bug-note (1) for why a projector is wrong.
    """
    if not _ensure_qutip():
        raise RuntimeError(f"QuTiP unavailable: {_QUTIP_ERR}")
    return [
        np.sqrt(cfg.Gamma_sd / 2.0) * SP,
        np.sqrt(cfg.Gamma_sd / 2.0) * SM,
        np.sqrt(2.0 * cfg.Gamma_sd) * SZ,
        np.sqrt(cfg.R_op) * SP,
    ]


def steady_state_response(Bx: float, cfg: LindbladConfig | None = None) -> dict:
    _ensure_qutip()
    """
    Exact steady-state spin expectation values for a static transverse field Bx.

    This is the cleanest possible validation observable: it has a closed-form
    expectation (linear in Bx for small Bx) and requires no time integration,
    so it isolates the operator algebra from any integrator error.
    """
    cfg = cfg or LindbladConfig()
    H = -cfg.gamma_eff * Bx * SX
    rho = steadystate(H, collapse_operators(cfg))
    return {
        "Sx": float((rho * SX).tr().real),
        "Sy": float((rho * SY).tr().real),
        "Sz": float((rho * SZ).tr().real),
        "Bx": Bx,
    }


def evolve(
    t_list: np.ndarray,
    Bx_of_t: Callable[[float], float],
    cfg: LindbladConfig | None = None,
) -> dict:
    """
    Time-evolve the spin under a time-dependent transverse field Bx(t).

    Parameters
    ----------
    t_list : time grid (s)
    Bx_of_t : callable returning the transverse field (T) at time t
    cfg : LindbladConfig

    Returns dict of <Sx>,<Sy>,<Sz> arrays. <Sy> is the SENSING channel.
    """
    _ensure_qutip()
    cfg = cfg or LindbladConfig()
    H_op = -cfg.gamma_eff * SX

    # QuTiP 5 deprecates f(t, args) in favour of f(t, **kwargs); accept both.
    def coeff(t, *args, **kwargs):
        return Bx_of_t(t)

    H = [0.0 * SZ, [H_op, coeff]]
    psi0 = basis(2, 0)                      # start fully polarised along +z
    res = mesolve(H, psi0, t_list, collapse_operators(cfg), e_ops=[SX, SY, SZ])
    return {
        "t": np.asarray(t_list),
        "Sx": np.asarray(res.expect[0]),
        "Sy": np.asarray(res.expect[1]),
        "Sz": np.asarray(res.expect[2]),
    }


def linewidth_scan(B_range: Sequence[float], cfg: LindbladConfig | None = None) -> dict:
    """
    Sweep a static transverse field and record the steady-state response.

    Used by the V&V suite to (a) confirm linearity in the small-field limit
    and (b) extract the magnetic resonance half-width, which must match the
    analytic prediction B_HWHM = (R_op + Gamma_sd)/gamma_eff.
    """
    cfg = cfg or LindbladConfig()
    B = np.asarray(list(B_range), dtype=float)
    Sy = np.array([steady_state_response(float(b), cfg)["Sy"] for b in B])
    Sz = np.array([steady_state_response(float(b), cfg)["Sz"] for b in B])
    return {"B": B, "Sy": Sy, "Sz": Sz,
            "B_hwhm_predicted": cfg.total_relaxation / cfg.gamma_eff}


def availability() -> dict:
    _ensure_qutip()
    return {"qutip_available": QUTIP_AVAILABLE, "error": _QUTIP_ERR}
