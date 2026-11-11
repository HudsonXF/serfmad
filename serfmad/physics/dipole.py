"""
serfmad.physics.dipole
======================
Magnetic point-dipole field and gradient tensor.

The full vector dipole field is used throughout -- NOT a scalar or
single-component shortcut. This matters because:
  (a) the SERF sensor is a VECTOR magnetometer sensitive to a specific
      transverse axis, so component structure is physically meaningful; and
  (b) an earlier iteration of this project used a truncated B_z expression
      with a hard-coded constant in place of a field-point-dependent term,
      which produced up to 800% field error across the survey domain. The
      full-vector form removes that class of error by construction.

CANONICAL FORM
--------------
    B(r) = (mu0/4pi) * [ 3 r_hat (m . r_hat) - m ] / |r|^3

Gradient tensor (used for gradiometry and for the tensor-invariant detector):
    dB_i/dx_j = (3*mu0/4pi/r^4) * [ m_i n_j + m_j n_i + n_i n_j (m.n)
                                     - 5 n_i n_j (m.n) + delta_ij (m.n) ] ...
We compute the gradient by exact analytic differentiation (see `gradient_tensor`)
and verify it against central finite differences in the V&V suite.
"""

from __future__ import annotations
import numpy as np
from ..constants import MU0_OVER_4PI


def dipole_field(r_vec: np.ndarray, m_vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Magnetic flux density of a point dipole.

    Parameters
    ----------
    r_vec : (...,3) displacement FROM the dipole TO the field point (m)
    m_vec : (3,) or (...,3) magnetic dipole moment (A*m^2)
    eps   : softening length (m) preventing a singularity at r=0. Chosen
            small relative to any physical standoff; results within ~eps of
            the source are not physically meaningful anyway (the point-dipole
            approximation itself fails within roughly one body length).

    Returns (...,3) field in tesla.
    """
    r_vec = np.asarray(r_vec, dtype=float)
    m_vec = np.asarray(m_vec, dtype=float)
    r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
    r = np.maximum(r, eps)
    r_hat = r_vec / r
    m_dot_rhat = np.sum(m_vec * r_hat, axis=-1, keepdims=True)
    return MU0_OVER_4PI * (3.0 * r_hat * m_dot_rhat - m_vec) / (r ** 3)


def gradient_tensor(r_vec: np.ndarray, m_vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Analytic magnetic gradient tensor G_ij = dB_i/dx_j of a point dipole.

    Derived by differentiating the canonical dipole form:
        G_ij = (3*mu0/4pi/r^4) * [ n_i m_j + n_j m_i + delta_ij (m.n)
                                    - 5 n_i n_j (m.n) ]

    The tensor is symmetric and traceless in free space (div B = 0 and
    curl B = 0), which the V&V suite checks explicitly as an independent
    correctness test -- a wrong gradient expression almost never happens to
    be both symmetric and traceless.

    Returns (...,3,3) in T/m.
    """
    r_vec = np.asarray(r_vec, dtype=float)
    m_vec = np.asarray(m_vec, dtype=float)
    r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
    r = np.maximum(r, eps)
    n = r_vec / r
    m_dot_n = np.sum(m_vec * n, axis=-1)

    shape = np.broadcast_shapes(n.shape[:-1], np.asarray(m_vec).shape[:-1])
    n_b = np.broadcast_to(n, shape + (3,))
    m_b = np.broadcast_to(m_vec, shape + (3,))
    r_b = np.broadcast_to(r, shape + (1,))
    mdn = np.broadcast_to(m_dot_n[..., None], shape + (1,))

    delta = np.eye(3)
    ni_mj = n_b[..., :, None] * m_b[..., None, :]
    nj_mi = n_b[..., None, :] * m_b[..., :, None]
    ni_nj = n_b[..., :, None] * n_b[..., None, :]

    pref = 3.0 * MU0_OVER_4PI / (r_b[..., 0][..., None, None] ** 4)
    G = pref * (ni_mj + nj_mi + delta * mdn[..., None]
                - 5.0 * ni_nj * mdn[..., None])
    return G


def gradient_tensor_fd(r_vec: np.ndarray, m_vec: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """
    Central finite-difference gradient tensor -- INDEPENDENT check on the
    analytic expression above. Used only in V&V, never in production paths.
    """
    r_vec = np.asarray(r_vec, dtype=float)
    G = np.zeros(r_vec.shape[:-1] + (3, 3))
    for j in range(3):
        dr = np.zeros(3)
        dr[j] = h
        Bp = dipole_field(r_vec + dr, m_vec)
        Bm = dipole_field(r_vec - dr, m_vec)
        G[..., :, j] = (Bp - Bm) / (2.0 * h)
    return G


def tensor_invariant_ctp(G: np.ndarray) -> np.ndarray:
    """
    Contraction of the gradient tensor (CTP scalar), a standard rotation-
    INVARIANT MAD detection statistic:

        C = sqrt( sum_ij G_ij^2 )

    Its value is independent of platform attitude, which is precisely why it
    is used operationally: it removes the vehicle-orientation dependence that
    otherwise dominates a moving-platform vector measurement.
    """
    return np.sqrt(np.sum(G ** 2, axis=(-2, -1)))


def moment_from_orientation(magnitude: float, theta: float, phi: float) -> np.ndarray:
    """Build a dipole moment vector from magnitude + spherical orientation."""
    return magnitude * np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def on_axis_peak_field(moment: float, r: float) -> float:
    """
    Closed-form on-axis field magnitude, B = 2*(mu0/4pi)*m/r^3.

    Kept as a separate scalar function so the V&V suite can check the full
    vector implementation against a textbook expression at a known geometry.
    """
    return 2.0 * MU0_OVER_4PI * moment / (r ** 3)


def detection_range(moment: float, b_min: float) -> float:
    """
    Range at which an on-axis dipole produces exactly b_min.
        r = (2*(mu0/4pi)*m / b_min)^(1/3)
    This is the standard 1/r^3 range law used for MAD performance prediction.
    """
    return (2.0 * MU0_OVER_4PI * moment / b_min) ** (1.0 / 3.0)
