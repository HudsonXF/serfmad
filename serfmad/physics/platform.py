"""
serfmad.physics.platform
========================
AUV platform kinematics, attitude dynamics, ambient-field nulling chain,
and the sensor error terms that dominate a moving SERF payload.

THE CENTRAL FEASIBILITY PROBLEM
-------------------------------
A SERF magnetometer only works when the residual STATIC field at the vapour
cell is within roughly +/-5 nT [Borna22]. The Earth's field is ~50,000 nT.
The required suppression is therefore ~10,000x (80 dB).

Worse, on a moving platform the AMBIENT FIELD IS NOT STATIC IN THE SENSOR
FRAME: as the vehicle rolls and pitches, the Earth field vector sweeps
through the sensor's body axes. A roll of angle theta projects a field
component B_earth*sin(theta) onto a body axis. For theta = 3 degrees this is

    50,000 nT * sin(3 deg) = 2,617 nT

which is ~500x outside the SERF operating window. This module models that
chain explicitly, because it -- not the sensor's intrinsic noise floor -- is
the true limiting factor for the application, and any feasibility claim that
ignores it is meaningless.

MITIGATION CHAIN MODELLED
-------------------------
  1. Passive shielding (mu-metal / ferrite): factor S_passive
  2. Active 3-axis nulling coils driven by a co-located reference
     magnetometer (fluxgate), closing a servo loop: factor S_active,
     limited by loop bandwidth and reference-sensor noise
  3. Residual -> must land inside +/-5 nT for SERF operation
  4. Whatever residual remains produces Cross-Axis Projection Error (CAPE),
     modelled per [Borna22]: a residual transverse field tilts the effective
     sensitive axis and introduces gain/phase error.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from ..constants import (
    EARTH_FIELD_NOMINAL_T,
    SERF_RESIDUAL_FIELD_TOLERANCE_T,
    OEHLER_RAW_INTERFERENCE_T,
    OEHLER_COMPENSATED_RESIDUAL_T,
)


@dataclass
class SeaState:
    """
    Platform attitude disturbance from wave action.

    Amplitudes/frequencies are ENGINEERING ASSUMPTIONS representative of a
    small UUV in a moderate sea; they are not measured values. They are
    chosen so roll and pitch occupy distinct frequencies, which is what makes
    reference-based cancellation tractable.
    """
    roll_amp_rad: float = np.radians(3.0)
    roll_freq_hz: float = 0.25
    pitch_amp_rad: float = np.radians(1.5)
    pitch_freq_hz: float = 0.40
    yaw_amp_rad: float = np.radians(1.0)
    yaw_freq_hz: float = 0.10
    label: str = "moderate (SS3-4 equivalent)"

    def attitude(self, t):
        """Return (roll, pitch, yaw) in radians at time t."""
        t = np.asarray(t, dtype=float)
        roll = self.roll_amp_rad * np.sin(2 * np.pi * self.roll_freq_hz * t)
        pitch = self.pitch_amp_rad * np.sin(2 * np.pi * self.pitch_freq_hz * t + 0.7)
        yaw = self.yaw_amp_rad * np.sin(2 * np.pi * self.yaw_freq_hz * t + 1.3)
        return roll, pitch, yaw


SEA_STATES = {
    "calm (SS1)": SeaState(np.radians(0.5), 0.20, np.radians(0.3), 0.35,
                           np.radians(0.2), 0.08, "calm (SS1)"),
    "moderate (SS3-4)": SeaState(),
    "rough (SS5-6)": SeaState(np.radians(8.0), 0.30, np.radians(4.0), 0.45,
                              np.radians(3.0), 0.12, "rough (SS5-6)"),
}


def rotation_matrix(roll, pitch, yaw):
    """
    Body-from-world rotation matrix, ZYX (yaw-pitch-roll) convention.

    Returns (...,3,3). Applied as B_body = R @ B_world.
    """
    roll = np.asarray(roll, dtype=float)
    pitch = np.asarray(pitch, dtype=float)
    yaw = np.asarray(yaw, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    shape = np.broadcast(roll, pitch, yaw).shape
    R = np.zeros(shape + (3, 3))
    R[..., 0, 0] = cp * cy
    R[..., 0, 1] = cp * sy
    R[..., 0, 2] = -sp
    R[..., 1, 0] = sr * sp * cy - cr * sy
    R[..., 1, 1] = sr * sp * sy + cr * cy
    R[..., 1, 2] = sr * cp
    R[..., 2, 0] = cr * sp * cy + sr * sy
    R[..., 2, 1] = cr * sp * sy - sr * cy
    R[..., 2, 2] = cr * cp
    return R


@dataclass
class NullingSystem:
    """
    Passive shield + active nulling servo that brings the ambient field into
    the SERF operating window.

    passive_factor : field suppression of the mu-metal/ferrite shield.
        A single mu-metal layer gives ~10-100x; multi-layer shields reach
        10^3-10^5 in laboratory settings. A UUV-compatible shield must be
        small and is assumed conservatively.
    active_factor : additional suppression from the closed-loop coil servo.
    servo_bandwidth_hz : above this frequency the servo cannot track, so
        disturbance rejection rolls off -- this is why platform motion
        frequency matters, not just amplitude.
    reference_noise_T : noise floor of the reference magnetometer driving
        the servo; the loop cannot null better than its own reference.
    """
    passive_factor: float = 100.0
    active_factor: float = 100.0
    servo_bandwidth_hz: float = 20.0
    reference_noise_T: float = 100e-12    # 100 pT fluxgate-class reference

    @property
    def dc_suppression(self) -> float:
        return self.passive_factor * self.active_factor

    def suppression_at(self, freq_hz):
        """
        Frequency-dependent suppression. The active loop contributes its full
        gain below the servo bandwidth and rolls off as a single pole above
        it; the passive shield is frequency-independent in this model
        (a simplification -- real mu-metal permeability falls at high
        frequency, while eddy-current shielding improves; both effects are
        outside the <5 Hz band of interest here).
        """
        freq_hz = np.asarray(freq_hz, dtype=float)
        active = 1.0 + (self.active_factor - 1.0) / np.sqrt(
            1.0 + (freq_hz / self.servo_bandwidth_hz) ** 2)
        return self.passive_factor * active

    def residual_field(self, B_in, freq_hz=0.0):
        """
        Residual field after shielding+nulling, floored by reference noise.

        The servo cannot null a field it cannot measure, so the residual can
        never be better than the reference magnetometer's own noise floor.
        """
        supp = self.suppression_at(freq_hz)
        return np.maximum(np.abs(B_in) / supp, self.reference_noise_T)

    def serf_viable(self, B_in, freq_hz=0.0) -> bool:
        return bool(np.all(self.residual_field(B_in, freq_hz)
                           <= SERF_RESIDUAL_FIELD_TOLERANCE_T))


@dataclass
class AUVPlatform:
    """AUV carrying a SERF gradiometer pair."""
    speed_mps: float = 2.0
    depth_m: float = 0.0
    start_x_m: float = -60.0
    baseline_m: float = 0.5
    sea_state: SeaState = None
    nulling: NullingSystem = None
    self_noise_T: float = OEHLER_RAW_INTERFERENCE_T
    heading_rad: float = 0.0

    def __post_init__(self):
        if self.sea_state is None:
            self.sea_state = SEA_STATES["moderate (SS3-4)"]
        if self.nulling is None:
            self.nulling = NullingSystem()

    def position(self, t):
        """Vehicle centre position (m) in the world frame at time t."""
        t = np.asarray(t, dtype=float)
        x = self.start_x_m + self.speed_mps * t * np.cos(self.heading_rad)
        y = self.speed_mps * t * np.sin(self.heading_rad)
        z = np.full_like(x, self.depth_m)
        return np.stack([x, y, z], axis=-1)

    def sensor_positions(self, t):
        """
        Positions of the two gradiometer cells, separated along the body
        x-axis (fore/aft), rotated into the world frame by vehicle attitude.
        Returns (…,2,3).
        """
        t = np.asarray(t, dtype=float)
        centre = self.position(t)
        roll, pitch, yaw = self.sea_state.attitude(t)
        R = rotation_matrix(roll, pitch, yaw)          # body-from-world
        # body x-axis expressed in world = first ROW of body-from-world matrix
        body_x_world = R[..., 0, :]
        half = 0.5 * self.baseline_m
        s1 = centre + half * body_x_world
        s2 = centre - half * body_x_world
        return np.stack([s1, s2], axis=-2)

    def ambient_field_body(self, t, B_world=None):
        """
        Earth's field expressed in the BODY frame.

        This is the term that makes SERF hard on a moving platform: even a
        perfectly stable Earth field becomes a large time-varying signal in
        body axes as the vehicle rolls.
        """
        if B_world is None:
            # Mid-latitude field: mostly vertical (inclination ~70 deg,
            # representative of Nova Scotia / North Atlantic operating areas)
            inc = np.radians(70.0)
            B_world = EARTH_FIELD_NOMINAL_T * np.array(
                [np.cos(inc), 0.0, np.sin(inc)])
        roll, pitch, yaw = self.sea_state.attitude(t)
        R = rotation_matrix(roll, pitch, yaw)
        return np.einsum('...ij,j->...i', R, B_world)

    def motion_induced_field_swing(self, t) -> np.ndarray:
        """
        Peak-to-peak variation of the body-frame ambient field caused purely
        by attitude motion. This is the quantity that must be nulled.
        """
        B_body = self.ambient_field_body(t)
        return B_body - B_body.mean(axis=0, keepdims=True)


def cape_error(B_residual_transverse: float, B_signal: float,
               R_tot: float, gamma_e: float) -> dict:
    """
    Cross-Axis Projection Error per [Borna22].

    A residual static field transverse to the nominal sensitive axis tilts
    the effective sensing axis and scales the gain. To first order the
    sensitive-axis tilt is

        alpha = arctan( gamma_e * B_res / R_tot )

    and the on-axis gain is reduced by cos(alpha), with a cross-term
    sin(alpha) leaking the orthogonal field component into the output.

    This is a REAL, DOCUMENTED error mechanism, not a modelling artefact,
    and it is why the residual-field budget in `NullingSystem` matters even
    when the residual is comfortably inside the nominal SERF window.
    """
    alpha = np.arctan2(gamma_e * B_residual_transverse, R_tot)
    return {
        "tilt_rad": alpha,
        "tilt_deg": np.degrees(alpha),
        "gain_factor": np.cos(alpha),
        "cross_axis_leakage": np.sin(alpha),
        "signal_error_T": B_signal * (1.0 - np.cos(alpha)),
    }


def field_budget(platform: AUVPlatform, t: np.ndarray) -> dict:
    """
    Full ambient-field budget for a mission leg -- the key feasibility table.

    Returns each stage of the suppression chain in tesla so the report can
    show exactly where the 10,000x comes from and whether it closes.
    """
    B_body = platform.ambient_field_body(t)
    swing = np.abs(B_body - B_body.mean(axis=0, keepdims=True)).max()
    static = np.abs(B_body).max()
    f_dom = max(platform.sea_state.roll_freq_hz, platform.sea_state.pitch_freq_hz)

    supp = float(platform.nulling.suppression_at(f_dom))
    residual_static = max(static / supp, platform.nulling.reference_noise_T)
    residual_dynamic = max(swing / supp, platform.nulling.reference_noise_T)

    return {
        "ambient_static_T": static,
        "motion_swing_T": swing,
        "dominant_motion_freq_hz": f_dom,
        "suppression_at_freq": supp,
        "residual_static_T": residual_static,
        "residual_dynamic_T": residual_dynamic,
        "serf_tolerance_T": SERF_RESIDUAL_FIELD_TOLERANCE_T,
        "serf_viable": residual_static <= SERF_RESIDUAL_FIELD_TOLERANCE_T,
        "margin_factor": SERF_RESIDUAL_FIELD_TOLERANCE_T / residual_static
        if residual_static > 0 else np.inf,
    }
