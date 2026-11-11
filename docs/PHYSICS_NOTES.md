# Physics notes

Short keys below map to full citations in `docs/REFERENCES.md`; the complete
bibliography with DOIs is in `docs/SR_report.pdf`.

## SERF Bloch equation convention

We adopt the Seltzer/Borna form in which 1/q multiplies the entire RHS
[Seltzer08], [Borna22]:

    dP/dt = (1/q)[ gamma_e (P x B) + R_op(s - P) - R_rel P ]

The slowing-down factor q arises from rapid spin exchange redistributing
angular momentum among hyperfine states [Happer77], [Savukov05]; the same
source back-calculates gamma_e/2pi = 27.9 Hz/nT, against 28 Hz/nT in [Wei23].

Consequence: q sets the response BANDWIDTH but cancels from the DC steady
state, so the DC resonance width in field units is independent of q. Some
papers place gamma_e/q on the precession term only, in which case q survives
in the DC width. We verify the resulting linewidth numerically (V&V X-01)
against the published +/-5 nT SERF window [Borna22] rather than asserting a
formula.

## Relaxation anisotropy (why the two models initially disagreed)

The textbook SERF Bloch equation uses a single isotropic relaxation rate,
implicitly R_1 = R_2 = R_op + Gamma_sd [Seltzer08]. That is an approximation:
optical pumping is inherently longitudinal. Diagonalising the Lindblad
superoperator [Lindblad76] for the collapse set in `quantum/lindblad.py`,
integrated with QuTiP [QuTiP13], gives

    R_1 = R_op + Gamma_sd                    (longitudinal)
    R_2 = (R_op + Gamma_sd)/2 + Gamma_sd     (transverse)

verified numerically: eigenvalues -505 and -257.5 s^-1 for R_op=500,
Gamma_sd=5. The Gamma_sd order of magnitude follows the 1.1 Hz K linewidth
measured at n=1e14 cm^-3 in [Allred02]. BlochConfig exposes both;
isotropic=True reproduces the textbook SERF form (production default,
comparable with published SERF modelling), isotropic=False uses the exact
Lindblad-derived rates (used by V&V test V-08, which then agrees to 2.8e-16).

Getting this wrong produced an initial 30% cross-model disagreement that was
NOT a solver bug -- it was a comparison between two different physical models.

## Why semiclassical is legitimate

A SERF vapour cell holds ~1e13-1e14 atoms [Allred02], [Griffith10]; the
ensemble-averaged spin obeys the Bloch equation to excellent approximation
[Seltzer08], [Budker07]. Inter-atomic quantum correlations matter only for
spin-squeezing schemes, which are out of scope -- so this model cannot predict
an absolute fT/rtHz figure of the kind reported in [Kominis03] or [Dang10].
The Lindblad model [Lindblad76] is retained as a single-spin reference
precisely to DEMONSTRATE this equivalence rather than assume it.

## Dipole field and gradiometry

The target signature is the static field of a magnetic dipole [Jackson98],
falling as 1/r^3, so a fixed-baseline gradiometer sees 1/r^4 and detection
range scales as m^(1/4) (stress test S-10, measured exponent 0.244). Magnetic
anomaly detection range laws follow [Wynn75]; platform interference is removed
by the Tolles-Lawson model [Tolles50], [Leliak61], with the aeromagnetic
compensation treatment of [Leach80] and the modern benchmark data of
[Gnadt22]. The post-compensation UUV residual used throughout (2.23 nT from a
raw 246.9 nT) is the measured figure in [Oehler24].

## Detection theory

Matched filtering, CFAR thresholding and the ROC treatment follow [Kay98] and
[VanTrees01]. Monte Carlo sampling uses the generator of [Marsaglia72];
dipole orientation is drawn uniformly on the sphere (uniform in cos-theta),
since an adversary's orientation is never known.

## Numerical method

Fixed-step RK4 [Butcher16], chosen over adaptive stepping because every
ensemble member must share a time grid for vectorisation (adaptive per-member
stepping would serialise the batch) and the stiffness is bounded and known a
priori (fastest rate = max(R_1,R_2)/q), so the stability condition can be
checked up front. check_stability() enforces |lambda|*dt < 2.785. Measured
convergence order: 4.01 (V&V V-06).

Verification and validation are separated in the sense of [Oberkampf10]:
V-series tests check the code solves its equations correctly (against closed
forms and a second, independent model), X-series tests check those equations
against published measurement.
