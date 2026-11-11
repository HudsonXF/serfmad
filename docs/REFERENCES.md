# References

Full bibliography with DOIs is in `docs/SR_report.pdf` (46 entries). This file
maps the short keys used in code comments to those entries.

## Core SERF physics
- **[Allred02]** Allred, Lyman, Kornack & Romalis, *Phys. Rev. Lett.* **89**(13),
  130801 (2002). DOI:10.1103/PhysRevLett.89.130801.
  SERF regime demonstrated. 1.1 Hz linewidth for K at n=1e14 cm^-3;
  10 fT/rtHz limited by Johnson currents in the shields; shot-noise floor
  estimated at 2e-18 T/rtHz. Source for `Gamma_sd` order of magnitude.
- **[Kominis03]** Kominis, Kornack, Allred & Romalis, *Nature* **422**(6932),
  596-599 (2003). DOI:10.1038/nature01484. Sub-fT multichannel.
- **[Dang10]** Dang, Maloof & Romalis, *Appl. Phys. Lett.* **97**(15), 151110
  (2010). DOI:10.1063/1.3491215. 0.16 fT/rtHz -- best reported.
- **[Savukov05]** Savukov & Romalis, *Phys. Rev. A* **71**, 023405 (2005).
  DOI:10.1103/PhysRevA.71.023405. Spin-exchange in low field; slowing-down
  factor q. Back-calculates gamma_e/2pi = 27.9 Hz/nT.
- **[Seltzer08]** Seltzer, PhD dissertation, Princeton (2008). Source of the
  SERF Bloch equation convention used in `quantum/bloch.py`.
- **[Happer77]** Happer & Tam, *Phys. Rev. A* **16**, 1877 (1977).
  Rapid spin exchange and resonance narrowing.
- **[Budker07]** Budker & Romalis, *Nature Physics* **3**(4), 227-234 (2007).
  DOI:10.1038/nphys566. Review.
- **[Li18]** Li et al., *IEEE Sensors J.* **18**(20), 8198-8207 (2018). Review.
- **[Wei23]** Wei et al., *Front. Phys.* **11**, 1181399 (2023).
  States gamma_e/2pi = 28 Hz/nT.

## MEMS vapour cells
- **[Griffith10]** Griffith, Knappe & Kitching, *Opt. Express* **18**(26),
  27167-27172 (2010). DOI:10.1364/OE.18.027167. **Design point**: 5 fT/rtHz,
  87-Rb, cavity etched in 1 mm Si wafer, anodically bonded Pyrex, 1 mm^3
  measurement volume, pump/probe near 795 nm.
- **[Shah07]** Shah, Knappe, Schwindt & Kitching, *Nature Photonics* **1**(11),
  649-652 (2007). Sub-pT microfabricated cell.
- **[Liew04]** Liew et al., *Appl. Phys. Lett.* **84**(14), 2694-2696 (2004).
  Microfabricated alkali vapour cells -- anodic bonding.
- **[Schwindt04]** Schwindt et al., *Appl. Phys. Lett.* **85**(26), 6409 (2004).

## The binding constraint: CAPE and field tolerance
- **[Borna22]** Borna, Iivanainen, Carter, McKay, Taulu, Stephen & Schwindt,
  *NeuroImage* **247**, 118818 (2022). DOI:10.1016/j.neuroimage.2021.118818.
  **Source of the +/-5 nT SERF residual-field window and the CAPE model.**
- **[Borna22b]** Borna et al., *NeuroImage* **262**, 119559 (2022).
  Cross-axis dynamic field compensation.
- **[Seltzer04]** Seltzer & Romalis, *Appl. Phys. Lett.* **85**(20), 4804 (2004).
  Unshielded three-axis vector SERF operation.
- **[Fang12]** Fang & Qin, *Rev. Sci. Instrum.* **83**(10), 103104 (2012).
  In-situ triaxial compensation.
- **[Lee08]** Lee & Romalis, *J. Appl. Phys.* **103**(8), 084904 (2008).
  Magnetic-shield Johnson noise -- floors R-05.

## Platform compensation and marine MAD
- **[Oehler24]** Oehler et al., *Geophysical Prospecting* **72** (2024).
  DOI:10.1111/1365-2478.13556. **UUV interference 246.90 -> 2.23 nT RMS after
  compensation; sub-10 nT anomaly detection.** The clutter benchmark.
- **[Seidel23]** Seidel et al., BASTA project, underwater UXO via hovering AUVs.
- **[Tolles50]** Tolles & Lawson, Airborne Instruments Laboratory Rept. 201-1
  (1950). Original T-L compensation model.
- **[Gnadt22]** Gnadt, Wollaber & Nielsen, arXiv:2212.09899 (2022).
  Modern derivation of Tolles-Lawson.
- **[Leliak61]** Leliak, *IRE Trans. Aerospace Nav. Electronics* **8**(3), 95
  (1961). Calibration manoeuvres / figure of merit.
- **[Leach80]** Leach (1980). Ridge regression for T-L -- justifies the
  Tikhonov term in `processing/compensation.py`.
- **[Holmes06]** Holmes, *Exploitation of a Ship's Magnetic Field Signatures*,
  Morgan & Claypool (2006). Submarine moment ranges.

## Detection theory, numerics, general
- **[VanTrees01]** Van Trees, *Detection, Estimation, and Modulation Theory,
  Part I*, Wiley (2001). Matched filter optimality.
- **[Kay98]** Kay, *Statistical Signal Processing Vol. II: Detection Theory*,
  Prentice Hall (1998). CFAR.
- **[Jackson98]** Jackson, *Classical Electrodynamics*, 3rd ed., Ch. 5.
  Dipole field and gradient tensor.
- **[Wynn75]** Wynn et al., *IEEE Trans. Magnetics* **11**(2), 701 (1975).
  Gradient tensor invariants for MAD.
- **[Lindblad76]** Lindblad, *Commun. Math. Phys.* **48**(2), 119 (1976).
- **[QuTiP13]** Johansson, Nation & Nori, *Comput. Phys. Commun.* **184**, 1234
  (2013).
- **[Butcher16]** Butcher, *Numerical Methods for ODEs*, 3rd ed. RK4 stability.
- **[Marsaglia72]** Marsaglia, *Ann. Math. Statist.* **43**(2), 645 (1972).
  Uniform sampling on the sphere -- used in the GPU orientation sweep.
- **[Oberkampf10]** Oberkampf & Roy, *Verification and Validation in Scientific
  Computing*, CUP (2010). V&V terminology.
- **[NCEI]** NOAA NCEI Geomagnetism FAQ / IGRF-13. Earth field 25,000-65,000 nT.

## Magnetic shielding materials and shield design
- **[Gohil20]** Gohil, Burrows, Blaskovic Kraljevic, Schulte & Heilig,
  "Measurements of sub-nT dynamic magnetic field shielding with soft iron and
  mu-metal for use in linear colliders," arXiv:2009.01519v2 (2020).
  **Measured** initial permeability mu_i = 55,955 +/- 7 for an annealed
  mu-metal cylinder (ID 5 cm, wall 1 mm, length 50 cm; 80Ni-15Fe-4.5Mo-0.4Mn-
  0.1Si), against an advertised 50,000; soft iron measured mu_i = 204 +/- 5
  against an advertised 300-500. **Critically**: three shields formed from
  mu-metal foil and NOT re-annealed measured mu_r = 3,670 / 3,602 / 4,660 --
  more than an order of magnitude below the annealed bulk value. Re-annealing
  requires >1000 C. Advertised B_sat: mu-metal 0.74 T, soft iron 2.15 T.
  Source for `MU_METAL_ANNEALED` and `MU_METAL_FOIL_FORMED`.
- **[Dubbers86]** Dubbers, *Nucl. Instrum. Meth. A* **243**, 511-517 (1986).
  Single-layer cylindrical shield amplitude response T = D / (mu_r * Delta),
  i.e. shielding factor S = mu_r * Delta / D for diameter D and wall
  thickness Delta. The closed form used in `materials.shielding_factor`.
- **[Sumner87]** Sumner, Pendlebury & Smith, *J. Phys. D: Appl. Phys.* **20**,
  1095 (1987). Standard treatment of practical magnetic shielding, including
  multi-shell configurations.
- **[Hoburg96]** Hoburg, *IEEE Trans. Electromagn. Compat.* **38**, 93-103
  (1996). Method for calculating the shielding factor of cylindrical shields;
  the transfer-function model fitted by [Gohil20].
- **[Celozzi08]** Celozzi, Araneo & Lovat, *Electromagnetic Shielding*, Wiley
  (2008). Flux-shunting vs eddy-current shielding mechanisms.

## Structural materials and magnetic cleanliness
- **[ASSDA]** Australian Stainless Steel Development Association, Technical
  FAQ 3, "Magnetic Effects of Stainless Steels." Annealed 316/317L magnetic
  permeability generally < 1.02 at 200 oersteds; DC permeabilities 1.003-1.005
  at a magnetising force of 200 Oe.
- **[BSSA]** British Stainless Steel Association, "Effect of cold work and heat
  treatment on the magnetic permeability of austenitic stainless steels."
  Cold work transforms austenite to martensite and raises permeability; grades
  with high austenite stability (304LN, 316LN, 310) retain low permeability in
  BOTH annealed and cold-worked conditions. Gives no numerical cold-worked
  values -- hence the cold-worked entry in `materials.py` is an
  EngineeringAssumption, not a cited datum.
- **[Electralloy50]** Electralloy / Carlson, NITRONIC 50 (XM-19, UNS S20910)
  product data sheet. Nitrogen-strengthened austenitic stainless; typical
  annealed magnetic permeability 1.004 at 50, 100 and 200 oersteds; remains
  low-permeability in the cold-worked condition and at cryogenic temperature.
- **[MatWebTi64]** MatWeb material data sheet, "Titanium Ti-6Al-4V (Grade 5),
  Annealed." Density, modulus, tensile and thermal properties. NOTE: open
  sources conflict on the magnetic susceptibility of Ti-6Al-4V (values from
  chi ~ -1.7e-6 to ~ +1.8e-4 SI appear in the literature), so the
  susceptibility used here is an EngineeringAssumption carrying that range,
  NOT a cited value.

## Towed-body mechanics and sensor spatial separation
- **[Seidel23sheet]** CONMAR project / GEOMAR Helmholtz Centre for Ocean
  Research Kiel, "AUV-based Magnetometry" product sheet (accessed 2026).
  Girona 500 hovering AUV carrying FOUR SENSYS FGM3D/100 UW II three-axis
  fluxgates sampled at 200 Hz, giving 2.5 mm along-track spacing at 0.5 m/s;
  the array geometry resolves all three spatial magnetic gradients. States
  that "the rigid construction that holds the magnetometers suppresses spatial
  errors that towed systems usually produce" -- the primary open-literature
  statement of the rigid-boom-over-tow-cable trade. Survey line spacing 1-3 m.
  Companion to [Seidel23].
- **[WaveGlider25]** "An Affordable Wave Glider-Based Magnetometry System for
  Marine Magnetic Measurement," *J. Mar. Sci. Eng.* **13**(11), 2089 (2025).
  DOI:10.3390/jmse13112089. Twin-body towed configuration with an S-shaped
  towing cable to decouple platform motion. **Measured**: mean cruising speed
  0.56 m/s; towed-body pitch held within +/-5 deg and roll within +/-1 deg;
  dynamic noise 0.0639 nT; repeat-line and cross-line internal consistency
  1.832 nT and 1.956 nT. The measured towed-body attitude figures used in
  Section 3.

## Source explicitly REJECTED
A widely-repeated open-web claim that degaussed submarines present
"1000-2000 nT signatures at 200-600 m" was tested and rejected: producing
1000 nT at 200 m requires m ~ 4e7 A*m^2, an order of magnitude ABOVE the
literature value for an UNdegaussed hull, so the claim is self-contradictory.
The source was a scraped-content wiki and is not citable. See
`vv_suite.x05_detection_range_law`.
