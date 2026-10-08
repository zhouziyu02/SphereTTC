# SphereTTC

For an architecture figure in an academic paper (e.g., CVPR), refer to:

- **[src/spherettc.py](src/spherettc.py)** — The complete SphereTTC method. `eligible_from_buffer` selects historical forecast–verification pairs; `SphereTTCCalibrator` implements spherical-harmonic encoding, weighted affine fitting, chronological validation gating, spectral tapering, and inverse-transform correction. `SphereTTC.calibrate()` and `SphereTTC.update()` define the online interface around a frozen forecasting backbone. **This file is sufficient for the main architecture figure.**
- **[scripts/run_ttc.py](scripts/run_ttc.py)** — Optional execution context. Follow the `--method sphere_ttc` path and `calibrate_sphere_lead()` for the chronological loop, per-lead memory, and calibrated forecast reconstruction.
