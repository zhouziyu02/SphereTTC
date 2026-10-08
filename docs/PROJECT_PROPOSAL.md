# SphereTTC: Spherical Test-Time Calibration for Medium-Range Weather Forecasting

## Research scope

This project studies a lightweight way to improve frozen global weather forecasts using feedback from previously verified forecasts. The focus is a single-backbone calibrator: each forecasting system supplies its own forecasts, and SphereTTC uses that system's historical forecast–truth pairs to correct subsequent outputs without gradient training of the backbone.

Global weather fields have spherical geometry, spatially coherent error structures and lead-dependent biases. Verification is delayed: an observation cannot be used simply because its timestamp matches the current initialization. The research protocol must account for the end of any daily averaging window and the observation product's actual availability time.

## Method

SphereTTC represents forecast fields in a truncated spherical-harmonic space and estimates conservative affine corrections with ridge shrinkage toward identity. Recency weighting emphasizes recent conditions, robust weighting limits anomalous residual influence, and a chronological validation gate controls the correction strength. Degree tapering and a no-op fallback provide further controls. The operational forecasting model stays frozen.

The retained implementation and main experiments use this single-backbone method. Multiple timescale and seasonal calibration experts remain experimental variants without frozen main-result evidence.

## Evaluation and evidence

The evaluation uses ERA5 daily means on a 1.5° grid, 49 shared variables and 1–10-day leads. It covers six locally trained backbones and five released forecasting systems. The main tables contain 22 raw/calibrated configurations; parameters were numerically selected on 2017, with evaluation on 2018–2019 and a subsequent GraphCast 2020 holdout.

Existing frozen records show macro nRMSE improvement across the 11 backbones, with exceptions in ACC and individual variable–lead cells. These are preliminary findings under the original evaluation protocol. The documented daily-mean truth-availability issue affects the five released models, including the GraphCast 2020 record, and must be fixed and rerun before making final causal-deployment claims. GraphCast's search history, differing selection protocols and potential humidity-conversion effects also require disclosure and checks.

## Planned validation

The next work is to repair and test truth-availability handling, repeat affected parameter selection and evaluation, compare simple bias correction and existing TTC methods under a common protocol, and measure the contributions of spherical representation and each risk-control component. Further evaluation should report variable-level failures, ACC, observation-delay sensitivity, future-year performance and computational cost. Novelty claims require a separate review of both machine-learning calibration and numerical-weather-prediction post-processing literature.
