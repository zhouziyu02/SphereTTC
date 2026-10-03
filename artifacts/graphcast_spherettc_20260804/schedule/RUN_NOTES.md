# Operational notes

- The formal 2020 raw evaluation completed successfully at
  `2026-08-04T12:34:33Z`; its complete `(366, 5, 49)` metrics and A100 profile
  are stored in `metrics/holdout_2020/raw.npz` and
  `profiles/holdout_2020_raw.json`.
- At `2026-08-04T12:37:54Z`, an unrelated duplicate raw launch without CUDA
  visibility failed before metric computation and overwrote only
  `logs/holdout_2020_raw.log`. It did not overwrite the successful metrics or
  profile (their timestamps precede the duplicate). The failed log is retained
  unchanged for transparency.
- All eight frozen SphereTTC candidate logs completed with `returncode=0`.
