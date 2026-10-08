# SphereTTC source provenance

`pre_consolidation_20261003/src/ttc/sphere.py` and `memory.py` are the original
SphereTTC calibration and eligible-memory implementations. They are preserved
byte-for-byte for `verify_migration.py` (AST equivalence) and
`tests/test_consolidation_equivalence.py` (numerical equivalence).

The active implementation is `src/spherettc.py`. The archived files are not
runtime entry points. Earlier mixed-project snapshots and migration utilities
were removed from this focused checkout on 2026-10-08; the complete pre-cleanup
checkout was saved separately before the change. Git history is unchanged.
