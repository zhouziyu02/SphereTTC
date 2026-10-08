# Read-only historical provenance

These integrity snapshots and audits describe the original 2026-08-04 experimental environment. They are retained verbatim as historical evidence. Their inventory may include paths from removed projects and external assets; those paths are not current repository dependencies, and the old whole-repository checks must not be run against the current focused tree.

Current reproducibility is checked by the root `verify_migration.py`, including independent frozen hashes, aggregate CSV regeneration, GraphCast scalar summaries, and SphereTTC core AST equivalence. This directory is not imported by the active calibration or table-generation code.
