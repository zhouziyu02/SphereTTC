# archive/ —— 历史与溯源（不参与运行）

| 路径 | 内容 |
|---|---|
| `ORIGINAL_SNAPSHOT_20261002.tar.gz` | 2026-10-02 整理前整个仓库的完整快照（422 个文件，SHA-256 见 `MANIFEST.sha256`）。确认整理无误后可以自行删除。 |
| `provenance/RESTRUCTURE_LOG_20261002.json` | 本次整理的每一次移动、每一处字符串替换，以及前后 SHA-256 |
| `provenance/restructure_20261002.py` | 执行整理的脚本（已运行，勿重复执行） |
| `provenance/MIGRATION_MANIFEST_20260806.sha256` | 2026-08-06 迁移包的原始哈希清单 |
| `provenance/verify_migration_20260806.py` | 2026-08-06 版验证器（原样保留；新版在仓库根目录） |
| `old_docs/README_full_archive_20260804.md` | 原 `README.md`，描述的是原服务器上约 14 GB 的全量归档 |
| `old_docs/README_ZH_lightweight_migration_20260806.md` | 原 `README_ZH.md`，描述 2026-08-06 的轻量迁移包 |
| `old_env/requirements_original.txt` | 原 `requirements.txt`（大量注释、未固定版本、缺少 torch-harmonics） |
| `old_env/sitecustomize.py` | 原托管容器的 psutil 兼容补丁，与实验无关 |
| `pre_consolidation_20261003/` | 2026-10-03 合并前的 SphereTTC / SphereDyn 原文件（`src/ttc/sphere.py`、`memory.py`，`src/baselines/models.py`、`spheredyn_v2/4/6/7/9.py`，`scripts/spheredyn/evaluate_goal_reference_ensemble.py`，`tests/test_spheredyn_v9.py`），逐字节保存；`tests/test_consolidation_equivalence.py` 用它们验证新单文件 |
| `provenance/CONSOLIDATION_LOG_20261003.json`、`provenance/consolidate_20261003.py` | 合并的日志与脚本（已运行，勿重复执行） |
| `full_archive_verifiers/` | 只能在原服务器全量归档里运行的三个验证/清单脚本，原样保留 |

如需完全回到整理前的状态：在空目录中执行 `tar -xzf ORIGINAL_SNAPSHOT_20261002.tar.gz`。
