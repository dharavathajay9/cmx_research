# CMX Research — RGB–Thermal Semantic Segmentation

Deep-learning research on the CMX RGB–Thermal semantic segmentation framework:
experiments **E0–E5** plus the **CRD-Net** counterfactual reliability experiment,
with their training code, evaluation notebooks, result figures and reports.

## Repository layout

| Path | Contents |
|---|---|
| `source_projects/CMX_Original` | **E0** – CMX official baseline (from `huaaaliu/RGBX_Semantic_Segmentation`) with local training/eval work |
| `source_projects/CMX_E1` | **E1** – SegFormer encoder experiment |
| `source_projects/CMX_E2` | **E2** – learnable weighted fusion experiment |
| `source_projects/CMX_E3` | **E3** – corruption/reliability-aware fusion experiment |
| `source_projects/CMX_E4` | **E4** – E4 fusion experiment |
| `source_projects/CMX_E5` | **E5** – counterfactual experiment |
| `source_projects/CRD_Net` | **CRD-Net** – counterfactual reliability distillation experiment (training, evaluation, comparison figures) |
| `experiments/` | Per-experiment descriptors (E0–E5) |
| `research_assets/` | Curated results: model-comparison tables, plots, presentation results, matrices |
| `docs/source_reports/` | Research reports (PDF/DOCX) |
| `results/`, `configs/`, `evaluation/`, `models/`, `scripts/`, `visualizations/` | Shared scaffolding for metrics, configs and tooling |
| `requirements.txt` | Python dependencies |

Each experiment under `source_projects/` is self-contained: `config.py`,
`train.py`, `eval.py`, `dataloader/`, `models/`, `engine/`, notebooks,
`comparison_results/` figures and training/evaluation logs.

## Not stored in git (kept on disk only)

`.gitignore` intentionally excludes:

- **Model checkpoints / weights** — `*.pth`, `*.pt`, `*.ckpt`, `*.onnx`, `*.safetensors` (~190 GB)
- **Python virtualenvs and caches** — `.venv/`, `__pycache__/`, `*.pyc`
- **Bulk per-sample prediction masks** — `predictions_*/` (regenerable dumps)
- **Bulk training-log image dumps** — everything `*.png`, *except* the
  re-included folders below: `datasets/` (TDD images), `results/`,
  `comparison_results/`, `CMX_E0_E5_ALL_GRAPHS/`, and the
  `research_assets/` figure folders

**Tracked data:** the TDD dataset (`datasets/TDD/{RGB,Thermal,Label}` —
7062 files plus the train/val/test split lists, ~978 MB per copy, in
`CMX_Original` and `CMX_E3`) is committed on purpose so a fresh clone is
self-contained.

Everything else — code, notebooks, logs, CSV/JSON metrics, reports and the
curated comparison figures — is tracked. The environment is pinned in
`requirements.txt` (Python 3.10, torch 2.14.1, timm 1.0.30).

## Standalone per-experiment repositories

Each experiment is also published on its own, split from this tree:

- [`cmx_e0`](https://github.com/dharavathajay9/cmx_e0) — `source_projects/CMX_Original`
- [`cmx_e1`](https://github.com/dharavathajay9/cmx_e1) — `source_projects/CMX_E1`
- [`cmx_e2`](https://github.com/dharavathajay9/cmx_e2) — `source_projects/CMX_E2`
- [`cmx_e3`](https://github.com/dharavathajay9/cmx_e3) — `source_projects/CMX_E3`
- [`cmx_e4`](https://github.com/dharavathajay9/cmx_e4) — `source_projects/CMX_E4`
- [`cmx_e5`](https://github.com/dharavathajay9/cmx_e5) — `source_projects/CMX_E5`
- [`crd_net`](https://github.com/dharavathajay9/crd_net) — `source_projects/CRD_Net`
