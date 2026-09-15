# Isolating Feedback Utility in Adaptive Optimizer Scheduling

Code and released experiment outputs for the ICONIP 2026 paper. We compare AEES with fixed, uniform-random, and cross-fitted frequency-matched controls to isolate the additional benefit of training-loss feedback.

The paper covers CIFAR-100 (ResNet-18, AdamW and SGD with momentum) and AG News (DistilBERT). All accuracy comparisons use paired seeds 0–4. The controller and adaptive optimizer implementation belongs to [pulseopt](https://github.com/davidkfoss/pulseopt); this repository provides experiment runners, controls, audits, and paper reproduction.

## Reproduce figures and tables

Use Python 3.11 and `uv`. The lockfile fixes the analysis environment; training libraries are optional.

```sh
uv sync --frozen --extra dev
make paper-reproduce
make paper-check
```

The [paper results release](https://github.com/davidkfoss/feedback-utility-experiments/releases/tag/iconip2026-v1) contains the 205 runs underlying the paper. `make paper-reproduce` downloads the pinned ZIP, verifies the archive and every result, and generates the artifacts. If results are already present, it verifies and reuses them. The archive stores every run at its root, including the RunPod controls. Reproduction does not download datasets or model weights and does not run training.

For an offline copy, supply the same verified bundle explicitly:

```sh
make paper-reproduce ARCHIVE=/path/to/iconip2026-results-v1.zip
```

```sh
make paper-artifacts  # existing results, offline
make paper-tables
make paper-plots
make paper-audit
uv run --frozen python -m scripts.paper.reproduce --artifact tab:cifar-ur-fixed
```

Outputs go to `reproduced_artifacts/iconip2026/`: two PDF/PNG figures, nine LaTeX tables, full-precision `analysis.json`, `seed_outcomes.csv`, control audits, and generation metadata. Figure filenames match the submitted paper. Table layouts and captions follow `paper_1215.zip`; numerical cells are generated from unrounded seed outcomes. Two occurrences of one AG News rounding discrepancy are explicitly documented.

## Layout

- `experiments/`: CIFAR-100 and AG News runners; shared NLP utilities.
- `experiments/utils/`: result serialization, runtime metrics, and precomputed CIFAR controls.
- `scripts/paper/`: archive management, explicit result loading, analysis, tables, figures, audits, and training plans.
- `scripts/run_cifar_attribution_controls.py`: existing multi-GPU control launcher.
- `paper/`: run manifest and configurations, artifact registry, release metadata, table templates.
- `tests/reference/`: numerical references captured before refactoring and values from the submitted paper.
- `archived_results/iconip2026/`: downloaded, immutable raw results (ignored by Git).
- `results/`: new training outputs (ignored by Git).

## Rerun training

Training requires a separate, larger environment and suitable compute:

```sh
uv sync --frozen --extra training --extra dev
make train-plan
uv run --frozen --extra training python -m scripts.paper.training \
  --run-id cifar100_sym40_adamw_aees_seed0 --execute
uv run --frozen --extra training python -m scripts.paper.training \
  --run-id agnews_sym20_adamw_aees_seed0 --execute
```

Without `--execute`, the planner only prints configurations and commands. It takes parameters from the released run manifest, including original schedule seeds. New runs use a separate results directory and never overwrite released outputs. Frequency-matched reruns use the four other released AEES traces as donors by default.

CIFAR-100 downloads to `data/cifar100` when the runner first loads it. AG News downloads and tokenizes with Hugging Face; see [training and datasets](docs/training.md) for preparation, split seeds, cache behavior, and multi-GPU controls.

See [artifact mapping and metric definitions](docs/reproduction.md), [provenance](docs/provenance.md), and [scientific discrepancies](docs/scientific_discrepancies.md). The original [thesis repository](https://github.com/davidkfoss/aees-thesis-experiments) retains SST-2 and the broader historical experiments. This repository preserves that Git ancestry but supports only the paper workflows.
