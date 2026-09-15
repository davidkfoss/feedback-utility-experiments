# Provenance and release scope

This repository derives from `davidkfoss/aees-thesis-experiments` at commit `fcc5ff9` and preserves its Git history. A separate import commit captures the local paper figure refinements (11-episode smoothing, line styles, and control marker shapes) before the paper-only cleanup. Historical thesis workflows remain in the original repository and Git ancestry, rather than in the supported checkout.

The old thesis results release `v0.1.1` remains unchanged. Its asset is `archived_results_aees_thesis_iconip2026_v0.1.1.tar.gz`, SHA-256 `6ac50f698c5347f84b33c291aedaac630e0b505501f8ec8e941ba16f489541ea`. The paper bundle is a separate ZIP assembled from local archived results, with every source file verified against `paper/experiments.json`.

| Source directory in the thesis archive | Selected files | Purpose |
|---|---:|---|
| `cifar_noisy` | 120 | AEES and all three fixed multipliers |
| `aees-runpod-results` | 60 | Uniform-random and frequency-matched CIFAR controls |
| `noisy_agnews` | 10 | Neutral and AEES noise-only AG News runs |
| `nlp_noise_ablation` | 15 | Two fixed-noise and uniform-random AG News controls |

The five AG News AEES files duplicated under `nlp_noise_ablation` are byte-identical to the selected `noisy_agnews` files. The manifest records those aliases. AEES-Dual runs, SST-2, clean-data experiments, compute studies, checkpointing studies, and exploratory variants are outside the paper release.

All 205 raw files are copied byte-for-byte. Their original embedded output paths remain intact as provenance; loaders do not rely on them. The ZIP has no enclosing directory and no nested RunPod folder or Git metadata. Unique filenames include dataset, noise regime, optimizer, policy, and seed. Original schedule RNG seeds are retained separately from these new filenames.

`tests/reference/legacy_analysis.json` was captured using the original fixed-control, open-loop, AG News, and arm-selection analysis functions before replacing the paper loading layer. It contains full-precision paired effects and Figure 1 mean/SD arrays. Table references and numeric cells come from the user-provided `paper_1215.zip`; its checksum is recorded in `paper/artifacts.json`. These references are fixed regression targets, not data used to calculate reported results.

## Package a release

```sh
make paper-bundle SOURCE_ROOT=/path/to/aees-thesis-experiments/archived_results
```

This creates `dist/iconip2026-results-v1.zip` and its checksum sidecar. Packaging checks every source hash, fixes ZIP timestamps, sorts entries, and does not alter source files. Repeat builds from identical inputs produce the same archive bytes in the tested Python environment.

The bundle is prepared for a new paper repository release. Publication is a separate step: create the GitHub repository, push the reviewed code, create a versioned release, attach the ZIP and checksum, and set `paper/release.json` to the actual immutable asset URL. Do not replace the thesis release or publish a guessed download URL. A supplied local bundle or `RESULTS_URL` override can exercise the complete fetch-and-reproduce workflow before publication; both must match the pinned archive hash.
