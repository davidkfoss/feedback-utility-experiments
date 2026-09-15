# Paper reproduction

Every artifact is registered in `paper/artifacts.json`, with exact run IDs and a generation command. Those IDs resolve to source paths, checksums, and recorded configurations in `paper/experiments.json`. No run is selected by recursive directory traversal or heuristic filename matching.

| Paper label | Generated file | Inputs |
|---|---|---:|
| `tab:audit-controls` | `tables/control_questions.tex` | 0 runs |
| `tab:cifar-fixed` | `tables/cifar_fixed.tex` | 120 runs |
| `fig:trajectory-actions` | `figures/arm_selection_cifar_sym40_optimizer_paired.pdf` | 10 runs |
| `tab:cifar-open-loop` | `tables/cifar_open_loop.tex` | 90 runs |
| `fig:cifar-control-effects` | `figures/cifar_aees_vs_control_effect_serif.pdf` | 180 runs |
| `tab:agnews-noise` | `tables/agnews_noise.tex` | 25 runs |
| `tab:appendix-all-fixed` | `tables/appendix/cifar_all_fixed.tex` | 120 runs |
| `tab:appendix-cifar-controller` | `tables/appendix/cifar_controller_diagnostics.tex` | 30 runs |
| `tab:cifar-ur-fixed` | `tables/appendix/cifar_ur_fixed.tex` | 120 runs |
| `tab:appendix-fm-ranges` | `tables/appendix/cifar_fm_ranges.tex` | 60 runs |
| `tab:appendix-agnews-controls` | `tables/appendix/agnews_controls.tex` | 25 runs |

All files are relative to `reproduced_artifacts/iconip2026/`. The uploaded paper embeds most tables directly in `main.tex`; replace those environments with the generated table files or corresponding `\input{...}` statements. The two existing appendix inputs and figure filenames retain their names. This repository does not edit the author's manuscript automatically.

## Metrics and inference

`analysis.json` is the common source for both tables and figures. Accuracies remain fractions at the seed level; summaries convert to percent and effects to percentage points. Means and sample SDs use the existing conventions. Peak accuracy is the maximum logged validation accuracy, final accuracy is the last logged value, drop is peak minus final, and peak epoch is the earliest maximizer (one-based). Rounding happens only when rendering tables.

Best fixed is selected separately for each dataset/noise/optimizer cell by the highest mean peak validation accuracy across all five paired seeds. Ties retain candidate order. It is selected once and held fixed in all bootstrap draws. Uniform-random versus best-fixed differences are calculated directly from unrounded paired seed outcomes, never by subtracting rounded table means or separate AEES contrasts.

The existing paired statistics function uses NumPy's dedicated `default_rng(0)`, 10,000 resamples of the five paired differences, and 2.5/97.5 percentiles. Positive paired differences count as wins; ties do not. For AG News, the JSON additionally records paired effects without introducing another manuscript table.

Controller shares pool executed episode actions across seeds. Median clipped rewards pool the selected action's episode rewards. They describe the observed feedback signal, not causal arm effects. AG News diagnostics use noise-only AEES, not AEES-Dual. The AG News action shares and reward medians quoted in the appendix prose are also in `analysis.json`.

Figure 1 smooths each seed's binary arm indicators with a centered 11-episode moving average, shortening windows at boundaries. It plots the seed mean and sample SD, clipping display bands to [0,1]. It uses solid, dashed, and dotted lines for multipliers 0.5, 1, and 2. Figure 2 uses circle, square, and triangle markers for best fixed, uniform random, and frequency matched. Smoothing affects presentation only.

## Validation without training

`make paper-check` first verifies the 205 raw-result hashes, then tests all displayed table values against the submitted paper (with the two documented AG News rounding differences), full-precision effects and curve/band arrays against the original analysis, and control reconstruction against every released schedule. It also checks the manifest, flat archive extraction, and mechanically preserved NLP function bodies. Optional training-environment tests parse every paper configuration without loading data or model weights. A five-step CPU example additionally compares parameter updates, training metrics, complete/partial episode logs, and RNG states with the original NLP functions for neutral, AEES, uniform-random, and fixed-noise policies.

`make paper-audit` validates all 30 AEES donor traces and reconstructs all 60 CIFAR control schedules using the unchanged control implementation. It checks donor exclusion, probabilities, Hamilton counts, original schedule seeds, planned and executed action indices/values, contiguous boundaries, and the 391-episode horizon. Python's global RNG must remain unchanged. Hamilton ties use candidate order; UR draws remain independent rather than forcibly balanced. The original multi-GPU launcher tests remain in place.

The regression suite checks numerical content, not PDF file hashes: PDF metadata and font rendering can vary. Figures also export PNG previews for visual inspection. Reproduction records its manifest hash, package versions, Git commit, and source-file hashes in `generation.json`.
