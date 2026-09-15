# Training and dataset preparation

`paper/experiments.json` records every selected run's original path, checksum, paired seed, policy, and training configuration. The planner reads these explicit settings; it does not infer policies from result filenames. It covers 180 CIFAR-100 runs and 25 AG News runs.

Install `uv sync --frozen --extra training --extra dev`. Training chooses CUDA when available, otherwise CPU; full CIFAR/DistilBERT experiments are expensive. The analysis-only installation excludes Torch, Transformers, Datasets, and pulseopt.

## CIFAR-100

The runner downloads the dataset to `data/cifar100`. It adapts ResNet-18 with a 3×3 stride-1 input convolution and no initial max pooling. Each run uses 200 epochs, batch size 128, and 78,200 optimizer steps. AdamW uses base LR 0.001; SGD uses LR 0.1 and momentum 0.9. Both use weight decay 0.0001 and a constant base schedule.

The paired training seed also determines label corruption. Symmetric corruption samples an incorrect class; asymmetric corruption cycles within superclasses. Do not change the order of seeding, loader construction, model initialization, augmentation, training, or diagnostic evaluation during structural cleanup.

```sh
uv run --frozen --extra training python -m scripts.paper.training --dataset cifar100
uv run --frozen --extra training python -m scripts.paper.training \
  --run-id cifar100_sym40_sgd_frequency_matched_seed0 --execute
```

FM reruns reconstruct schedules using the other four released AEES traces. A temporary path adapter exposes the flat archive to the unchanged trace validator. Original schedule seeds are explicitly supplied, so renaming archive files cannot change schedules. To study newly trained donors, use the existing launcher with a complete compatible donor directory and keep that new study separate from the frozen paper references.

The existing multi-GPU launcher remains available:

```sh
uv run --frozen --extra training python -m scripts.run_cifar_attribution_controls \
  --aees-trace-root /path/to/completed/cifar_noisy \
  --output-root results/new_controls --gpus 0,1,2,3,4 --dry-run
```

That older launcher's dry run writes a run-plan manifest but starts no training. The paper planner's default prints its plan without writing outputs.

## AG News

The runner uses `distilbert-base-uncased`, dynamic padding, maximum length 128, a 90/10 training/validation split with seed 42, and 20% symmetric training-label noise with seed 42. Paired model/training seeds are 0–4. Paper runs use five epochs, batch size 16, LR 0.00005, weight decay 0.01, and `warmup_linear` with zero warmup. Noise choices are 0, 0.005, and 0.01. Noise-managed runs contain 169 episodes; the final episode has 150 steps.

Prepare the paper's noisy tokenized dataset once, with network access:

```sh
uv run --frozen --extra training python -m experiments.task_agnews \
  --label-noise-rate 0.2 --label-noise-seed 42 --max-length 128 --pretokenize-only
```

Tokenized datasets live under `data/agnews_tokenized`; model/tokenizer caches live under `.hf_cache`. Preparing tokens does not necessarily download all model weights. On an offline training host, copy both prepared data and a complete model/tokenizer cache. Keep the cache's split, corruption seed, and maximum-length settings consistent; see the documented cache limitation.

The AG News runner imports common training functions from `experiments/nlp_common.py`. The extraction preserves all 27 function bodies; tests compare their syntax-tree hashes to the original implementation. SST-2 has no runner, configuration, result, or artifact in the paper checkout.

`pulseopt` owns discounted UCB, episodic reward calculation, optimizer wrappers, and gradient-noise RNGs. This repository owns datasets, loops, schedule construction/replay, result records, and reporting. Refactoring must preserve that boundary.
