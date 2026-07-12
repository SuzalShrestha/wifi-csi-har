# Ablation configs (defense-required, Phase 4)

Committed run definitions for the three ablations from IMPLEMENTATION_PLAN.md:

- `ablation_receivers.json` — 1 vs 2 vs 3 receivers (all 7 subsets of rx0–rx2)
- `ablation_window.json` — observation window length: 1.0 / 1.5 / 2.0 / 3.0 s
  (center-crop; capped at the assembled window)
- `ablation_rate.json` — packet rate: 100 / 50 / 25 Hz (time decimation)

All three: model `cnn`, splits `random` + `cross-session`, seeds `[0, 1, 2]`,
30 epochs. Cross-subject (LOSO) runs are added once the real dataset has
enough subjects: pass `--split cross-subject` on the command line.

## Running

```bash
.venv/bin/python -m csihar.experiments \
    --config experiments/configs/ablation_receivers.json \
    --data datasets/assembled/<dataset>.npz
```

The dataset path is deliberately NOT in the configs — it is machine-specific.
Any CLI flag overrides the corresponding config value (e.g. `--epochs 2
--seeds 0 --variants rx0 50Hz` for a quick smoke run).

Each run appends a provenance row (git SHA, seed, full TrainConfig JSON,
variant in the `notes` column) to `experiments/results/ablation_<name>.csv`;
checkpoints and confusion-matrix figures go to per-variant subdirectories
under `experiments/checkpoints/ablations/` and `docs/figures/ablations/`.

## Where the real runs happen

This machine has no GPU. The real-data runs (30 epochs x 3 seeds x 2 splits x
up to 7 variants) happen on **Colab/Kaggle**: clone the repo, upload the
assembled `.npz` from Drive, run the commands above, then commit the results
CSVs back under `experiments/results/`. Keep the configs here as the single
source of truth for what was run — edit the JSON, not ad-hoc flags, for
anything you intend to report at the defense.
