# Reproducing SHARED-ONSET

Run commands from the repository root. Python 3.11 and Linux are recommended; benchmark workers use Linux CPU affinity and process memory limits. The original measurements used Python 3.11, single-threaded CPU inference, and GPU model evaluation. Wall times depend on the machine.

## 1. Install and run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,research]'
python examples/minimal.py
python examples/complete.py --samples 4
python scripts/reproduce.py smoke
pytest -q
```

The zero-motion example prints shared rhythms. The full-target example adds motion cost `beta=0.02`, uses Visible + Early rejection, verifies every completion, and writes MIDI files and sampling diagnostics to `runs/example/`. The smoke run checks three backends on the same released learned R=3 request with two draws each. It is a functional check, not a new paper result.

For inference without PyTorch or plotting, `pip install -e '.[dev]'` is sufficient. Neural training/tests require `.[research]`. The `shared_onset` public imports from the earlier release remain available; `tri` is the internal research package name. `scripts/python.sh` uses the active interpreter (`PYTHON` can override it).

## 2. Re-run the paper's fixed-input comparisons

The archives contain the original full-vocabulary probability tables, hard constraints, fixed observations, seeds, and input metadata. They are extracted under the chosen output directory. Source-machine path strings were replaced with portable paths; the numeric inputs are unchanged. No model or data download is needed for these comparisons.

```bash
python scripts/reproduce.py synthetic --cpu 4
python scripts/reproduce.py real --cpu 4
python scripts/reproduce.py real_proposal --cpu 4
python scripts/reproduce.py long --cpu 4
python scripts/reproduce.py report
```

Choose an available CPU with `--cpu`; omitted, it uses the first CPU in the launch process's affinity. Run stages serially on an idle machine. Use `--plan-only` to inspect work without running solvers; `--resume` continues an interrupted run with unchanged inputs and configuration. Choose a fresh `--out` for a different protocol. The same frozen probabilities and draw seeds are used across methods. Failures and budget limits remain in the results.

| Stage | Fixed targets | Recorded rows | Protocol |
|---|---:|---:|---|
| synthetic | 60 | 480 | L=16, D=16, K=4; R=2,3,4,6,8; beta=0,0.02; unknown/partial/known |
| real | 24 from 24 works | 168 | L=16, K=4, R=3,4,6,8; 32 independent draws |
| real_proposal | same 24 | 144 | Drop, Boundary, Visible, early rejection, and product baseline |
| long | 8 requests, 16 targets | 128 | L=32, K=8, R=3,4; narrow/wide pitch domains; 32 draws |

Settings and budgets are in `configs/shared_rhythm_extension.yaml` and `configs/shared_rhythm_proposals.yaml`. The synthetic rows comprise two timing repeats; real rows use independent draws with one engine per target. Target partition comparisons apply only to backends that compute the same normalizer. The zero-motion `onset_reset` backend is unsupported for motion-weighted targets and is retained as such in the protocol. A timed-out baseline is not converted into a speedup.

## 3. Inspect the original measurements and rebuild plots

```bash
python scripts/reproduce.py recorded
python paper_results/build_figures.py
```

The first command unpacks all 920 original rows, including failures and per-draw diagnostics, into `runs/reproduction/recorded/` and reconstructs a report. These are recorded measurements, not newly timed runs. The second command regenerates the submitted paper's experiment plots from its archived CSV observations, with internal checks against `figure_metrics.json`; output is in `paper_results/figures/`. The plot inputs and full recorded rows are both supplied, so success-only plot views can be checked against all attempted cases.

Model computation is measured when q is frozen and reported separately from discrete inference. The included inference checkpoint is the best validation checkpoint at step 3000 (validation cross-entropy 1.3381165294); its model configuration and tensors are unchanged. Optimizer and RNG states are omitted because this checkpoint is for inference. Fresh training produces resumable `best.pt` and `last.pt`.

## 4. Prepare data and train from scratch

```bash
python scripts/fetch_data.py
python scripts/paper_pipeline.py prepare-data
python scripts/paper_pipeline.py train --device cuda:0
```

`fetch_data.py` fetches the pinned original POP909, aligned structure annotations, and curated chord/meter sources; see [DATA.md](DATA.md). Data defaults to `data/`. Set `SHARED_ONSET_DATA_ROOT` or pass `--data-root` to use another location. Existing source files are not overwritten.

The adapter uses the released manually aligned symbolic timeline: 16 cells per bar, 16-bar contexts, explicit 4/4 meter, aligned chord features, at most 12 windows per work, and a work-level split established before filtering. Only train/validation windows are materialized. Whole-window activity filtering is part of data preparation; later request selection uses visible conditions. Original expressive MIDI timing is not this grid timeline.

`configs/paper_training.yaml` specifies the paper's 2-layer, 128-hidden, 4-head denoiser, 38 chord features, 3000 steps, batch size 16, and fixed 32-cell training gaps. Training is deterministic under the same hardware/software; different runtimes need not reproduce bitwise identical q. Use `--resume` to continue your own unchanged run. The paper's frozen q remains available for exact solver comparisons independently of retraining.

## 5. Select requests and regenerate model probabilities

```bash
python scripts/paper_pipeline.py requests --cohort real
python scripts/paper_pipeline.py freeze --cohort real --device cuda:0
python scripts/paper_pipeline.py requests --cohort long
python scripts/paper_pipeline.py freeze --cohort long --device cuda:0
```

These commands default to `assets/bar16_inference.pt`. Add `--checkpoint runs/fresh/train/best.pt` to both `requests` commands to evaluate a newly trained checkpoint. Selection completes before model loading. The frozen output goes to `runs/fresh/real_inputs/` or `long_inputs/`; the long command additionally creates `long_scale_inputs/`, pairing each of the eight q tables with narrow and wide domains. CUDA extraction reports model load and provider times; CPU-only users can run all supplied frozen-q experiments.

To benchmark newly frozen inputs, copy the relevant experiment YAML, change its stage `manifest` to the new `cases.json`, set `common.cpu`, and run:

```bash
python -m tri.evaluation.shared_rhythm_extension --config your_config.yaml --stage real --out runs/fresh/real_sampling
```

Use stage `real_proposal` or `long` for the proposal config. Keep these results in a new run directory instead of replacing the archived paper results.

## 6. Listen to a completion

After extracting inputs, supply a case JSON (not the manifest) to:

```bash
python examples/complete.py --case runs/reproduction/inputs/real/real_bar16_w666_c768_R3_L16_unknown.json --samples 4
```

The exported MIDI contains the full 256-cell symbolic context, retaining its observed notes and filling the requested spans. Open it in a MIDI player or DAW; a SoundFont synthesizer is needed to render WAV audio. This is a monophonic grid rendering, not reconstruction of the original accompaniment or expressive timing. No human-listening scores are claimed.
