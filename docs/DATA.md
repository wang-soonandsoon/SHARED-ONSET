# Data and released artifacts

`python scripts/fetch_data.py` obtains these original repositories at the versions used for the paper:

| Source | Version | Role |
|---|---|---|
| [POP909](https://github.com/music-x-lab/POP909-Dataset) | `d83e6edba6872a704f5d3b8b32f5cb540088dae6` | Primary work IDs and original MIDI corpus |
| [Hierarchical structure annotations](https://github.com/Dsqvival/hierarchical-structure-analysis) | `3a8d0d096e8dd2f62e38e48500bf0b230eeca585` | Manually aligned melody, chord, and tempo annotations |
| [POP909-CL](https://github.com/AndyWeasley2004/POP909-CL-Dataset) | `be9094392903c471a930519e1c0bacf8b6be5d62` | Curated MIDI meter metadata |

Upstream notices are preserved in `docs/third_party/`. The raw corpus is downloaded from its publishers rather than duplicated in this repository. The adapter reads `raw/pop909`, `raw/pop909_structure`, and `raw/pop909_cl` under the data root. The fetch script creates these aliases automatically. If you already have those directories, skip fetching and pass your data root to the preparation command.

## Included artifacts

- `assets/bar16_inference.pt`: the trained denoiser's original model tensors and configuration; no optimizer or local data paths.
- `assets/synthetic_inputs.zip`: 60 controlled probability/constraint cases.
- `assets/real_inputs.zip`: 24 learned-probability requests, with visible-context constraints and work identifiers.
- `assets/long_inputs.zip`: 16 narrow/wide targets from eight additional learned requests.
- `assets/recorded_runs.zip`: all 920 recorded rows, plans, run configurations, status records, and model-extraction metadata.
- `paper_results/data/`: numeric observations and summary values used by the paper's plotting script.

Real-case visible tokens and anchors derive from the aligned POP909 sources above. Model q tables and generated feasible witnesses are not ground-truth completions. No raw audio, original full MIDI collection, participant identities, private listening keys, or manuscript drafts are included. The upstream notices describe the corresponding source materials; they do not imply ownership of the underlying musical compositions.
