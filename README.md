# SHARED-ONSET

Full implementation of **Exact Joint Sampling for Shared-Onset Music Infilling**: onset-time decomposition, exact residual rejection, standard baselines, model training, and paper experiments.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,research]'
python examples/complete.py
python scripts/reproduce.py smoke
pytest -q
```

Pretrained weights, frozen experiment inputs, and all 920 recorded experiment rows are included. Raw POP909 data is obtained separately from its original repositories.

[Reproduce the paper](docs/REPRODUCING.md) · [Method and API](docs/METHOD.md) · [Data sources](docs/DATA.md)
