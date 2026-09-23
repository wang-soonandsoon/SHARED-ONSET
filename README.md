# SHARED-ONSET

Core implementation for **Exact Joint Sampling for Shared-Onset Music Infilling**.

**Full code coming soon.**

This preview includes the onset-reset joint sampler, a synthetic example, and
correctness tests. It supports the base target with zero internal motion cost;
the full motion-weighted sampler and experimental pipeline will follow.

## Quick start

Requires Python 3.11+. No GPU, dataset, or checkpoint is needed.

```bash
git clone https://github.com/wang-soonandsoon/SHARED-ONSET.git
cd SHARED-ONSET
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/minimal.py
python -m pytest -q
```
