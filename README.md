# PI-FDM

Physics-informed frozen diffusion models for radiation-diffusion problems.

This repository contains the core implementation and three benchmark cases:

- `case_studies/radiative_1d` — 1D single-material radiative diffusion
- `case_studies/radiative_2d` — 2D two-material radiative diffusion
- `case_studies/radiative_initial_2d` — 2D three-material point explosion

## Installation

```bash
python -m pip install -e ".[ml,data]"
```

## Run

```bash
python case_studies/radiative_1d/generate.py
python case_studies/radiative_1d/train.py
python case_studies/radiative_1d/test.py
```

Each case provides `generate.py`, `train.py`, `test.py`, `generate_generalization.py`, and `test_generalization.py`. Training and evaluation settings are defined at the top of the corresponding files.