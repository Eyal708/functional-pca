# Contributing

Thanks for your interest in improving `functional-pca`. Bug reports, questions, documentation fixes and pull requests are all welcome.

## Reporting a problem

Please open an issue with:

- a small, self-contained example that reproduces it (simulated data is ideal),
- what you expected and what happened instead (full traceback for errors),
- your Python version and the versions of `numpy`, `scipy`, `scikit-learn`, `pandas` and `matplotlib`.

For numerical questions ("the eigenfunctions look wrong"), it helps to include the observation design: number of curves, observations per curve, and whether the times are shared or irregular.

## Development setup

```bash
git clone https://github.com/Eyal708/functional-pca.git
cd functional-pca
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,examples]"
python -m pytest            # run the test suite
ruff check src tests        # lint
```

## Pull requests

- Keep each pull request focused on one change.
- Add or update tests in `tests/` for any behaviour change. For numerical changes, include a simulated example with a known answer (true eigenfunctions, scores or variance) and state the tolerance you expect.
- Keep the public API (`PACE`, `TwoLevelPACE`, `MFPCA`, `fpca.plotting`, `fpca.utils`, `fpca.stats`) backward compatible where possible, and update the README, docstrings and `CHANGELOG.md` when user-facing behaviour changes.
- Docstrings follow the NumPy style. Cite the paper and section an estimation step implements.
- Do not commit datasets, credentials or project-specific pipelines. Examples should simulate data or download a public dataset.
