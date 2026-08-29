# Inverse Optimization for Shortest-Path Decisions

A research sandbox for **inverse linear optimization**: observe optimal decisions and infer
objective parameters that rationalize them.

This repository uses shortest-path demonstrations on a DAG. The forward model minimizes edge
cost. The inverse model solves a linear program for positive edge costs satisfying max-margin
preference inequalities between demonstrated and alternative paths, with L1 regularization
toward a prior and a normalization constraint for identifiability.

## Run

```bash
pip install -e ".[dev]"
python scripts/demo.py
pytest
```

## Research scope

The recovered objective need not equal the latent ground-truth cost vector; inverse problems
can be non-identifiable. The primary criterion here is **decision rationalization**:
demonstrated paths should be optimal under the inferred objective.

Grounded in the modern inverse-optimization framework surveyed by Chan, Mahmood, and Zhu,
*Operations Research* (2023/2025 volume publication).

## License

PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.
