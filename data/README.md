# Data

This folder is empty on purpose. The thesis experiments were run on proprietary
company data that cannot be redistributed, and everything in `data/` except
this file is ignored by git.

To run the experiments, put your own CSVs here. The scripts look for these
default names (each can also be passed as a command-line argument):

| File | Used by |
|---|---|
| `20k_sample_data.csv` | sweeps, benchmarks, anomaly recovery, Sun characterisation |
| `scalability_slim_1M.csv` | streaming scalability, coreset sweep, convergence check |
| `scalability_slim_10M.csv` | scalability comparison against the baselines |
| `200k_wide_sample.csv` | dimensionality experiment (any wide numeric table) |

See the main README for the columns each file needs.
