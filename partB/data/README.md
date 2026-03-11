# Data README

This folder contains the datasets used in Part B.

- `main_moons_dataset.csv`: synthetic binary classification dataset generated with `sklearn.datasets.make_moons` using a fixed random seed. It is used in Question 2 and Question 3.1.
- `failure_linear_dataset.csv`: synthetic binary classification dataset generated with `sklearn.datasets.make_classification` using a fixed random seed. It is used in Question 3.2.

How the datasets were obtained:

- Both datasets are generated locally in `submission_support/build_submission.py`.
- No manual download, external API, or undocumented preprocessing step is required.

How the datasets are used:

- The CSV files are loaded directly by the notebooks.
- Features are standardized inside the modeling pipeline.
- Labels are binary and stored in the `label` column.
