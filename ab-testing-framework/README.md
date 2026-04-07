# A/B Testing Framework for ML Models

A simulation-based experimentation framework for comparing anomaly detection model variants. Computes statistical significance, effect sizes, and power analysis — with an interactive Streamlit dashboard.

## Stack
- **SciPy / Statsmodels** — hypothesis testing, power analysis
- **Scikit-learn** — precision, recall, F1 metrics
- **Streamlit** — interactive dashboard
- **Pandas / Matplotlib** — analysis and visualisation

## Quickstart
```bash
pip install scipy statsmodels scikit-learn streamlit pandas numpy matplotlib

# Run notebook
jupyter notebook ab_testing_framework.ipynb

# Run interactive dashboard
streamlit run dashboard.py
```

## Features
- Proportions Z-test for precision and recall differences
- McNemar's test for paired model comparison
- Cohen's h effect size for proportions
- Power analysis and sample size calculator
- Threshold sweep — F1 vs threshold for both variants
- Full experiment summary report
- Interactive Streamlit dashboard with live sliders

## Dashboard
The Streamlit dashboard lets you adjust sample size, anomaly rate, model precision/recall, and significance level interactively — all tests and charts update in real time.
