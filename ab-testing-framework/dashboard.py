"""
A/B Testing Framework — Streamlit Dashboard
Run: streamlit run dashboard.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import TTestIndPower
from sklearn.metrics import precision_score, recall_score, f1_score

st.set_page_config(page_title="A/B Testing Framework", layout="wide")
st.title("A/B Testing Framework for Anomaly Detection Models")
st.caption("Compare two model variants with statistical rigour — precision, recall, F1, significance tests, and power analysis.")

# ── Sidebar Controls ──────────────────────────────────────────
st.sidebar.header("Experiment Settings")
n_samples        = st.sidebar.slider("Sample Size", 200, 5000, 1000, step=100)
anomaly_rate     = st.sidebar.slider("True Anomaly Rate", 0.02, 0.30, 0.08, step=0.01)
alpha            = st.sidebar.slider("Significance Level (α)", 0.01, 0.10, 0.05, step=0.01)
control_prec     = st.sidebar.slider("Control Precision",   0.50, 0.99, 0.72, step=0.01)
control_rec      = st.sidebar.slider("Control Recall",      0.50, 0.99, 0.80, step=0.01)
treat_prec       = st.sidebar.slider("Treatment Precision", 0.50, 0.99, 0.83, step=0.01)
treat_rec        = st.sidebar.slider("Treatment Recall",    0.50, 0.99, 0.85, step=0.01)
seed             = st.sidebar.number_input("Random Seed", value=42)

# ── Data Simulation ───────────────────────────────────────────
np.random.seed(int(seed))
gt = (np.random.rand(n_samples) < anomaly_rate).astype(int)
pos_idx = np.where(gt == 1)[0]
neg_idx = np.where(gt == 0)[0]

def make_preds(prec, rec, gt, pos_idx, neg_idx):
    preds = np.zeros(len(gt), dtype=int)
    tp = int(len(pos_idx) * rec)
    preds[np.random.choice(pos_idx, min(tp, len(pos_idx)), replace=False)] = 1
    fp = int(tp / prec - tp) if prec > 0 else 0
    fp = min(fp, len(neg_idx))
    if fp > 0:
        preds[np.random.choice(neg_idx, fp, replace=False)] = 1
    return preds

preds_A = make_preds(control_prec, control_rec, gt, pos_idx, neg_idx)
preds_B = make_preds(treat_prec,   treat_rec,   gt, pos_idx, neg_idx)

def metrics(gt, preds, name):
    return {
        "Model":     name,
        "Precision": precision_score(gt, preds, zero_division=0),
        "Recall":    recall_score(gt, preds, zero_division=0),
        "F1":        f1_score(gt, preds, zero_division=0),
        "FP Rate":   ((preds==1)&(gt==0)).sum() / max((gt==0).sum(), 1),
        "FN Rate":   ((preds==0)&(gt==1)).sum() / max((gt==1).sum(), 1),
    }

mA = metrics(gt, preds_A, "Control (A)")
mB = metrics(gt, preds_B, "Treatment (B)")
df_metrics = pd.DataFrame([mA, mB]).set_index("Model")

# ── Metrics ───────────────────────────────────────────────────
st.subheader("Model Performance")
col1, col2, col3, col4 = st.columns(4)
for col, metric in zip([col1, col2, col3, col4], ["Precision", "Recall", "F1", "FP Rate"]):
    delta = mB[metric] - mA[metric]
    col.metric(f"{metric} — Treatment vs Control", f"{mB[metric]:.4f}", f"{delta:+.4f}")

st.dataframe(df_metrics.style.format("{:.4f}").highlight_max(axis=0, color="#d4edda"))

# ── Significance Tests ────────────────────────────────────────
st.subheader("Statistical Significance")
tp_A, tp_B = ((preds_A==1)&(gt==1)).sum(), ((preds_B==1)&(gt==1)).sum()
n_pos = gt.sum()

rows = []
if preds_A.sum() > 0 and preds_B.sum() > 0:
    z, p = proportions_ztest([tp_A, tp_B], [preds_A.sum(), preds_B.sum()])
    rows.append({"Test": "Precision Z-test", "Statistic": round(z,4), "p-value": round(p,4), "Significant": p < alpha})
z, p = proportions_ztest([tp_A, tp_B], [n_pos, n_pos])
rows.append({"Test": "Recall Z-test", "Statistic": round(z,4), "p-value": round(p,4), "Significant": p < alpha})
b = ((preds_A==1)&(preds_B==0)).sum(); c = ((preds_A==0)&(preds_B==1)).sum()
if b+c > 0:
    chi2 = (abs(b-c)-1)**2/(b+c); pmc = 1 - stats.chi2.cdf(chi2, df=1)
    rows.append({"Test": "McNemar Test", "Statistic": round(chi2,4), "p-value": round(pmc,4), "Significant": pmc < alpha})

sig_df = pd.DataFrame(rows).set_index("Test")
st.dataframe(sig_df.style.applymap(lambda v: "background-color: #d4edda" if v is True else ("background-color: #f8d7da" if v is False else ""), subset=["Significant"]))

# ── Power Analysis ────────────────────────────────────────────
st.subheader("Power Analysis")
h = 2*(np.arcsin(np.sqrt(mB["Precision"])) - np.arcsin(np.sqrt(mA["Precision"])))
pa = TTestIndPower()
req_n = pa.solve_power(effect_size=abs(h)+1e-6, alpha=alpha, power=0.80)
c1, c2 = st.columns(2)
c1.metric("Cohen's h (precision effect size)", f"{h:.4f}")
c2.metric("Min sample size per group (power=0.80)", f"{int(np.ceil(req_n))}")

sizes = np.arange(50, 3000, 50)
powers = [pa.solve_power(effect_size=abs(h)+1e-6, alpha=alpha, nobs1=n) for n in sizes]
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(sizes, powers, linewidth=2)
ax.axhline(0.80, color="red", linestyle="--", label="Power = 0.80")
ax.axvline(req_n, color="green", linestyle="--", label=f"Required n = {int(req_n)}")
ax.set_xlabel("Sample Size per Group"); ax.set_ylabel("Power"); ax.legend()
ax.set_title("Power Curve"); plt.tight_layout()
st.pyplot(fig)

st.caption("Built by Sachin M D — A/B Testing Framework for ML Anomaly Detection")
