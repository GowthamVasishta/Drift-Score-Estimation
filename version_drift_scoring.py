# -*- coding: utf-8 -*-
"""
Enhanced Drift Detection with Outlier Contributors (including categorical)
"""

# Import standard libraries
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance, chi2_contingency, skew, kurtosis
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# PSI (Population Stability Index) calculator for numeric columns
# Measures how values shift between quantile-based buckets
# -----------------------------------
def psi_buckets(expected, actual, buckets=10, eps=1e-6):
    quantiles = np.linspace(0, 100, buckets + 1)
    breakpoints = np.percentile(expected, quantiles)
    breakpoints = np.unique(breakpoints)  # avoid duplicate edges
    if len(breakpoints) <= 2:
        return 0.0  # Not enough distinct bins

    breakpoints[0], breakpoints[-1] = -np.inf, np.inf  # Extend bin range
    expected_bins = pd.cut(expected, bins=breakpoints).value_counts()
    actual_bins = pd.cut(actual, bins=breakpoints).value_counts()
    expected_bins = expected_bins / expected_bins.sum()
    actual_bins = actual_bins / actual_bins.sum()

    # Calculate PSI
    psi = ((expected_bins - actual_bins) * np.log((expected_bins + eps) / (actual_bins + eps))).sum()
    return psi

# -----------------------------------
# Main numeric drift detection function
# Combines statistical distances and distribution shape metrics
# -----------------------------------
def compute_numeric_drift(x1, x2, return_outliers=False):
    x1, x2 = np.array(x1), np.array(x2)
    
    # Use IQR to normalize scale differences
    iqr = max(np.percentile(np.concatenate([x1, x2]), 75) - np.percentile(np.concatenate([x1, x2]), 25), 1e-9)
    
    # Compute multiple statistical distance metrics
    ks = ks_2samp(x1, x2).statistic
    w = np.tanh(wasserstein_distance(x1, x2) / iqr)
    psi = np.tanh(psi_buckets(x1, x2))
    
    # Drift in central tendency and distribution shape
    stat_drift = abs(np.mean(x1) - np.mean(x2)) / (np.std(x1) + 1e-9)
    shape_drift = abs(skew(x1) - skew(x2)) + abs(kurtosis(x1) - kurtosis(x2))
    
    # Average all drift metrics
    drift_score = np.mean([ks, w, psi, np.tanh(stat_drift), np.tanh(shape_drift)])

    if return_outliers:
        # Optional: return top contributors to drift
        med = np.median(x1)
        contrib = abs(x2 - med) / iqr
        contrib_pct = 100 * contrib / (contrib.sum() + 1e-9)
        top_indices = np.argsort(-contrib)[:5]
        outliers = [{"index": int(i), "value": x2[i], "contrib_pct": round(contrib_pct[i], 2)} for i in top_indices]
        return drift_score, outliers

    return drift_score

# -----------------------------------
# Categorical drift detection using Jensen-Shannon & Chi-Square
# -----------------------------------
def compute_categorical_drift(p, q, eps=1e-6, return_outliers=False):
    all_cats = p.index.union(q.index)  # unify all categories
    p = p.reindex(all_cats, fill_value=0) + eps
    q = q.reindex(all_cats, fill_value=0) + eps
    p /= p.sum()
    q /= q.sum()

    js = jensenshannon(p, q)
    
    # Use chi-square test to detect statistical independence
    try:
        chi2, _, _, _ = chi2_contingency([p * 1000, q * 1000])
        chi_score = min(1.0, chi2 / 1000.0)  # Normalize chi2
    except:
        chi_score = 0

    drift_score = 0.5 * js + 0.5 * chi_score

    if return_outliers:
        # Show top category shifts
        change = (q - p).abs()
        change_pct = 100 * change / (change.sum() + 1e-9)
        top_cats = change_pct.sort_values(ascending=False).head(5)
        outliers = [{"category": str(cat), "contrib_pct": round(val, 2)} for cat, val in top_cats.items()]
        return drift_score, outliers

    return drift_score

# -----------------------------------
# Calculates drift in missing values
# -----------------------------------
def nan_drift_ratio(v1_col, v2_col):
    return abs(v1_col.isna().mean() - v2_col.isna().mean())

# -----------------------------------
# Complete drift report generator
# Returns: per-column drift %, overall drift %, and outliers
# -----------------------------------
def generate_drift_report(v1: pd.DataFrame, v2: pd.DataFrame):
    drift_scores = {}
    outliers_report = {}
    
    numeric_cols = [c for c in v1.columns if pd.api.types.is_numeric_dtype(v1[c])]
    categorical_cols = [c for c in v1.columns if c not in numeric_cols]

    # Process numeric columns
    for col in numeric_cols:
        drift, outliers = compute_numeric_drift(v1[col].dropna(), v2[col].dropna(), return_outliers=True)
        nan_drift = nan_drift_ratio(v1[col], v2[col])
        drift_scores[col] = 0.9 * drift + 0.1 * nan_drift
        outliers_report[col] = outliers

    # Process categorical columns
    for col in categorical_cols:
        p = v1[col].value_counts(normalize=True)
        q = v2[col].value_counts(normalize=True)
        drift, outliers = compute_categorical_drift(p, q, return_outliers=True)
        nan_drift = nan_drift_ratio(v1[col], v2[col])
        drift_scores[col] = 0.9 * drift + 0.1 * nan_drift
        outliers_report[col] = outliers

    # Compute overall drift score using weighted average
    total = sum(drift_scores.values()) + 1e-12
    weights = {k: drift_scores[k] / total for k in drift_scores}
    drift_vector = np.array([drift_scores[k] for k in drift_scores])
    weight_vector = np.array([weights[k] for k in drift_scores])
    overall_drift = round(100 * (drift_vector @ weight_vector) / weight_vector.sum(), 2)

    # Format per-column drift as %
    drift_pct = {k: round(100 * v, 2) for k, v in drift_scores.items()}
    return drift_pct, overall_drift, outliers_report

# -----------------------------------
# Plot per-column drift + overall drift as bar chart
# -----------------------------------
def plot_drift(drift_dict, overall_drift):
    plt.figure(figsize=(10, 5))
    cols, values = list(drift_dict.keys()), list(drift_dict.values())
    sns.barplot(x=cols, y=values, palette='magma')
    plt.axhline(overall_drift, color='blue', linestyle='--', label='Overall Drift')
    plt.title('Enhanced Drift Detection with Top Contributors')
    plt.ylabel('Drift (%)')
    plt.xticks(rotation=45)
    for i, v in enumerate(values):
        plt.text(i, v + 1, f"{v}%", ha='center')
    plt.text(-0.4, overall_drift + 1, f"Overall {overall_drift}%", color='blue')
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------------
# Example usage: test with toy data
# -----------------------------------
if __name__ == "__main__":
    v1 = pd.DataFrame({
        'sales': [100, 120, 130, 140, 150, 160, 155, 148],
        'margin': [0.25, 0.26, 0.24, 0.27, 0.28, 0.29, 0.26, 0.30],
        'city': ['NY', 'SF', 'LA', 'NY', 'SF', 'LA', 'NY', 'LA']
    })

    v2 = pd.DataFrame({
        'sales': [100, 120, 130, 140, 150, 160, 155, 148, 152, 300, 100],
        'margin': [0.25, 0.26, 0.24, 0.27, 0.28, 0.29, 0.26, 0.30, 0.10, 0.99, 0.85],
        'city': ['NY', 'SF', 'LA', 'NY', 'SF', 'LA', 'NY', 'LA', 'KA', 'RDE', 'RTE']
    })

    # Generate drift report and print results
    drift_scores, overall_drift, outliers = generate_drift_report(v1, v2)
    print("📊 Drift Scores (%):", drift_scores)
    print("🔥 Overall Drift (%):", overall_drift)

    print("\n🔍 Top Outlier Contributors by Column:")
    for col, outlier_info in outliers.items():
        print(f"\n{col}:")
        for o in outlier_info:
            print(f"  {o}")
    
    # Visualize the drift
    plot_drift(drift_scores, overall_drift)
