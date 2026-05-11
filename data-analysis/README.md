# Data Analysis — Zoo Dataset

End-to-end exploratory and statistical analysis of the UCI Zoo dataset (101 animals, 16 features, 7 classes) using Python's scientific stack. The notebook (`zoo_analysis.ipynb`) goes from raw CSV inspection to dimensionality reduction, walking through the full data-analysis lifecycle.

## What this project demonstrates

- **Data quality auditing**: schema inspection, missing-value checks, outlier diagnostics with boxplots, and categorical/binary attribute profiling.
- **Descriptive statistics & visualization**: count plots, donut charts, histograms, proportion bars, and stacked bar charts segmented by class — built with `matplotlib` and `seaborn`.
- **Correlation analysis tailored to data type**: implemented the **phi coefficient** for binary–binary correlations (via chi-square) in addition to the standard Pearson heatmap, because phi is the statistically appropriate measure for dichotomous variables.
- **Inferential statistics**: hypothesis testing chosen to match the question being asked:
  - **Two-proportion Z-test** comparing aquatic prevalence between mammals and birds.
  - **Fisher's exact test** for the backbone × class association (small-sample categorical).
  - **One-way ANOVA** on `legs` across the seven classes.
  - **Chi-square test of independence** between `milk` and `class_type`.
  Each test includes interpretation of the statistic, p-value, and biological plausibility of the result.
- **Dimensionality reduction with PCA**: standardized numerical features, fit a 5-component PCA, examined explained variance (83% cumulative), inspected component loadings, ranked the most informative attributes, and visualized class separation in the PC1–PC2 plane.

## Tech stack

`pandas` · `numpy` · `scipy.stats` · `statsmodels` · `scikit-learn` (PCA, StandardScaler) · `matplotlib` · `seaborn`

## Key takeaways

- The analysis pairs **statistical rigor** (matching test to data type and assumptions) with **clear visual communication**.
- PCA loadings recover biologically meaningful structure: `legs`, `predator`, `tail`, `aquatic`, and `backbone` emerge as the most informative discriminators across animal classes.
- All findings are interpreted in domain terms, not just numerically — a habit relevant to applied data-science roles where stakeholder communication matters.

## Dependencies

- Python ≥ 3.9
- `pandas`
- `numpy`
- `scipy`
- `statsmodels`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `jupyter` (to run the notebook)

Install with:

```bash
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn jupyter
```

## Files

- `zoo_analysis.ipynb` — full analysis notebook
- `zoo.csv`, `class.csv` — source data (UCI / Kaggle Zoo Animal Classification)