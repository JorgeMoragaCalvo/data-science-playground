# Data Analysis Playground

A collection of self-contained Python projects covering the data-science stack end-to-end: exploratory statistics, neural-network training in PyTorch, and high-performance numerical computing with NumPy / CuPy. Each subdirectory is an independent project with its own notebook(s), README, and dependency list.

## Projects

### [📊 data-analysis/](./data-analysis/README.md) — Statistical analysis of the Zoo dataset
Full EDA pipeline on the UCI Zoo dataset: data-quality auditing, descriptive visualizations, correlation analysis with the **phi coefficient** for binary attributes, inferential testing (**Z-test, Fisher's exact, ANOVA, chi-square**), and **PCA** for dimensionality reduction and class-separation analysis.

**Stack:** `pandas` · `scipy.stats` · `statsmodels` · `scikit-learn` · `matplotlib` · `seaborn`

---

### [🧠 model-training/](./model-training/README.md) — Feed-forward neural networks in PyTorch
Two notebooks training FFNNs **from scratch** (explicit training loop): binary classification with **bootstrap-aggregation ensembling** on the Breast Cancer dataset, and multi-class classification on **MNIST**. Covers BatchNorm, Dropout, Xavier/He initialization, optimizer sweeps (SGD/Momentum/Nesterov/RMSprop/Adam), and a custom **early-stopping** implementation.

**Stack:** `PyTorch` · `torchvision` · `scikit-learn` · `numpy` · `matplotlib`

---

### [⚡ numpy-comway/](./numpy-comway/README.md) — N-dimensional generalized Conway's automaton
A benchmarking study of three implementations of the same cellular automaton — **pure Python loops vs. NumPy vectorization vs. CuPy on GPU** — across grids up to 3.2M cells. Uses convolution with toroidal boundary conditions, and analyzes the system's dynamics via total sum, variance, and **Shannon entropy** time series.

**Stack:** `NumPy` · `SciPy` (`ndimage.convolve`) · `CuPy` · `pandas` · `matplotlib`

---

## What this repo demonstrates

- **Statistical literacy**: choosing the right test for the data type (binary, categorical, continuous) and interpreting results in domain terms.
- **Deep-learning fundamentals**: writing the PyTorch training loop by hand, with proper regularization, validation, and ensembling.
- **Performance engineering**: profiling and accelerating numerical Python via vectorization and GPU offloading.
- **Communication**: each project ships with a self-contained README explaining methods, tools, and outcomes.

## Repo layout

```
.
├── data-analysis/        # Zoo dataset: EDA, hypothesis tests, PCA
├── model-training/       # PyTorch FFNNs: Breast Cancer + MNIST
├── numpy-comway/         # N-D cellular automaton: NumPy vs CuPy benchmarks
└── README.md             # this file
```

## Getting started

Each project lists its own dependencies and install command in its README. Notebooks are Jupyter-compatible and also runnable in Google Colab (badge links inside each notebook).

## Author

**Jorge Moraga Calvo** — [GitHub](https://github.com/JorgeMoragaCalvo)