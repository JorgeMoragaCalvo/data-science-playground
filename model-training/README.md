# Model Training — Feed-Forward Neural Networks in PyTorch

Two PyTorch notebooks training feed-forward neural networks (FFNN) end-to-end, with a focus on **training stability, regularization, and ensembling**. Built from scratch (no high-level wrappers) to demonstrate fluency with the PyTorch training loop.

## Notebooks

### `lab1.ipynb` — Binary classification with ensembling
Trains an FFNN ensemble on two binary-classification tasks:
- a synthetic Bernoulli-generated dataset (`RandomDataSet`), and
- the **scikit-learn Breast Cancer Wisconsin** dataset (real-world tabular).

### `lab1_2.ipynb` — Multi-class classification on MNIST
Adapts the same architecture and training loop for 10-class digit classification on **MNIST**, using cross-entropy loss and a softmax output head.

## What these projects demonstrate

- **Custom `nn.Module` architecture**: configurable two-hidden-layer FFNN with BatchNorm, Dropout, and a sigmoid/softmax output, all wired by hand.
- **Weight initialization strategies**: switchable **Xavier** (paired with tanh/sigmoid) and **He / Kaiming** (paired with ReLU) — initialization is matched to the activation function deliberately.
- **Optimizer comparison**: SGD, SGD + momentum, Nesterov, RMSprop, and Adam, all with L2 weight decay.
- **Regularization stack**: BatchNorm1d, Dropout, weight decay, and **early stopping** with patience and a minimum-delta tolerance (custom `EarlyStopping` class).
- **Ensembling via bootstrap aggregation (bagging)**: `create_bootstrap_samples` resamples the training set with replacement to train multiple models, with bootstrap sampling also wired into the per-epoch `DataLoader` via `SubsetRandomSampler`.
- **Proper data hygiene**: `train_test_split` for train/val/test, `StandardScaler` for feature standardization, and a separate validation loop each epoch tracking loss + accuracy.
- **Training-loop instrumentation**: per-epoch timing, live progress reporting, and loss/accuracy curves plotted across optimizer × initializer combinations for comparison.
- **Device-aware code**: GPU/CPU switch via a single flag.

## Tech stack

`PyTorch` · `torchvision` (MNIST) · `scikit-learn` (datasets, preprocessing, splits) · `numpy` · `matplotlib`

## Key takeaways

- These notebooks show the full training loop **written explicitly**: forward pass, loss, backward, optimizer step, zero-grad, validation pass — useful evidence of understanding what frameworks like Lightning abstract away.
- The experiment runner (`run_experiments_ensemble` + `plot_results`) is set up to **systematically compare** hyperparameter configurations, not just train a single model.
- Architectural choices are paired correctly with their theoretical motivation (He ↔ ReLU, Xavier ↔ tanh; bagging ↔ variance reduction; early stopping ↔ overfitting prevention).

## Dependencies

- Python ≥ 3.9
- `torch` (PyTorch)
- `torchvision` (MNIST dataset & transforms)
- `numpy`
- `scikit-learn` (`load_breast_cancer`, `train_test_split`, `StandardScaler`)
- `matplotlib`
- `psutil`
- `jupyter` (to run the notebooks)

Install with:

```bash
pip install torch torchvision numpy scikit-learn matplotlib psutil jupyter
```

GPU usage (the `run_in_GPU=True` flag) requires a CUDA-enabled PyTorch build — see the official [PyTorch install selector](https://pytorch.org/get-started/locally/) for the right command for your CUDA version.

## Files

- `lab1.ipynb` — binary classification + ensembling (synthetic & Breast Cancer)
- `lab1_2.ipynb` — multi-class classification (MNIST)