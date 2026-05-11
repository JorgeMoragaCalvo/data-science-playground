# Generalized Conway's Automaton — Vectorization & GPU Benchmarking

A study in **high-performance numerical Python**: an N-dimensional generalization of Conway's Game of Life implemented three ways — pure Python loops, NumPy/SciPy vectorization, and CuPy on GPU — and benchmarked head-to-head across grid sizes up to 3.2 million cells.

## What this project demonstrates

- **Algorithmic generalization**: extends a 2D cellular automaton to arbitrary dimension `d`, with a configurable neighborhood radius `r` (Euclidean ball), `m` discrete states, and **toroidal (periodic) boundary conditions**.
- **Three implementations of the same logic** for direct apples-to-apples comparison:
  1. **Pure-Python loops** (`update_toroid_cycles`) — `np.ndindex` walks every cell.
  2. **NumPy vectorized** (`update_toroid_vectorized`) — boolean masks apply update rules to entire grid slices at once.
  3. **CuPy GPU** (`update_toroid_vectorized_cupy`) — same vectorized logic, dispatched to GPU.
- **Convolution as the core primitive**: neighbor sums are computed with `scipy.ndimage.convolve` (and `cupyx.scipy.ndimage.convolve`) in `mode='wrap'`, which handles periodic boundaries for free and replaces what would otherwise be deeply nested boundary-aware loops.
- **Empirical performance benchmarking**: 10 experiments sweeping `d ∈ {3,4,5}`, varying `n`, `m`, `r`, ending with a comparative table and bar chart. Representative result on the largest 3.2M-cell grid (d=5, n=20): **~88 s loops → 5.4 s NumPy → 0.4 s CuPy** (~200× end-to-end speedup over the naive version).
- **Dynamical-systems analysis**: tracks three time-series indicators per simulation — **total sum**, **variance**, and **Shannon entropy** — to search for periodic or semi-periodic patterns in the automaton's evolution, and reports the negative result honestly (wave-like oscillations, no clean periodicity).

## Tech stack

`NumPy` · `SciPy` (`ndimage.convolve`) · `CuPy` (GPU) · `pandas` · `matplotlib`

## Key takeaways

- The project is a concrete demonstration of **why vectorization matters**: identical math, identical results, two-orders-of-magnitude wall-clock difference.
- It also shows when **GPU pays off and when it doesn't**: for small grids CuPy is bottlenecked by kernel-launch overhead and barely beats NumPy; the gap widens dramatically as the grid grows.
- Performance scales with **number of neighbors** (driven by `d` and `r`), not just cell count — experiments 9 and 10 illustrate this directly.
- Demonstrates comfort with **scientific computing fundamentals**: convolution, boundary conditions, boolean masking, broadcasting, and information-theoretic metrics (Shannon entropy).

## Dependencies

- Python ≥ 3.9
- `numpy`
- `scipy` (`scipy.ndimage.convolve`)
- `pandas`
- `matplotlib`
- `jupyter` (to run the notebook)
- **Optional (GPU path)**: `cupy` — required only for `update_toroid_vectorized_cupy`. Pick the wheel matching your CUDA toolkit (e.g. `cupy-cuda12x`).

Install (CPU-only path):

```bash
pip install numpy scipy pandas matplotlib jupyter
```

Add GPU support:

```bash
pip install cupy-cuda12x   # or cupy-cuda11x, depending on your CUDA version
```

> Note: the notebook toggles between NumPy and CuPy via the import line in the setup cell — comment one out depending on whether you want to run the GPU benchmarks.

## File

- `numpy_comway.ipynb` — implementation, benchmarks, pattern-search analysis, and conclusions