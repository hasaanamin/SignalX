# SignalX Contribution & Engineering Guidelines

This document outlines the software engineering standards, branching policies, and quality assurance workflows for **SignalX** (*Development of a Portable Multi-Mode Signal Generation and Real-Time Signal Analysis Platform with LTE-Based Validation Framework*).

SignalX is developed by **Muhammad Hasaan (23i-0019)** and **Saleh Mubashar (23i-0724)** under the supervision of **Dr. Hina Ayaz** at the **FAST School of Computing, National University of Computer and Emerging Sciences (NUCES)** in collaboration with **RIMS**.

---

## 1. Repository Branching Strategy

SignalX adheres to an industry-standard **GitFlow** model to ensure code stability, reproducible releases, and clear feature isolation:

```
  main ──────●─────────────────────────────● (v1.0.0-mid-eval Tagged Release)
              \                           /
  develop ─────●──────●───────────●──────● (Integration & Validation)
                \    /             \    /
  feature/* ─────●──●               ●──● (Feature Branches)
```

### Branch Hierarchy:
1. **`main`**:
   - Production-grade, verified release baseline.
   - Every commit on `main` is tagged with a semantic version (e.g., `v1.0.0-mid-eval`).
   - Direct pushes to `main` are restricted; code must be merged via Pull Request with green CI status.
2. **`develop`**:
   - Primary active integration branch.
   - Aggregates completed features that have passed all unit, regression, and numerical checks.
3. **`feature/<feature-name>`**:
   - Scoped branches for distinct functional requirements (e.g., `feature/lte-dsp-and-protocol-studio`, `feature/realtime-iq-stream-processing`).
   - Created off `develop` and merged back into `develop` via PR.
4. **`release/<version>`**:
   - Release candidate preparation, version bumping, and final acceptance testing.

---

## 2. Commit Message Standard (Conventional Commits)

Commit messages must follow the [Conventional Commits v1.0.0](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

### Types:
- `feat`: A new user-facing feature or DSP processing block.
- `fix`: A bug fix or numerical precision correction.
- `docs`: Documentation updates, architecture guides, or user manuals.
- `test`: Adding missing unit tests or regression verification suites.
- `ci`: Changes to GitHub Actions workflows, build scripts, or release configs.
- `perf`: Code changes that improve runtime throughput or reduce latency.
- `refactor`: Internal reorganization without changing public contracts.

---

## 3. Local Development & Verification Workflow

### 3.1 Environment Setup
```bash
# Clone the repository with Git LFS
git clone https://github.com/hasaanamin/SignalX.git
cd SignalX
git lfs pull

# Create virtual environment and install in editable mode
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -e .[test,ui]
```

### 3.2 Running Automated Tests
```bash
# Run full unit and regression test suite
python -m pytest tests/ -v

# Run targeted LTE receiver suite
python -m pytest tests/test_lte_pipeline_inspector.py tests/test_lte_pbch.py tests/test_lte_turbo.py tests/test_lte_sib.py -v
```

### 3.3 Running LTE Verification & Ground Truth Benchmark
```bash
# Verify installed LTE receiver contracts and sample captures
python benchmarks/verify_lte_install.py
```

### 3.4 Launching the Live Functional Interface
```bash
# Launch the 8-Step LTE DSP & Protocol Studio
python -m signal_processing.ui.lte_app

# Or double-click the Windows batch runner:
run_lte_ui.bat
```

---

## 4. Continuous Integration & Quality Gates

Every Pull Request and commit triggers the GitHub Actions CI pipeline:
1. **Lint & Code Quality**: Syntax verification and style analysis using `ruff`.
2. **Multi-OS Test Matrix**: Automated test execution on Ubuntu and Windows across Python 3.12 and 3.13.
3. **Numerical References Gate**: Independent validation of Fourier transforms, Welch PSD, and circular statistics against SciPy.
4. **Over-The-Air Regression Gate**: Automated verification against published over-the-air capture (`f1860_s1.92_g0_1s_strong_rtlsdr.bin`).

All CI checks must pass before merging into `develop` or `main`.
