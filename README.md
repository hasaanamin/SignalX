# SignalX: Portable Multi-Mode Signal Generation and Real-Time Signal Analysis Platform with LTE-Based Validation Framework

[![CI Pipeline](https://github.com/hasaanamin/SignalX/actions/workflows/ci.yml/badge.svg)](https://github.com/hasaanamin/SignalX/actions/workflows/ci.yml)
[![CD Release](https://github.com/hasaanamin/SignalX/actions/workflows/cd.yml/badge.svg)](https://github.com/hasaanamin/SignalX/actions/workflows/cd.yml)
[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Tests Status](https://img.shields.io/badge/tests-1%2C878%2B%20passed-brightgreen.svg)](tests/)
[![Milestone](https://img.shields.io/badge/Milestone-FYP--1%20Mid--Supervisory%20Evaluation-orange.svg)](#)
[![Institution](https://img.shields.io/badge/FAST-School%20of%20Computing-red.svg)](http://nu.edu.pk/)
[![Industry Partner](https://img.shields.io/badge/Partner-RIMS-blueviolet.svg)](#)

---

## 1. Executive Summary & Product Vision

**SignalX** is an industry-grade software-defined radio (SDR) platform providing real-time RF signal ingestion, causal noise estimation, adaptive energy detection, transient burst segmentation, zero-copy circular buffering, and mathematical feature extraction. It incorporates an end-to-end **3GPP 4G LTE Receiver and Protocol Studio** serving as an over-the-air validation benchmark for hardware, DSP pipelines, and cellular network telemetry.

Developed as a Final Year Project at the **FAST School of Computing, National University of Computer and Emerging Sciences (NUCES)** in collaboration with **RIMS**:
- **Muhammad Hasaan (23i-0019)**: Core DSP Architecture, 8-Step LTE Receiver Pipeline, Native srsRAN Turbo Decoding, Protocol Studio UI.
- **Saleh Mubashar (23i-0724)**: High-Throughput Ring Buffering, Zero-Copy Shared Transport, SDR Ingestion (ADALM-PLUTO / RTL-SDR), and CI/CD Automation.
- **Project Supervisor**: **Dr. Hina Ayaz**

---

## 2. Architecture & Pipeline Overview

The platform is designed following strict IT industry practices, featuring zero-allocation hot paths, immutable sample contracts, and decoupled multi-process execution:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Signal Ingestion Layer                          │
│  [ADALM-PLUTO SDR]  │  [RTL-SDR / HackRF]  │  [Binary Replay (cf32)]   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Positioned complex64 IQ Blocks
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   High-Throughput Transport & Buffering                │
│       • CircularIQBuffer (16k - 64k samples, zero-copy slot reuse)     │
│       • Windows MMCSS / HighQoS Performance Scheduling                 │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
┌──────────────────────────────────────┐ ┌───────────────────────────────┐
│     Adaptive Detection Engine        │ │   8-Step LTE Protocol Studio  │
│  • Causal AR1/Gaussian Noise Tracker │ │  1. Kaiser FIR Resampler      │
│  • Multi-band Energy Detector        │ │  2. Coarse/Fine CFO Sweeper   │
│  • Pre/Post Trigger Event Segmenter  │ │  3. PSS/SSS Synchronization   │
│  • 56-Scalar Numerical Feature Vector│ │  4. Physical Cell ID (PCI)    │
│    (Welch PSD, Cumulants, Phase)     │ │  5. PBCH Slot Extractor       │
│  • Labelled HDF5 / Smoke Testbench   │ │  6. CRS SFBC Equalization     │
└──────────────────────────────────────┘ │  7. MIB Viterbi Decoder (CRC) │
                                         │  8. srsRAN C Turbo Decoder &  │
                                         │     SIB1 ASN.1 Telemetry      │
                                         └───────────────┬───────────────┘
                                                         │
                                                         ▼
                                         ┌───────────────────────────────┐
                                         │  Live Functional Interface    │
                                         │   • Tab 1: Pipeline Inspector │
                                         │   • Tab 2: Telemetry JSON     │
                                         │   • Tab 3: 60 FPS Real-Time   │
                                         └───────────────────────────────┘
```

---

## 3. The 8-Step LTE DSP & Protocol Studio (POC to Product)

As presented during the proposal defense proof-of-concept and baselined for the Mid-Supervisory Evaluation, SignalX implements the full LTE physical and control channel decoding chain:

| Step | Processing Block | Description & Standards Implementation |
|---|---|---|
| **Step 1** | **Raw IQ Ingestion** | Ingests complex64 samples from live SDR, `.cf32`, `.bin`, or `.npy` files with spectral power analysis. |
| **Step 2** | **Kaiser FIR Resampler** | 1201-tap Kaiser-windowed polyphase resampler converting $2.0\text{ MS/s} \to 1.92\text{ MS/s}$ (1:1 bypass for native rate). |
| **Step 3** | **PSS Synchronization** | Frequency-domain Zadoff-Chu correlation across roots $u \in \{25, 29, 34\}$ with coarse CFO sweep ($\pm 45\text{ kHz}$). |
| **Step 4** | **SSS & PCI Resolution** | Subframe-boundary detection, fine CFO phase estimation, and Physical Cell ID computation ($N_{\text{ID}}^{\text{cell}} = 3 N_{\text{ID}}^{(1)} + N_{\text{ID}}^{(2)}$). |
| **Step 5** | **PBCH Slot Extraction** | Extracts the 72 central subcarriers across symbols 0–3 of subframe 0 with multi-frame 40 ms BCH tracking. |
| **Step 6** | **CRS & Equalization** | Cell-specific Reference Signal (CRS) channel estimation and Space-Frequency Block Coding (SFBC) diversity equalization. |
| **Step 7** | **MIB Viterbi Decoding** | Rate un-matching, 64-state tail-biting Viterbi decoding, and 16-bit CRC mask check for 1, 2, or 4 Tx antenna ports. |
| **Step 8** | **PDSCH Turbo & SIB1** | High-performance 3GPP Turbo decoding via native **srsRAN C** library (`srsran_fec.dll` / `.so`), CRC-24A validation, and ASN.1 PER SIB1 parsing. |

---

## 4. Live Functional Interface (GUI)

The platform provides a modern, high-framerate desktop interface developed with Qt and Tkinter:

- **Tab 1: Step-by-Step Inspector (Single Snapshot)**:
  Examines each of the 8 DSP pipeline stages on any selected dataset snapshot or live capture. Displays spectrum, Zadoff-Chu correlation peaks, constellation diagrams (QPSK/SFBC), and equalized channel frequency responses.
- **Tab 2: Decoded Protocol Data (JSON & Telemetry Cards)**:
  Real-time telemetry dashboard presenting decoded cellular information:
  - **Cell Identity**: Physical Cell ID (PCI), $N_{\text{ID}}^{(1)}$, $N_{\text{ID}}^{(2)}$, ECI, eNodeB ID, Sector ID, Tracking Area Code (TAC).
  - **Network Operator**: Mobile Network Operator (MNO) name, PLMN ID, MCC, MNC.
  - **Master Information Block (MIB)**: System Frame Number (SFN), Downlink Bandwidth (MHz & Resource Blocks), Antenna Ports, PHICH config.
  - **Control & Broadcast**: PCFICH CFI, PDCCH DCI 1A, SIB1 payload bits.
- **Tab 3: Continuous Live Stream (60 FPS Real-Time)**:
  Zero-lag 60 FPS real-time waterfall, FFT spectrum analyzer, and running cell synchronization status at 2 MS/s streaming rates.

---

## 5. Over-The-Air Validation Against MATLAB Ground Truth

SignalX was evaluated against real-world over-the-air LTE transmissions recorded via RTL-SDR at 1860 MHz (`f1860_s1.92_g0_1s_strong_rtlsdr.bin`). The decoded parameters achieved 100% agreement with published MATLAB LTE Toolbox ground truth:

| Decoded Metric | SignalX Observed Result | MATLAB Ground Truth | Verification Status |
|---|---|---|:---:|
| **Physical Cell ID (PCI)** | **142** ($N_{\text{ID}}^{(1)}=47, N_{\text{ID}}^{(2)}=1$) | **142** ($N_{\text{ID}}^{(1)}=47, N_{\text{ID}}^{(2)}=1$) | **MATCH** |
| **Antenna Ports** | **2 Tx Ports** | **2 Tx Ports** | **MATCH** |
| **Downlink Bandwidth** | **20.0 MHz** (100 Resource Blocks) | **20.0 MHz** (100 Resource Blocks) | **MATCH** |
| **System Frame Number (SFN)** | **589** | **589** | **MATCH** |
| **Carrier Frequency Offset (CFO)**| **-41.4 kHz** | **-41.5 kHz** | **MATCH** |
| **PHICH Configuration** | Normal Duration, Resource = 1 | Normal Duration, Resource = 1 | **MATCH** |
| **PBCH CRC-16 Check** | **PASSED** (Valid Ant-2 Mask) | **PASSED** | **MATCH** |

---

## 6. Git Branching Strategy & CI/CD Pipeline

To meet FAST School of Computing and international IT industry product standards, SignalX implements a structured **GitFlow** model with full CI/CD automation:

```
  main ──────●─────────────────────────────● (v1.0.0-mid-eval Tagged Release)
              \                           /
  develop ─────●──────●───────────●──────● (Integration & Verification)
                \    /             \    /
  feature/* ─────●──●               ●──● (Isolated Feature Workstreams)
```

### 6.1 Meaningful Branch Hierarchy
- **`main`**: Production-ready release branch. Strictly tagged (`v1.0.0-mid-eval`). Direct commits disabled.
- **`develop`**: Integration branch for ongoing verified engineering work.
- **`feature/lte-dsp-and-protocol-studio`**: Contains the 8-step LTE receiver, native srsRAN C Turbo decoder, and live UI studio.
- **`feature/realtime-iq-stream-processing`**: Contains the high-throughput circular buffer, causal noise estimation, and SDR adapters.
- **`release/v1.0.0-mid-eval`**: Dedicated release staging branch for the FYP-1 mid-evaluation milestone.

### 6.2 GitHub Actions Continuous Integration (CI)
The automated CI pipeline (`.github/workflows/ci.yml`) triggers on every push and pull request:
1. **Linting & Code Style**: Syntax and compliance verification via `ruff` and `black`.
2. **Multi-OS Test Matrix**: Runs full pytest test suites across **Ubuntu** and **Windows** on Python 3.12 and 3.13.
3. **Native C Shared Library Build**: Cross-compiles the pinned srsRAN C Turbo and rate-matching engine (`gcc` on Linux, MSVC on Windows).
4. **DSP & Numerical Gate**: Compares Periodograms, Welch PSD, circular variance, and cumulants against independent SciPy references.
5. **Over-The-Air Regression Gate**: Validates continuous synchronization and MIB decoding on reference OTA data.

### 6.3 GitHub Actions Continuous Delivery (CD)
The automated CD pipeline (`.github/workflows/cd.yml`) triggers on version tags (`v*`):
1. **Python Distribution Packages**: Builds standard wheels (`.whl`) and source archives (`.tar.gz`).
2. **Windows Standalone Portable Bundle**: Packages an out-of-the-box portable application zip including runners (`run_lte_ui.bat`), configurations, test data, and pre-compiled DLLs.
3. **SHA-256 Checksums**: Automatically computes cryptographic hashes for supply chain security.
4. **Automated GitHub Release**: Publishes a release with changelog and downloadable binary assets.

---

## 7. Quickstart Guide

### 7.1 Installation
```bash
# 1. Clone repository with Git LFS
git lfs install
git clone https://github.com/hasaanamin/SignalX.git
cd SignalX
git lfs pull

# 2. Set up virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -e .[test,ui]
```

### 7.2 Launching the Live Functional Studio
```bash
# Windows Quick Launch:
run_lte_ui.bat

# Or via Python module:
python -m signal_processing.ui.lte_app
```

### 7.3 Running the Complete Dataset UI
```bash
# Windows Quick Launch:
run_detector_complete_ui.bat

# Or via Python module:
python -m signal_processing.ui.hdf5_app --smoke-dataset detector_dataset_complete
```

### 7.4 Running Automated Verification Suites
```bash
# Run all 1,878+ unit and regression tests
python -m pytest tests/ -v

# Run LTE receiver install verification
python benchmarks/verify_lte_install.py
```

---

## 8. Requirements Traceability Matrix

| Requirement ID | Specification Description | Primary Implementation | Verification Suite | Status |
|---|---|---|---|:---:|
| **FR-01** | Immutable complex64 stream contracts | `signal_processing/core/contracts.py` | `tests/core/` | **Implemented** |
| **FR-02** | Timebase and sample discontinuity detection | `signal_processing/core/stream_compatibility.py` | `tests/core/` | **Implemented** |
| **FR-03** | Fixed-capacity ring buffering and zero-copy transport | `signal_processing/buffering/ring_buffer.py` | `tests/buffering/` | **Implemented** |
| **FR-04** | Causal energy detection and online AR1 noise tracker | `signal_processing/detection/online_noise.py` | `tests/test_online_noise.py` | **Implemented** |
| **FR-05** | Transient event segmentation with pre/post trigger | `signal_processing/segmentation/` | `tests/segmentation/` | **Implemented** |
| **FR-06** | 56-scalar feature extraction vector | `signal_processing/features/` | `tests/features/` | **Implemented** |
| **FR-07** | Decoupled multi-process worker architecture | `signal_processing/runtime/worker.py` | `tests/runtime/` | **Implemented** |
| **FR-08** | Desktop dataset replay and evaluation testbench | `signal_processing/ui/hdf5_runner.py` | `tests/test_hdf5_pipeline_ui.py` | **Implemented** |
| **FR-09** | Hardware SDR ingestion (ADALM-PLUTO / RTL-SDR) | `signal_processing/sources/pluto_source.py` | `tests/test_pluto_source.py` | **Implemented** |
| **FR-10** | Modulation classification baseline integration | `signal_processing/classification/` | Iteration 2 | *Planned* |
| **FR-11** | Carrier frequency recovery and constellation demod | `signal_processing/demodulation/` | Iteration 2 | *Planned* |
| **FR-12** | 8-step LTE receiver, MIB decoding & Turbo SIB1 | `signal_processing/lte/` | `tests/test_lte_*.py` | **Implemented** |

---

## 9. Project Governance & Compliance

- **Branching & Contribution Policy**: See [CONTRIBUTING.md](CONTRIBUTING.md).
- **Issue Templates**: Available in `.github/ISSUE_TEMPLATE/` for bug reports and feature requests.
- **Pull Request Template**: Available in `.github/pull_request_template.md`.
- **License**: Proprietary / FAST School of Computing Academic Project.
