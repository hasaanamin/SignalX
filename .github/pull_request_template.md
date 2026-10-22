## SignalX Pull Request

### 1. Summary of Changes
<!-- Provide a concise description of what this PR introduces, refactors, or fixes. -->

### 2. Workstream & Author
- **Author**: Muhammad Hasaan (23i-0019) / Saleh Mubashar (23i-0724)
- **Workstream**:
  - [ ] Core Signal Processing & DSP (Transforms, Noise, Feature Extraction)
  - [ ] Real-time Streaming & Ring Buffering (`signal_processing/buffering/`, `runtime/`)
  - [ ] LTE Cell Search & Protocol Decoding (`signal_processing/lte/`)
  - [ ] Hardware SDR Ingestion (ADALM-PLUTO, RTL-SDR, File Replay)
  - [ ] Live User Interface & Telemetry Studio (`signal_processing/ui/`)
  - [ ] CI/CD Pipeline & Build Infrastructure (`.github/`)

### 3. Traceability to Functional Requirements
<!-- Mark the relevant FR IDs addressed by this change -->
- [ ] **FR-01**: Complex64 IQ stream representation and immutable contracts
- [ ] **FR-02**: Timebase tracking and sample-discontinuity detection
- [ ] **FR-03**: Fixed-capacity ring buffering and zero-copy circular storage
- [ ] **FR-04**: Causal energy detection and adaptive noise baseline tracking
- [ ] **FR-05**: Event segmentation with pre/post-trigger retention
- [ ] **FR-06**: Feature extraction vector generation (56 numerical scalars)
- [ ] **FR-07**: Multi-process worker execution and decoupled analysis
- [ ] **FR-08**: Interactive graphical interface and dataset replay
- [ ] **FR-09**: SDR hardware source acquisition (Pluto / RTL-SDR)
- [ ] **FR-10**: Modulation classification baseline integration
- [ ] **FR-11**: Carrier recovery and constellation demodulation
- [ ] **FR-12**: LTE synchronization, MIB decoding, and SIB1 Turbo parsing

### 4. Verification & Testing Checklist
- [ ] `pytest tests/` passes with 100% green status locally
- [ ] No regression introduced to existing DSP transforms or memory contracts
- [ ] Independent numerical reference checks verified (SciPy/NumPy comparison)
- [ ] LTE synchronization and MIB CRC verified against reference test vectors
- [ ] Documentation / docstrings updated to reflect contract modifications
- [ ] No credentials, API tokens, or secrets committed

### 5. Reviewers & Supervisors
- **Supervisor**: Dr. Hina Ayaz
- **Co-Author**: [Saleh Mubashar / Muhammad Hasaan]
