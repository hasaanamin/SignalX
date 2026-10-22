"""Tests for LTE Step-by-Step Inspector and Streaming Pipeline Engine."""

import time

import numpy as np

from signal_processing.ui.lte_pipeline_inspector import (
    ContinuousStreamRunner,
    LTEStepInspector,
    PipelineInspectionResult,
    Step1Result,
    Step2Result,
    Step3Result,
    Step4Result,
    Step5Result,
    Step6Result,
    Step7Result,
    SyntheticLTEStreamSource,
    waveform,
)


def test_step_by_step_inspection_synthetic_2msps():
    """Verify all 7 steps extract valid intermediate data on compliant 2 MS/s LTE signal."""
    # Synthesize 4 frames (80,000 samples) with PCI=42, 2 antenna ports, SNR=22 dB, CFO=+1000 Hz
    iq = waveform(pci=42, ports=2, frames=4, rate=2_000_000.0, cfo=1000.0, snr_db=22.0)
    inspector = LTEStepInspector(sample_rate_hz=2_000_000.0)

    # Step 1: Input Ingestion
    step1 = inspector.inspect_step1_input(iq)
    assert isinstance(step1, Step1Result)
    assert step1.sample_rate_hz == 2_000_000.0
    assert step1.num_samples == len(iq)
    assert len(step1.time_ms) > 0
    assert len(step1.psd_freqs_mhz) > 0
    assert step1.rms_power_dbfs < 10.0

    # Step 2: Preprocessing & Resampling
    step2 = inspector.inspect_step2_preprocessing(iq)
    assert isinstance(step2, Step2Result)
    assert step2.input_rate_hz == 2_000_000.0
    assert step2.output_rate_hz == 1_920_000.0
    assert step2.resampling_ratio_str == "24/25"
    assert len(step2.raw_iq_filtered) > 0
    assert len(step2.filter_freqs_mhz) > 0
    assert len(step2.output_psd_freqs_mhz) > 0

    # Step 3: PSS Search & Timing
    step3 = inspector.inspect_step3_pss(step2.raw_iq_filtered)
    assert isinstance(step3, Step3Result)
    # PCI = 42 -> N_ID_2 = 42 % 3 = 0
    assert step3.detected_nid2 == 0
    assert step3.peak_metric > 0.5
    assert step3.detection_passed is True
    assert abs(step3.coarse_cfo_hz - 1000.0) < 5000.0

    # Step 4: SSS Correlation & Cell ID
    step4 = inspector.inspect_step4_sss(step2.raw_iq_filtered, step3)
    assert isinstance(step4, Step4Result)
    # N_ID_1 = 42 // 3 = 14
    assert step4.detected_nid1 == 14
    assert step4.detected_pci == 42
    assert step4.detection_passed is True
    assert step4.peak_ratio > 1.5

    # Step 5: PBCH Slot & Resource Grid
    step5 = inspector.inspect_step5_slot_grid(step2.raw_iq_filtered, step3, step4)
    assert isinstance(step5, Step5Result)
    assert step5.grid_magnitude_db.shape == (5, 72)
    assert step5.re_type_map.shape == (5, 72)
    # CRS pilots marked (1) and PBCH data marked (2)
    assert np.any(step5.re_type_map == 1)
    assert np.any(step5.re_type_map == 2)
    assert abs(step5.total_cfo_hz - 1000.0) < 200.0

    # Step 6: Channel Estimation & Equalization
    step6 = inspector.inspect_step6_channel(step5, step4)
    assert isinstance(step6, Step6Result)
    assert step6.h_magnitude_port0.shape == (72,)
    assert step6.soft_symbols_i.shape == (240,)
    assert step6.soft_symbols_q.shape == (240,)

    # Step 7: Viterbi Demodulation & Decoded MIB
    step7 = inspector.inspect_step7_mib_decode(step5, step4)
    assert isinstance(step7, Step7Result)
    assert step7.crc_valid is True
    assert step7.antenna_ports == 2
    assert step7.downlink_bandwidth_hz in (1_400_000, 3_000_000, 5_000_000, 10_000_000, 15_000_000, 20_000_000)
    assert len(step7.raw_payload_bits) == 24
    assert len(step7.constellation_i) == 240


def test_full_pipeline_inspection():
    """Verify full pipeline execution wrapper returns complete structure."""
    iq = waveform(pci=174, ports=1, frames=4, rate=2_000_000.0, cfo=0.0, snr_db=25.0)
    inspector = LTEStepInspector(sample_rate_hz=2_000_000.0)
    res = inspector.process_full_inspection(iq)

    assert isinstance(res, PipelineInspectionResult)
    assert res.success is True
    assert res.step4.detected_pci == 174
    assert res.step7.crc_valid is True
    assert res.step7.antenna_ports == 1
    assert "Successfully Synchronized & Decoded MIB!" in res.summary_message
    assert res.step8 is not None
    assert res.step8.has_sib1 is True
    assert res.step8.crc24a_valid is True
    assert len(res.step8.operator_name) > 0
    assert len(res.step8.raw_bits) == 120


def test_synthetic_stream_source_and_runner():
    """Verify real-time 2 MS/s continuous streaming engine."""
    src = SyntheticLTEStreamSource(pci=42, ports=2, snr_db=25.0, cfo_hz=0.0, block_size=16384)
    runner = ContinuousStreamRunner(src, sample_rate_hz=2_000_000.0, block_size=16384)

    runner.start()
    time.sleep(0.4)
    snap = runner.get_latest_snapshot()
    runner.stop()

    assert snap is not None
    assert snap.total_samples > 0
    assert snap.actual_rate_msps > 0.0
    assert len(snap.latest_block_iq) > 0
    assert len(snap.spectrum_dbfs) > 0
    assert snap.pci == 42
    assert snap.sync_state in ("CANDIDATE", "SYNCHRONIZED")
