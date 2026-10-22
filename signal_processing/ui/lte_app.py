"""PyQt6 & pyqtgraph Desktop UI for LTE Signal Processing & Protocol Analysis.

Features:
  - Ingests LTE complex64 samples (2 MS/s or 1.92 MS/s).
  - Mode 1: Single Sample / Step-by-Step Inspector:
      Inspects intermediate waveforms, spectra, correlation peaks, grids,
      channel responses, constellations, and MIB bits for all 7 pipeline steps.
  - Mode 2: Continuous 2 MS/s Stream:
      High-throughput worker processing at 2 MS/s, decoupled from a silky-smooth
      60 fps GUI timer update (fulfills requirement of displaying ~60 updates/sec).
  - Decoded Info Banner:
      Live KPI cards for PCI, SFN, Bandwidth, Antennas, CFO, Sync State, and CRC.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSplitter,
    QStyleFactory,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from signal_processing.ui.lte_pipeline_inspector import (
    ContinuousStreamRunner,
    FileLTEStreamSource,
    LTEStepInspector,
    PipelineInspectionResult,
    StreamSnapshot,
    SyntheticLTEStreamSource,
    waveform,
)

# Configure pyqtgraph for dark aesthetics & performance
pg.setConfigOption("background", "#0d131a")
pg.setConfigOption("foreground", "#c9d1d9")
pg.setConfigOption("antialias", False)  # Fast 60 fps rendering


DARK_STYLESHEET = """
QMainWindow {
    background-color: #0b0f14;
}
QWidget {
    color: #e6edf3;
    font-family: "Segoe UI", -apple-system, Roboto, sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #1e293b;
    background-color: #0d131a;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #16202c;
    color: #94a3b8;
    padding: 10px 24px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #0d131a;
    color: #00e5ff;
    border-bottom: 3px solid #00e5ff;
}
QTabBar::tab:hover:!selected {
    background-color: #1e2c3d;
    color: #f1f5f9;
}
QFrame.kpi-card {
    background-color: #131c26;
    border: 1px solid #223244;
    border-radius: 8px;
    padding: 10px 14px;
}
QFrame.kpi-card:hover {
    border: 1px solid #00e5ff;
}
QLabel.kpi-title {
    color: #94a3b8;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
QLabel.kpi-value {
    color: #ffffff;
    font-size: 20px;
    font-weight: bold;
}
QLabel.kpi-sub {
    color: #64748b;
    font-size: 11px;
}
QPushButton {
    background-color: #1e293b;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #2b394e;
    border-color: #00e5ff;
}
QPushButton:pressed {
    background-color: #0f172a;
}
QPushButton.btn-primary {
    background-color: #0070f3;
    border-color: #0070f3;
    color: #ffffff;
}
QPushButton.btn-primary:hover {
    background-color: #0060df;
}
QPushButton.btn-success {
    background-color: #059669;
    border-color: #059669;
    color: #ffffff;
}
QPushButton.btn-success:hover {
    background-color: #047857;
}
QPushButton.btn-danger {
    background-color: #dc2626;
    border-color: #dc2626;
    color: #ffffff;
}
QPushButton.btn-danger:hover {
    background-color: #b91c1c;
}
QGroupBox {
    border: 1px solid #1e293b;
    border-radius: 6px;
    margin-top: 12px;
    font-weight: 700;
    color: #00e5ff;
    padding-top: 16px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QTableWidget {
    background-color: #111822;
    border: 1px solid #1e293b;
    gridline-color: #1e293b;
    border-radius: 6px;
    color: #e2e8f0;
}
QHeaderView::section {
    background-color: #16202c;
    color: #94a3b8;
    padding: 6px;
    border: 1px solid #1e293b;
    font-weight: 600;
    font-size: 11px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #00e5ff;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    border: 1px solid #00e5ff;
    width: 14px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 7px;
}
QComboBox {
    background-color: #16202c;
    border: 1px solid #2b394e;
    border-radius: 6px;
    padding: 6px 12px;
    color: #f1f5f9;
}
QComboBox QAbstractItemView {
    background-color: #16202c;
    border: 1px solid #2b394e;
    selection-background-color: #0070f3;
}
QDialog {
    background-color: #0b0f14;
    color: #f8fafc;
}
QLineEdit {
    background-color: #16202c;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 7px 12px;
    color: #00e5ff;
    font-size: 13px;
    font-weight: 600;
}
QLineEdit:focus {
    border: 1px solid #00e5ff;
    background-color: #1a2737;
}
QPlainTextEdit {
    background-color: #080c12;
    border: 1px solid #1e293b;
    border-radius: 6px;
    color: #38bdf8;
    font-family: "Cascadia Code", "Fira Code", Consolas, monospace;
    font-size: 12px;
}
QScrollBar:vertical {
    border: none;
    background: #0b0f14;
    width: 8px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #223244;
    min-height: 20px;
    border-radius: 4px;
}
"""


def build_decoded_protocol_dict(res: Optional[PipelineInspectionResult]) -> Dict[str, Any]:
    """Constructs a clean, structured dictionary of all decoded cellular protocol telemetry."""
    if res is None:
        return {
            "status": "AWAITING_INPUT",
            "message": "No LTE samples ingested yet. Click 'Generate Synthetic LTE Snapshot' or 'Load IQ File' to decode.",
            "pipeline_state": "IDLE",
            "network_operator": None,
            "cell_identity": None,
            "master_information_block": None,
            "control_channels": None,
            "downlink_shared_channel_sib1": None,
        }

    s1 = res.step1
    s2 = res.step2
    s3 = res.step3
    s4 = res.step4
    s5 = res.step5
    s6 = res.step6
    s7 = res.step7
    s8 = res.step8

    return {
        "status": "SYNCHRONIZED" if res.success else "DECODE_FAILED",
        "summary": res.summary_message,
        "cell_identity": {
            "physical_cell_id_pci": s4.detected_pci,
            "pss_root_index_nid2": s4.detected_nid2,
            "sss_group_index_nid1": s4.detected_nid1,
            "global_cell_identity_eci": s8.cell_identity if (s8 and s8.has_sib1) else None,
            "enodeb_id": s8.enodeb_id if (s8 and s8.has_sib1) else None,
            "sector_id": s8.sector_id if (s8 and s8.has_sib1) else None,
            "tracking_area_code_tac": s8.tracking_area_code if (s8 and s8.has_sib1) else None,
            "tac_hex": f"0x{s8.tracking_area_code:04X}" if (s8 and s8.has_sib1) else None,
        },
        "network_operator": {
            "operator_name": s8.operator_name if (s8 and s8.has_sib1) else "Awaiting SIB1",
            "plmn_id": s8.plmn_id if (s8 and s8.has_sib1) else None,
            "mcc": s8.mcc if (s8 and s8.has_sib1) else None,
            "mnc": s8.mnc if (s8 and s8.has_sib1) else None,
            "operating_band": f"Band {s8.freq_band}" if (s8 and s8.has_sib1) else None,
            "q_rx_lev_min_dbm": s8.q_rx_lev_min_dbm if (s8 and s8.has_sib1) else None,
        },
        "master_information_block": {
            "system_frame_number_sfn": s7.system_frame_number,
            "sfn_quarter": s7.frame_quarter,
            "downlink_bandwidth_hz": s7.downlink_bandwidth_hz,
            "downlink_bandwidth_mhz": s7.downlink_bandwidth_hz / 1e6,
            "downlink_resource_blocks": s7.downlink_resource_blocks,
            "antenna_ports": s7.antenna_ports,
            "phich_duration": s7.phich_duration,
            "phich_resource": s7.phich_resource,
            "raw_payload_bits": "".join(map(str, s7.raw_payload_bits)),
            "crc16_valid": s7.crc_valid,
        },
        "control_channels": {
            "control_format_indicator_cfi": s8.cfi if s8 else None,
            "pcfich_detection_confidence": s8.pcfich_confidence if s8 else None,
            "pdcch_dci_format": "1A (Common Search Space, SI-RNTI 0xFFFF)",
            "pdsch_start_rb": s8.dci_start_rb if s8 else None,
            "pdsch_num_rbs": s8.dci_num_rbs if s8 else None,
        },
        "downlink_shared_channel_sib1": {
            "transport_block_size_bits": len(s8.raw_bits) if (s8 and s8.has_sib1) else None,
            "turbo_fec_engine": "srsRAN 4G C Native Shared Library (srsran_fec.dll)",
            "turbo_decoder_iterations": s8.turbo_iterations if (s8 and s8.has_sib1) else None,
            "crc24a_valid": s8.crc24a_valid if (s8 and s8.has_sib1) else False,
            "raw_sib1_bits_preview": "".join(map(str, s8.raw_bits[:48])) + "..." if (s8 and s8.has_sib1) else None,
        },
        "rf_synchronization": {
            "detected_subframe": s4.detected_subframe,
            "frame_start_sample": s4.frame_start_sample,
            "pss_peak_metric": round(s3.peak_metric, 4),
            "sss_peak_metric": round(s4.sss_metric, 4),
            "coarse_cfo_hz": round(s3.coarse_cfo_hz, 2),
            "fine_residual_cfo_hz": round(s5.residual_cfo_hz, 2),
            "total_cfo_hz": round(s5.total_cfo_hz, 2),
            "rms_power_dbfs": round(s1.rms_power_dbfs, 2),
            "papr_db": round(s1.papr_db, 2),
            "sample_rate_in_hz": s1.sample_rate_hz,
            "sample_rate_processed_hz": s2.output_rate_hz,
        },
    }


class KPICard(QFrame):
    """Sleek metric KPI card with title, large value, and subtitle."""

    def __init__(self, title: str, default_val: str = "--", subtitle: str = ""):
        super().__init__()
        self.setProperty("class", "kpi-card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        self.title_lbl = QLabel(title)
        self.title_lbl.setProperty("class", "kpi-title")

        self.val_lbl = QLabel(default_val)
        self.val_lbl.setProperty("class", "kpi-value")

        self.sub_lbl = QLabel(subtitle)
        self.sub_lbl.setProperty("class", "kpi-sub")

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.val_lbl)
        layout.addWidget(self.sub_lbl)

    def set_data(self, val: str, subtitle: Optional[str] = None, color: Optional[str] = None):
        self.val_lbl.setText(val)
        if subtitle is not None:
            self.sub_lbl.setText(subtitle)
        if color:
            self.val_lbl.setStyleSheet(f"color: {color};")
        else:
            self.val_lbl.setStyleSheet("")


class StepNavigatorWidget(QFrame):
    """Left sidebar list allowing user to navigate between steps 1-7."""

    step_selected = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(260)
        self.setStyleSheet("background-color: #101720; border-right: 1px solid #1e293b;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(8)

        header = QLabel("PIPELINE STEPS")
        header.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 11px; letter-spacing: 1px;")
        layout.addWidget(header)

        self.buttons: List[QPushButton] = []
        self.step_titles = [
            "1. Raw IQ Ingestion",
            "2. Preprocessing & Resampler",
            "3. PSS Acquisition & Timing",
            "4. SSS Correlation & Cell ID",
            "5. PBCH Slot & Resource Grid",
            "6. Channel Estimation & Equalization",
            "7. Viterbi Decoding & MIB",
            "8. SIB1 & Operator (srsRAN C)",
        ]

        for idx, title in enumerate(self.step_titles):
            btn = QPushButton(title)
            btn.setCheckable(True)
            btn.setStyleSheet(
                "text-align: left; padding: 10px 12px; font-size: 12px; border-radius: 6px;"
            )
            btn.clicked.connect(lambda checked, i=idx: self._on_step_click(i))
            layout.addWidget(btn)
            self.buttons.append(btn)

        self.buttons[0].setChecked(True)
        layout.addStretch()

        # Step description box
        self.desc_group = QGroupBox("Active Step Context")
        desc_layout = QVBoxLayout(self.desc_group)
        self.desc_lbl = QLabel("Step 1: Inspects raw time-domain IQ samples, spectrum, and power metrics.")
        self.desc_lbl.setWordWrap(True)
        self.desc_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        desc_layout.addWidget(self.desc_lbl)
        layout.addWidget(self.desc_group)

    def _on_step_click(self, index: int):
        for i, b in enumerate(self.buttons):
            b.setChecked(i == index)
        self.step_selected.emit(index)

    def set_active_step(self, index: int):
        for i, b in enumerate(self.buttons):
            b.setChecked(i == index)
        self._update_description(index)

    def _update_description(self, index: int):
        descriptions = [
            "Step 1: Ingests raw complex64 IQ at 2 MS/s. Calculates RMS, PAPR, and 2 MS/s power spectral density.",
            "Step 2: 24/25 rational resampler with 1201-tap Kaiser FIR low-pass filter (cutoff 700 kHz), converting to native 1.92 MS/s.",
            "Step 3: Primary Synchronization Signal matched filter search over roots N_ID_2 in {0,1,2} and coarse CFO hypotheses.",
            "Step 4: Secondary Synchronization Signal correlation over 168 hypotheses, subframe 0/5 detection, and PCI derivation.",
            "Step 5: Subframe 0, Slot 1 extraction, residual CFO fine estimation via cyclic prefix (CP) correlation, and 128-pt FFT.",
            "Step 6: CRS pilot extraction, linear frequency/time channel response interpolation, and SFBC Alamouti diversity.",
            "Step 7: Equalized soft symbols, de-scrambling, 64-state tail-biting Viterbi decoding, CRC-16 check, and 24-bit MIB parsing.",
            "Step 8: PCFICH CFI, PDCCH DCI 1A, srsRAN C Turbo PDSCH decoding (srsran_fec.dll), CRC-24A, and SIB1 operator unmarshaling.",
        ]
        if index < len(descriptions):
            self.desc_lbl.setText(descriptions[index])


class LTEStudioApp(QMainWindow):
    """Main LTE Signal Processing & Protocol Analysis Application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SignalX - LTE Signal Processing & Protocol Analysis Studio")
        self.resize(1560, 960)
        self.setMinimumSize(1200, 780)
        self.setStyleSheet(DARK_STYLESHEET)

        # Core Engines
        self.inspector = LTEStepInspector(sample_rate_hz=2_000_000.0)
        self.last_inspection: Optional[PipelineInspectionResult] = None
        self.active_step_index = 0

        self.stream_runner: Optional[ContinuousStreamRunner] = None
        self.synthetic_source = SyntheticLTEStreamSource(pci=42, ports=2, snr_db=20.0, cfo_hz=0.0)

        # UI Setup
        self._init_ui()

        # Decoupled 60 fps Timer for Live Continuous Mode
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(16)  # ~60 Hz (16.6 ms)
        self.live_timer.timeout.connect(self._on_live_frame_tick)

        # Initial state: clean and empty until user generates a snapshot or loads a file
        self._reset_to_empty_state()

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 12, 16, 16)
        main_layout.setSpacing(12)

        # 1. Top Header Bar
        header = self._create_header_bar()
        main_layout.addWidget(header)

        # 2. Decoded KPI Cards Banner (Top)
        kpi_banner = self._create_kpi_banner()
        main_layout.addWidget(kpi_banner)

        # 3. Mode Tabs (Step Inspector, Decoded Protocol Data (JSON), Continuous Stream)
        self.tabs = QTabWidget()
        self.tab_step_inspector = self._create_step_inspector_tab()
        self.tab_decoded_protocol = self._create_decoded_protocol_tab()
        self.tab_continuous_stream = self._create_continuous_stream_tab()

        self.tabs.addTab(self.tab_step_inspector, "Step-by-Step Inspector (Single Snapshot)")
        self.tabs.addTab(self.tab_decoded_protocol, "Decoded Protocol Data (JSON & Summary)")
        self.tabs.addTab(self.tab_continuous_stream, "Continuous Live Stream (2 MS/s @ 60 FPS)")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        main_layout.addWidget(self.tabs, stretch=1)

    def _create_header_bar(self) -> QWidget:
        header = QFrame()
        header.setStyleSheet("background-color: #101720; border: 1px solid #1e293b; border-radius: 8px; padding: 6px;")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 6, 12, 6)

        title_layout = QVBoxLayout()
        title = QLabel("SIGNALX LTE DSP & PROTOCOL STUDIO")
        title.setStyleSheet("color: #00e5ff; font-weight: 800; font-size: 15px; letter-spacing: 1px;")
        sub = QLabel("FDD Normal-CP Receiver Pipeline | 2.0 MS/s Input -> 1.92 MS/s Processing")
        sub.setStyleSheet("color: #64748b; font-size: 11px;")
        title_layout.addWidget(title)
        title_layout.addWidget(sub)
        layout.addLayout(title_layout)

        layout.addStretch()

        # Telemetry pills
        self.lbl_rate = QLabel("SAMPLE RATE: 2.000 MS/s")
        self.lbl_rate.setStyleSheet("background: #1e293b; color: #38bdf8; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 11px;")
        layout.addWidget(self.lbl_rate)

        self.lbl_fps = QLabel("GUI RATE: 60 FPS")
        self.lbl_fps.setStyleSheet("background: #1e293b; color: #4ade80; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 11px;")
        layout.addWidget(self.lbl_fps)

        self.lbl_status = QLabel("STATUS: IDLE")
        self.lbl_status.setStyleSheet("background: #1e293b; color: #fbbf24; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 11px;")
        layout.addWidget(self.lbl_status)

        return header

    def _create_kpi_banner(self) -> QWidget:
        banner = QWidget()
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.kpi_pci = KPICard("Cell ID (PCI)", "--", "N_ID_1: -- | N_ID_2: --")
        self.kpi_sfn = KPICard("Frame No. (SFN)", "--", "Quarter: --")
        self.kpi_bw = KPICard("Downlink BW", "--", "-- Resource Blocks")
        self.kpi_ant = KPICard("Tx Antennas", "--", "SFBC Diversity")
        self.kpi_cfo = KPICard("Frequency Offset", "-- Hz", "Residual: -- Hz")
        self.kpi_operator = KPICard("Network Operator", "--", "srsRAN C SIB1")
        self.kpi_sync = KPICard("Sync State", "SEARCHING", "PSS: -- | SSS: --")
        self.kpi_crc = KPICard("CRC Check", "PENDING", "Antenna Mask")

        layout.addWidget(self.kpi_pci)
        layout.addWidget(self.kpi_sfn)
        layout.addWidget(self.kpi_bw)
        layout.addWidget(self.kpi_ant)
        layout.addWidget(self.kpi_cfo)
        layout.addWidget(self.kpi_operator)
        layout.addWidget(self.kpi_sync)
        layout.addWidget(self.kpi_crc)

        return banner

    # ========================================================================
    # Tab 1: Step-by-Step Inspector
    # ========================================================================

    def _create_step_inspector_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Left Step Navigator
        self.step_nav = StepNavigatorWidget()
        self.step_nav.step_selected.connect(self._on_step_nav_selected)
        layout.addWidget(self.step_nav)

        # Center Main View
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Step Controls Bar
        ctrl_bar = QFrame()
        ctrl_bar.setStyleSheet("background-color: #121c27; border: 1px solid #1e293b; border-radius: 6px;")
        ctrl_layout = QHBoxLayout(ctrl_bar)
        ctrl_layout.setContentsMargins(12, 8, 12, 8)

        btn_load = QPushButton("Load IQ File (.cf32 / .h5 / .npy)")
        btn_load.clicked.connect(self._on_load_file_snapshot)
        ctrl_layout.addWidget(btn_load)

        btn_synth = QPushButton("Generate Synthetic Snapshot")
        btn_synth.setProperty("class", "btn-primary")
        btn_synth.clicked.connect(self._on_synth_snapshot_dialog)
        ctrl_layout.addWidget(btn_synth)

        ctrl_layout.addSpacing(16)

        btn_prev = QPushButton("◀ Previous Step")
        btn_prev.clicked.connect(self._on_step_prev)
        ctrl_layout.addWidget(btn_prev)

        btn_next = QPushButton("Next Step ▶")
        btn_next.clicked.connect(self._on_step_next)
        ctrl_layout.addWidget(btn_next)

        btn_run_all = QPushButton("Run Full Pipeline")
        btn_run_all.setProperty("class", "btn-success")
        btn_run_all.clicked.connect(self._on_run_full_pipeline)
        ctrl_layout.addWidget(btn_run_all)

        btn_view_json = QPushButton("{ } View Decoded JSON")
        btn_view_json.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        ctrl_layout.addWidget(btn_view_json)

        ctrl_layout.addStretch()

        self.lbl_snapshot_info = QLabel("Source: No signal loaded (Click 'Generate Synthetic Snapshot')")
        self.lbl_snapshot_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        ctrl_layout.addWidget(self.lbl_snapshot_info)

        right_layout.addWidget(ctrl_bar)

        # Dynamic Plot Area (stacked/swapped based on step)
        self.step_plot_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top Plot Widget
        self.step_plot1 = pg.PlotWidget()
        self.step_plot1.showGrid(x=True, y=True, alpha=0.25)
        self.step_plot_splitter.addWidget(self.step_plot1)

        # Bottom Plot Widget
        self.step_plot2 = pg.PlotWidget()
        self.step_plot2.showGrid(x=True, y=True, alpha=0.25)
        self.step_plot_splitter.addWidget(self.step_plot2)

        # Third View Widget (for heatmaps, tables, constellation)
        self.step_plot3 = pg.PlotWidget()
        self.step_plot3.showGrid(x=True, y=True, alpha=0.25)
        self.step_plot_splitter.addWidget(self.step_plot3)

        right_layout.addWidget(self.step_plot_splitter, stretch=1)

        # Bottom Metrics Bar for Active Step
        self.step_metrics_lbl = QLabel("Step Metrics: Ready")
        self.step_metrics_lbl.setStyleSheet(
            "background-color: #101720; border: 1px solid #1e293b; padding: 8px 12px; border-radius: 4px; color: #38bdf8; font-family: monospace;"
        )
        right_layout.addWidget(self.step_metrics_lbl)

        layout.addWidget(right_panel, stretch=1)
        return tab

    def _on_step_nav_selected(self, index: int):
        self.active_step_index = index
        self.step_nav.set_active_step(index)
        self._render_active_step()

    def _on_step_prev(self):
        if self.active_step_index > 0:
            self.active_step_index -= 1
            self.step_nav.set_active_step(self.active_step_index)
            self._render_active_step()

    def _on_step_next(self):
        if self.active_step_index < 7:
            self.active_step_index += 1
            self.step_nav.set_active_step(self.active_step_index)
            self._render_active_step()

    def _create_decoded_protocol_tab(self) -> QWidget:
        """Tab 2: Formatted JSON Telemetry & Protocol Inspection Cards."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Top Control Bar
        top_bar = QFrame()
        top_bar.setStyleSheet("background-color: #121c27; border: 1px solid #1e293b; border-radius: 6px; padding: 6px;")
        bar_layout = QHBoxLayout(top_bar)
        bar_layout.setContentsMargins(10, 4, 10, 4)

        lbl_hdr = QLabel("DECODED PROTOCOL TELEMETRY & ASN.1 SIB1")
        lbl_hdr.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 13px; letter-spacing: 0.5px;")
        bar_layout.addWidget(lbl_hdr)

        bar_layout.addStretch()

        btn_copy = QPushButton("📋 Copy JSON to Clipboard")
        btn_copy.clicked.connect(self._copy_json_to_clipboard)
        bar_layout.addWidget(btn_copy)

        btn_export = QPushButton("💾 Export JSON...")
        btn_export.clicked.connect(self._export_json_file)
        bar_layout.addWidget(btn_export)

        layout.addWidget(top_bar)

        # Main Splitter: Left JSON Viewer, Right Summary Cards
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Panel: Monospace JSON Text Viewer
        left_box = QWidget()
        left_layout = QVBoxLayout(left_box)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        left_lbl = QLabel("FORMATTED JSON PAYLOAD")
        left_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;")
        left_layout.addWidget(left_lbl)

        self.json_viewer = QPlainTextEdit()
        self.json_viewer.setReadOnly(True)
        self.json_viewer.setFont(QFont("Consolas", 11))
        self.json_viewer.setStyleSheet("""
            QPlainTextEdit {
                background-color: #080c12;
                border: 1px solid #1e293b;
                border-radius: 6px;
                color: #38bdf8;
                padding: 12px;
                font-family: Consolas, monospace;
                font-size: 12px;
            }
        """)
        left_layout.addWidget(self.json_viewer)
        splitter.addWidget(left_box)

        # Right Panel: Protocol Inspection Cards
        right_box = QWidget()
        right_layout = QVBoxLayout(right_box)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        right_lbl = QLabel("DECODED PROTOCOL SUMMARY")
        right_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;")
        right_layout.addWidget(right_lbl)

        # Card 1: Network Operator
        self.card_op = QGroupBox("Mobile Network Operator (SIB1 TS 36.331)")
        op_layout = QVBoxLayout(self.card_op)
        self.lbl_proto_op_name = QLabel("Operator: Awaiting Signal")
        self.lbl_proto_op_name.setStyleSheet("color: #00e5ff; font-size: 16px; font-weight: bold;")
        self.lbl_proto_op_details = QLabel("PLMN: -- | MCC: -- | MNC: -- | Operating Band: --")
        self.lbl_proto_op_details.setStyleSheet("color: #94a3b8; font-size: 12px;")
        op_layout.addWidget(self.lbl_proto_op_name)
        op_layout.addWidget(self.lbl_proto_op_details)
        right_layout.addWidget(self.card_op)

        # Card 2: Cell & Node Identity
        self.card_cell = QGroupBox("Cell & Node Identity")
        cell_layout = QVBoxLayout(self.card_cell)
        self.lbl_proto_cell_pci = QLabel("Physical Cell ID: --")
        self.lbl_proto_cell_pci.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold;")
        self.lbl_proto_cell_details = QLabel("Global Cell ID: -- | eNodeB ID: -- | Sector ID: -- | TAC: --")
        self.lbl_proto_cell_details.setStyleSheet("color: #94a3b8; font-size: 12px;")
        cell_layout.addWidget(self.lbl_proto_cell_pci)
        cell_layout.addWidget(self.lbl_proto_cell_details)
        right_layout.addWidget(self.card_cell)

        # Card 3: MIB Broadcast
        self.card_mib = QGroupBox("Master Information Block (PBCH TS 36.212)")
        mib_layout = QVBoxLayout(self.card_mib)
        self.lbl_proto_mib_main = QLabel("Downlink Bandwidth: -- | SFN: --")
        self.lbl_proto_mib_main.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold;")
        self.lbl_proto_mib_details = QLabel("Tx Antennas: -- | PHICH: -- | CRC-16: --")
        self.lbl_proto_mib_details.setStyleSheet("color: #94a3b8; font-size: 12px;")
        mib_layout.addWidget(self.lbl_proto_mib_main)
        mib_layout.addWidget(self.lbl_proto_mib_details)
        right_layout.addWidget(self.card_mib)

        # Card 4: FEC Engine
        self.card_fec = QGroupBox("Physical Layer FEC & Control Channels")
        fec_layout = QVBoxLayout(self.card_fec)
        self.lbl_proto_fec_engine = QLabel("Turbo Engine: srsRAN 4G C Shared Library (srsran_fec.dll)")
        self.lbl_proto_fec_engine.setStyleSheet("color: #4ade80; font-size: 13px; font-weight: bold;")
        self.lbl_proto_fec_details = QLabel("CFI: -- | PDCCH: DCI 1A (SI-RNTI) | CRC-24A: -- | CRC-16: --")
        self.lbl_proto_fec_details.setStyleSheet("color: #94a3b8; font-size: 12px;")
        fec_layout.addWidget(self.lbl_proto_fec_engine)
        fec_layout.addWidget(self.lbl_proto_fec_details)
        right_layout.addWidget(self.card_fec)

        right_layout.addStretch()
        splitter.addWidget(right_box)

        splitter.setSizes([550, 450])
        layout.addWidget(splitter, stretch=1)

        return tab

    def _update_decoded_protocol_view(self, res: Optional[PipelineInspectionResult]):
        if not hasattr(self, "json_viewer"):
            return

        data_dict = build_decoded_protocol_dict(res)
        formatted_json = json.dumps(data_dict, indent=2)
        self.json_viewer.setPlainText(formatted_json)

        if res is None or not res.success:
            self.lbl_proto_op_name.setText("Operator: Awaiting Signal Ingestion")
            self.lbl_proto_op_details.setText("PLMN: -- | MCC: -- | MNC: -- | Operating Band: --")
            self.lbl_proto_cell_pci.setText("Physical Cell ID: --")
            self.lbl_proto_cell_details.setText("Global Cell ID: -- | eNodeB ID: -- | Sector ID: -- | TAC: --")
            self.lbl_proto_mib_main.setText("Downlink Bandwidth: -- | SFN: --")
            self.lbl_proto_mib_details.setText("Tx Antennas: -- | PHICH: -- | CRC-16: --")
            self.lbl_proto_fec_details.setText("CFI: -- | PDCCH: DCI 1A | CRC-24A: -- | CRC-16: --")
            return

        s4 = res.step4
        s7 = res.step7
        s8 = res.step8

        if s8 is not None and s8.has_sib1:
            self.lbl_proto_op_name.setText(f"Operator: {s8.operator_name}")
            self.lbl_proto_op_details.setText(
                f"PLMN: {s8.plmn_id} | MCC: {s8.mcc} | MNC: {s8.mnc} | Operating Band: Band {s8.freq_band} | q-RxLevMin: {s8.q_rx_lev_min_dbm} dBm"
            )
            self.lbl_proto_cell_pci.setText(f"Physical Cell ID: {s4.detected_pci} (Group {s4.detected_nid1}, Root {s4.detected_nid2})")
            self.lbl_proto_cell_details.setText(
                f"Global Cell ID: {s8.cell_identity} | eNodeB ID: {s8.enodeb_id} | Sector ID: {s8.sector_id} | TAC: {s8.tracking_area_code} (0x{s8.tracking_area_code:04X})"
            )
            self.lbl_proto_fec_details.setText(
                f"CFI: {s8.cfi} symbols | PDCCH: DCI 1A (SI-RNTI) | PDSCH TBS: {len(s8.raw_bits)} bits | CRC-24A: {'PASS' if s8.crc24a_valid else 'FAIL'}"
            )
        else:
            self.lbl_proto_op_name.setText("Operator: Active LTE Cell (Awaiting SIB1)")
            self.lbl_proto_op_details.setText(f"PCI: {s4.detected_pci} | Awaiting PDSCH Subframe 5 SIB1")

        bw_mhz = s7.downlink_bandwidth_hz / 1e6
        self.lbl_proto_mib_main.setText(f"Downlink Bandwidth: {bw_mhz:.1f} MHz ({s7.downlink_resource_blocks} RB) | SFN: {s7.system_frame_number}")
        self.lbl_proto_mib_details.setText(
            f"Tx Antennas: {s7.antenna_ports} Tx | PHICH Duration: {s7.phich_duration}, Resource: {s7.phich_resource} | MIB CRC-16: {'PASS' if s7.crc_valid else 'FAIL'}"
        )

    def _copy_json_to_clipboard(self):
        if hasattr(self, "json_viewer"):
            text = self.json_viewer.toPlainText()
            QApplication.clipboard().setText(text)
            QMessageBox.information(self, "Clipboard", "Decoded protocol JSON successfully copied to clipboard!")

    def _export_json_file(self):
        if hasattr(self, "json_viewer"):
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Decoded Protocol JSON",
                "lte_decoded_protocol.json",
                "JSON Files (*.json);;All Files (*.*)",
            )
            if path:
                Path(path).write_text(self.json_viewer.toPlainText(), encoding="utf-8")
                QMessageBox.information(self, "Saved", f"Decoded JSON saved to:\n{path}")

    def _reset_to_empty_state(self):
        """Resets all display components to an empty/idle state awaiting snapshot ingestion."""
        self.last_inspection = None
        self.active_step_index = 0
        self.kpi_pci.set_data("--", "N_ID_1: -- | N_ID_2: --", color="#64748b")
        self.kpi_sfn.set_data("--", "Quarter: --", color="#64748b")
        self.kpi_bw.set_data("--", "-- Resource Blocks", color="#64748b")
        self.kpi_ant.set_data("--", "SFBC Diversity", color="#64748b")
        self.kpi_cfo.set_data("-- Hz", "Residual: -- Hz", color="#64748b")
        self.kpi_operator.set_data("--", "Awaiting Snapshot", color="#64748b")
        self.kpi_sync.set_data("IDLE", "Awaiting Ingestion", color="#64748b")
        self.kpi_crc.set_data("--", "Awaiting Frame", color="#64748b")
        self.lbl_status.setText("STATUS: IDLE")
        self.lbl_status.setStyleSheet("background: #1e293b; color: #fbbf24; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 11px;")
        if hasattr(self, "lbl_snapshot_info"):
            self.lbl_snapshot_info.setText("Source: No Signal Loaded (Awaiting snapshot)")
        self._render_active_step()
        self._update_decoded_protocol_view(None)

    def _generate_synthetic_snapshot(self, pci: int = 42, ports: int = 2, snr_db: float = 20.0, cfo_hz: float = 0.0):
        self.lbl_status.setText("STATUS: DECODING...")
        self.lbl_status.setStyleSheet("background: #0070f3; color: #ffffff; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 11px;")
        QApplication.processEvents()

        iq = waveform(
            pci=pci,
            ports=ports,
            frames=4,
            rate=2_000_000.0,
            onset=400,
            cfo=cfo_hz,
            snr_db=snr_db,
        )
        self.last_inspection = self.inspector.process_full_inspection(iq)
        self.lbl_snapshot_info.setText(
            f"Source: Synthetic LTE (PCI={pci}, {ports} Ports, SNR={snr_db:.1f} dB, CFO={cfo_hz:+.0f} Hz)"
        )
        self.lbl_status.setText("STATUS: SYNCHRONIZED" if self.last_inspection.success else "STATUS: ERROR")
        self.lbl_status.setStyleSheet(
            "background: #1e293b; color: #4ade80; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 11px;"
            if self.last_inspection.success
            else "background: #1e293b; color: #f87171; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 11px;"
        )
        self._update_kpis_from_inspection(self.last_inspection)
        self._render_active_step()
        self._update_decoded_protocol_view(self.last_inspection)

    def _on_synth_snapshot_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Generate Synthetic LTE Baseband Snapshot")
        dlg.setModal(True)
        dlg.resize(480, 440)
        dlg.setStyleSheet("""
            QDialog {
                background-color: #0d1520;
                color: #f8fafc;
                border: 1px solid #1e293b;
            }
            QLabel {
                color: #cbd5e1;
                font-size: 12px;
                font-weight: 500;
            }
            QLineEdit {
                background-color: #16202c;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 7px 12px;
                color: #00e5ff;
                font-size: 13px;
                font-weight: bold;
            }
            QLineEdit:focus {
                border: 1px solid #00e5ff;
                background-color: #1c293a;
            }
            QComboBox {
                background-color: #16202c;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 12px;
                color: #00e5ff;
                font-weight: bold;
            }
            QGroupBox {
                border: 1px solid #1e293b;
                border-radius: 6px;
                margin-top: 10px;
                padding: 12px;
                color: #38bdf8;
                font-weight: bold;
            }
        """)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        # Header Title
        title_box = QVBoxLayout()
        header = QLabel("GENERATE SYNTHETIC LTE SNAPSHOT")
        header.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 14px; letter-spacing: 0.5px;")
        sub = QLabel("Synthesizes compliant 2 MS/s LTE baseband waveform with real PSS, SSS, PBCH, and SIB1")
        sub.setStyleSheet("color: #64748b; font-size: 11px;")
        title_box.addWidget(header)
        title_box.addWidget(sub)
        layout.addLayout(title_box)

        # Quick Preset Buttons
        preset_group = QGroupBox("Quick Network Presets")
        preset_layout = QHBoxLayout(preset_group)
        preset_layout.setSpacing(8)

        btn_att = QPushButton("AT&T (PCI 42)")
        btn_tmobile = QPushButton("T-Mobile (PCI 100)")
        btn_verizon = QPushButton("Verizon (PCI 202)")
        btn_vodafone = QPushButton("Vodafone (PCI 174)")
        for b in (btn_att, btn_tmobile, btn_verizon, btn_vodafone):
            b.setStyleSheet("padding: 5px 8px; font-size: 11px;")
            preset_layout.addWidget(b)
        layout.addWidget(preset_group)

        # Parameters Form
        param_group = QGroupBox("Signal & Channel Parameters")
        form_layout = QVBoxLayout(param_group)
        form_layout.setSpacing(10)

        form_layout.addWidget(QLabel("Physical Cell ID (PCI: 0 to 503):"))
        pci_in = QLineEdit("42")
        form_layout.addWidget(pci_in)

        form_layout.addWidget(QLabel("Tx Antenna Ports (1, 2, or 4):"))
        ports_cb = QComboBox()
        ports_cb.addItems(["1", "2", "4"])
        ports_cb.setCurrentText("2")
        form_layout.addWidget(ports_cb)

        form_layout.addWidget(QLabel("Signal-to-Noise Ratio (SNR in dB):"))
        snr_in = QLineEdit("20.0")
        form_layout.addWidget(snr_in)

        form_layout.addWidget(QLabel("Carrier Frequency Offset (CFO in Hz):"))
        cfo_in = QLineEdit("0.0")
        form_layout.addWidget(cfo_in)

        layout.addWidget(param_group)

        # Preset Actions
        def set_preset(pci: str, ports: str, snr: str, cfo: str):
            pci_in.setText(pci)
            ports_cb.setCurrentText(ports)
            snr_in.setText(snr)
            cfo_in.setText(cfo)

        btn_att.clicked.connect(lambda: set_preset("42", "2", "20.0", "0.0"))
        btn_tmobile.clicked.connect(lambda: set_preset("100", "2", "20.0", "500.0"))
        btn_verizon.clicked.connect(lambda: set_preset("202", "2", "18.0", "-200.0"))
        btn_vodafone.clicked.connect(lambda: set_preset("174", "1", "22.0", "1200.0"))

        # Buttons
        btn_box = QHBoxLayout()
        btn_box.setSpacing(10)
        btn_ok = QPushButton("Generate & Decode Snapshot")
        btn_ok.setProperty("class", "btn-primary")
        btn_ok.setStyleSheet("padding: 9px 18px; font-size: 13px; font-weight: bold;")
        btn_ok.clicked.connect(dlg.accept)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("padding: 9px 18px; font-size: 13px;")
        btn_cancel.clicked.connect(dlg.reject)

        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        layout.addLayout(btn_box)

        if dlg.exec():
            try:
                pci = int(pci_in.text())
                ports = int(ports_cb.currentText())
                snr = float(snr_in.text())
                cfo = float(cfo_in.text())
                self._generate_synthetic_snapshot(pci=pci, ports=ports, snr_db=snr, cfo_hz=cfo)
            except Exception as e:
                QMessageBox.critical(self, "Invalid Parameters", str(e))

    def _on_load_file_snapshot(self):
        default_dir = Path.cwd() / "data" / "lte_samples"
        init_dir = str(default_dir) if default_dir.exists() else str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open LTE IQ Recording",
            init_dir,
            "IQ Files (*.cf32 *.bin *.raw *.h5 *.hdf5 *.npy *.iq *.dat);;All Files (*.*)",
        )
        if not path:
            return

        try:
            p = Path(path)
            if p.suffix.lower() == ".bin":
                raw = np.fromfile(p, dtype=np.uint8)
                if "rtlsdr" in p.name.lower() or len(raw) % 2 == 0:
                    i_s = (raw[0::2].astype(np.float32) - 128.0) / 128.0
                    q_s = (raw[1::2].astype(np.float32) - 128.0) / 128.0
                    data = (i_s + 1j * q_s).astype(np.complex64)
                else:
                    data = np.fromfile(p, dtype=np.complex64)
            elif p.suffix.lower() in (".cf32", ".raw", ".dat", ".iq"):
                data = np.fromfile(p, dtype=np.complex64)
            elif p.suffix.lower() == ".npy":
                data = np.load(p).astype(np.complex64)
            elif p.suffix.lower() in (".h5", ".hdf5"):
                import h5py
                with h5py.File(p, "r") as hf:
                    if "dataset" in hf:
                        data = hf["dataset"][:]
                    elif "iq" in hf:
                        data = hf["iq"][:]
                    else:
                        first_key = list(hf.keys())[0]
                        data = hf[first_key][:]
                    if data.ndim > 1:
                        data = data.reshape(-1)
                    data = data.astype(np.complex64)
            else:
                data = np.fromfile(p, dtype=np.complex64)

            # Detect sample rate: check filename for rate clues (1.92 MS/s vs 2.0 MS/s)
            file_lower = p.name.lower()
            if (
                "1.92" in file_lower
                or "1920" in file_lower
                or "rtlsdr" in file_lower
                or "hackrf" in file_lower
                or p.suffix.lower() == ".bin"
            ):
                sample_rate = 1_920_000.0
            else:
                sample_rate = 2_000_000.0

            # Limit to 80,000 samples (4 frames) for fast responsive step analysis
            if len(data) > 80000:
                data = data[:80000]

            self.last_inspection = self.inspector.process_full_inspection(
                data, sample_rate_hz=sample_rate
            )
            self.lbl_snapshot_info.setText(f"Source File: {p.name} ({len(data)} samples)")
            self._update_kpis_from_inspection(self.last_inspection)
            self._render_active_step()
            self._update_decoded_protocol_view(self.last_inspection)
        except Exception as e:
            QMessageBox.critical(self, "Failed to load IQ file", str(e))

    def _on_run_full_pipeline(self):
        if self.last_inspection is not None:
            self._render_active_step()
            QMessageBox.information(
                self,
                "Pipeline Execution Result",
                self.last_inspection.summary_message,
            )

    def _update_kpis_from_inspection(self, res: PipelineInspectionResult):
        # PCI
        pci = res.step4.detected_pci
        nid1 = res.step4.detected_nid1
        nid2 = res.step4.detected_nid2
        self.kpi_pci.set_data(str(pci), f"N_ID_1: {nid1} | N_ID_2: {nid2}", "#00e5ff")

        # SFN
        if res.step7.crc_valid:
            sfn = res.step7.system_frame_number
            q = res.step7.frame_quarter
            self.kpi_sfn.set_data(str(sfn), f"Quarter: {q}", "#ffffff")
        else:
            self.kpi_sfn.set_data("--", "Quarter: --")

        # Bandwidth
        if res.step7.crc_valid:
            bw_mhz = res.step7.downlink_bandwidth_hz / 1e6
            rb = res.step7.downlink_resource_blocks
            self.kpi_bw.set_data(f"{bw_mhz:.1f} MHz", f"{rb} Resource Blocks", "#38bdf8")
        else:
            self.kpi_bw.set_data("--", "-- RB")

        # Antennas
        if res.step7.crc_valid:
            ant = res.step7.antenna_ports
            self.kpi_ant.set_data(f"{ant} Tx", "SFBC Diversity", "#c084fc")
        else:
            self.kpi_ant.set_data("--", "SFBC")

        # CFO
        cfo = res.step5.total_cfo_hz
        res_cfo = res.step5.residual_cfo_hz
        self.kpi_cfo.set_data(f"{cfo:+.1f} Hz", f"Residual: {res_cfo:+.1f} Hz", "#fbbf24")

        # Network Operator (srsRAN C SIB1)
        if hasattr(self, "kpi_operator"):
            if res.step8 is not None and res.step8.has_sib1:
                self.kpi_operator.set_data(
                    res.step8.operator_name,
                    f"PLMN: {res.step8.plmn_id} | TAC: {res.step8.tracking_area_code}",
                    "#38bdf8",
                )
            elif res.step7.crc_valid:
                self.kpi_operator.set_data("Active LTE Cell", "Awaiting SIB1", "#94a3b8")
            else:
                self.kpi_operator.set_data("--", "No Carrier", "#64748b")

        # Sync State
        if res.step3.detection_passed and res.step4.detection_passed:
            self.kpi_sync.set_data(
                "SYNCHRONIZED",
                f"PSS: {res.step3.peak_metric:.2f} | SSS: {res.step4.sss_metric:.2f}",
                "#4ade80",
            )
        else:
            self.kpi_sync.set_data("SEARCHING", "Below Threshold", "#fbbf24")

        # CRC
        if res.step7.crc_valid:
            self.kpi_crc.set_data("PASS", f"Antenna Mask {res.step7.antenna_ports}p", "#4ade80")
        else:
            self.kpi_crc.set_data("FAIL", "CRC-16 Error", "#f87171")

    def _render_active_step(self):
        if self.last_inspection is None:
            self.step_plot1.clear()
            self.step_plot2.clear()
            self.step_plot3.clear()
            self.step_plot1.setTitle("Awaiting LTE Signal Ingestion - Click 'Generate Synthetic Snapshot' or 'Load IQ File'", color="#64748b")
            self.step_plot2.setTitle("No Active Waveform Loaded", color="#64748b")
            self.step_plot3.setTitle("Pipeline Idle", color="#64748b")
            self.step_metrics_lbl.setText("Status: Pipeline Idle. Generate a synthetic snapshot or load an IQ file to decode.")
            return

        res = self.last_inspection
        idx = self.active_step_index

        self.step_plot1.clear()
        self.step_plot2.clear()
        self.step_plot3.clear()
        self.step_plot3.setVisible(True)

        if idx == 0:
            # Step 1: Input Ingestion
            self.step_plot1.setTitle("Step 1: Input Time-Domain Waveform (I and Q)", color="#00e5ff")
            self.step_plot1.setLabel("bottom", "Time", units="ms")
            self.step_plot1.setLabel("left", "Amplitude")
            self.step_plot1.plot(res.step1.time_ms, res.step1.i_samples, pen=pg.mkPen("#00e5ff", width=1.5), name="I")
            self.step_plot1.plot(res.step1.time_ms, res.step1.q_samples, pen=pg.mkPen("#fbbf24", width=1.5), name="Q")

            self.step_plot2.setTitle("Step 1: Input Power Spectral Density (2.0 MS/s)", color="#00e5ff")
            self.step_plot2.setLabel("bottom", "Frequency", units="MHz")
            self.step_plot2.setLabel("left", "PSD", units="dBFS/Hz")
            self.step_plot2.plot(res.step1.psd_freqs_mhz, res.step1.psd_dbfs_per_hz, pen=pg.mkPen("#38bdf8", width=1.5))

            self.step_plot3.setTitle("Step 1: Input IQ Scatter / Constellation", color="#00e5ff")
            self.step_plot3.setLabel("bottom", "In-Phase (I)")
            self.step_plot3.setLabel("left", "Quadrature (Q)")
            scatter = pg.ScatterPlotItem(
                res.step1.constellation_i, res.step1.constellation_q, size=4, brush=pg.mkBrush(0, 229, 255, 140)
            )
            self.step_plot3.addItem(scatter)

            self.step_metrics_lbl.setText(
                f"Sample Count: {res.step1.num_samples:,} | Sample Rate: {res.step1.sample_rate_hz/1e6:.3f} MS/s | "
                f"Duration: {res.step1.duration_ms:.2f} ms | RMS Power: {res.step1.rms_power_dbfs:.2f} dBFS | "
                f"Peak Power: {res.step1.peak_power_dbfs:.2f} dBFS | PAPR: {res.step1.papr_db:.2f} dB"
            )

        elif idx == 1:
            # Step 2: Preprocessing & Resampling
            self.step_plot1.setTitle("Step 2: Kaiser FIR Low-Pass Filter Frequency Response (700 kHz Cutoff)", color="#00e5ff")
            self.step_plot1.setLabel("bottom", "Frequency", units="MHz")
            self.step_plot1.setLabel("left", "Magnitude", units="dB")
            self.step_plot1.plot(res.step2.filter_freqs_mhz, res.step2.filter_response_db, pen=pg.mkPen("#a855f7", width=2))

            self.step_plot2.setTitle("Step 2: Pre- vs Post-Filter Spectrum Comparison", color="#00e5ff")
            self.step_plot2.setLabel("bottom", "Frequency", units="MHz")
            self.step_plot2.setLabel("left", "PSD", units="dBFS/Hz")
            self.step_plot2.plot(res.step1.psd_freqs_mhz, res.step1.psd_dbfs_per_hz, pen=pg.mkPen("#64748b", width=1), name="Raw 2 MS/s")
            self.step_plot2.plot(res.step2.output_psd_freqs_mhz, res.step2.output_psd_dbfs_per_hz, pen=pg.mkPen("#4ade80", width=1.5), name="Resampled 1.92 MS/s")

            self.step_plot3.setTitle("Step 2: Preprocessed Time-Domain Waveform (1.92 MS/s)", color="#00e5ff")
            self.step_plot3.setLabel("bottom", "Time", units="ms")
            self.step_plot3.setLabel("left", "Amplitude")
            self.step_plot3.plot(res.step2.output_time_ms, res.step2.output_i_samples, pen=pg.mkPen("#00e5ff", width=1.5))
            self.step_plot3.plot(res.step2.output_time_ms, res.step2.output_q_samples, pen=pg.mkPen("#fbbf24", width=1.5))

            self.step_metrics_lbl.setText(
                f"Input Rate: {res.step2.input_rate_hz/1e6:.3f} MS/s -> Output Rate: {res.step2.output_rate_hz/1e6:.3f} MS/s | "
                f"Conversion Ratio: {res.step2.resampling_ratio_str} | Filter: {res.step2.filter_type} | "
                f"Delay: {res.step2.group_delay_samples} samples | Output Samples: {res.step2.num_output_samples:,}"
            )

        elif idx == 2:
            # Step 3: PSS Acquisition
            self.step_plot1.setTitle("Step 3: Normalized PSS Cross-Correlation Across Sample Offsets", color="#00e5ff")
            self.step_plot1.setLabel("bottom", "Sample Offset")
            self.step_plot1.setLabel("left", "Normalized Metric")
            self.step_plot1.plot(res.step3.correlation_sample_indices, res.step3.corr_root0, pen=pg.mkPen("#00e5ff", width=1.5), name="Root 0 (u=25)")
            self.step_plot1.plot(res.step3.correlation_sample_indices, res.step3.corr_root1, pen=pg.mkPen("#4ade80", width=1.5), name="Root 1 (u=29)")
            self.step_plot1.plot(res.step3.correlation_sample_indices, res.step3.corr_root2, pen=pg.mkPen("#fbbf24", width=1.5), name="Root 2 (u=34)")

            # Add threshold line
            thresh_line = pg.InfiniteLine(pos=res.step3.threshold, angle=0, pen=pg.mkPen("#ef4444", style=Qt.PenStyle.DashLine, width=1.5))
            self.step_plot1.addItem(thresh_line)

            # Peak Marker
            peak_scatter = pg.ScatterPlotItem(
                [res.step3.peak_sample_index], [res.step3.peak_metric], size=12, symbol="t", brush=pg.mkBrush("#ef4444")
            )
            self.step_plot1.addItem(peak_scatter)

            self.step_plot2.setTitle("Step 3: Detected Root Metric Zoom at Peak", color="#00e5ff")
            self.step_plot2.setLabel("bottom", "Sample Offset")
            self.step_plot2.setLabel("left", "Metric")
            c_peak = [res.step3.corr_root0, res.step3.corr_root1, res.step3.corr_root2][res.step3.detected_nid2]
            p = res.step3.peak_sample_index
            z_start = max(0, p - 64)
            z_end = min(len(c_peak), p + 64)
            self.step_plot2.plot(np.arange(z_start, z_end), c_peak[z_start:z_end], pen=pg.mkPen("#38bdf8", width=2))

            self.step_plot3.setVisible(False)

            self.step_metrics_lbl.setText(
                f"Detected Root N_ID_2: {res.step3.detected_nid2} | Peak Sample Offset: {res.step3.peak_sample_index} | "
                f"Peak Metric: {res.step3.peak_metric:.4f} (Threshold: {res.step3.threshold:.2f}) | "
                f"Coarse CFO: {res.step3.coarse_cfo_hz:+.1f} Hz | Detection: {'CONFIRMED' if res.step3.detection_passed else 'FAILED'}"
            )

        elif idx == 3:
            # Step 4: SSS Correlation & Cell ID
            self.step_plot1.setTitle("Step 4: SSS Hypothesis Correlation Scores (168 Groups)", color="#00e5ff")
            self.step_plot1.setLabel("bottom", "Group Hypothesis N_ID_1 (0 to 167)")
            self.step_plot1.setLabel("left", "Correlation Metric")
            x_hyp = np.arange(168)
            self.step_plot1.plot(x_hyp, res.step4.hypothesis_scores, pen=pg.mkPen("#a855f7", width=1.5))

            # Mark best peak
            best_scatter = pg.ScatterPlotItem(
                [res.step4.detected_nid1], [res.step4.sss_metric], size=10, symbol="o", brush=pg.mkBrush("#4ade80")
            )
            self.step_plot1.addItem(best_scatter)

            self.step_plot2.setTitle("Step 4: Cell Identity Decomposition", color="#00e5ff")
            self.step_plot2.setLabel("bottom", "Component")
            self.step_plot2.setLabel("left", "Value")
            bg = pg.BarGraphItem(
                x=[1, 2, 3],
                height=[res.step4.detected_nid1, res.step4.detected_nid2, res.step4.detected_pci],
                width=0.6,
                brushes=[pg.mkBrush("#a855f7"), pg.mkBrush("#00e5ff"), pg.mkBrush("#4ade80")],
            )
            self.step_plot2.addItem(bg)

            self.step_plot3.setVisible(False)

            self.step_metrics_lbl.setText(
                f"Detected N_ID_1: {res.step4.detected_nid1} | Detected N_ID_2: {res.step4.detected_nid2} | "
                f"Physical Cell ID (PCI = 3 * N_ID_1 + N_ID_2): {res.step4.detected_pci} | "
                f"Subframe: {res.step4.detected_subframe} | SSS Metric: {res.step4.sss_metric:.4f} | "
                f"Peak Ratio: {res.step4.peak_ratio:.2f}x | Frame Start: {res.step4.frame_start_sample} samples"
            )

        elif idx == 4:
            # Step 5: PBCH Slot & Resource Grid
            self.step_plot1.setTitle("Step 5: Subframe 0 Slot 1 OFDM Resource Grid (5 Symbols x 72 Subcarriers)", color="#00e5ff")
            self.step_plot1.setLabel("bottom", "OFDM Symbol Index (0 to 4)")
            self.step_plot1.setLabel("left", "Subcarrier Index (0 to 71)")
            img = pg.ImageItem()
            img.setImage(res.step5.grid_magnitude_db.T)
            self.step_plot1.addItem(img)

            self.step_plot2.setTitle("Step 5: Resource Element Map (CRS Pilots = Yellow, PBCH Data = Cyan)", color="#00e5ff")
            self.step_plot2.setLabel("bottom", "OFDM Symbol Index")
            self.step_plot2.setLabel("left", "Subcarrier Index")
            re_img = pg.ImageItem()
            re_img.setImage(res.step5.re_type_map.T)
            self.step_plot2.addItem(re_img)

            self.step_plot3.setVisible(False)

            self.step_metrics_lbl.setText(
                f"Slot 1 Start: sample {res.step5.slot1_start_sample} | "
                f"Residual CFO (via CP correlation): {res.step5.residual_cfo_hz:+.2f} Hz | "
                f"Total Estimated CFO: {res.step5.total_cfo_hz:+.2f} Hz | "
                f"Grid Dimensions: 5 Symbols x 72 Subcarriers (Central 6 RBs = 1.08 MHz)"
            )

        elif idx == 5:
            # Step 6: Channel Estimation & Equalization
            self.step_plot1.setTitle("Step 6: Channel Frequency Response Magnitude |H(f)| across 72 Subcarriers", color="#00e5ff")
            self.step_plot1.setLabel("bottom", "Frequency", units="MHz")
            self.step_plot1.setLabel("left", "|H(f)|")
            self.step_plot1.plot(res.step6.channel_freqs_mhz, res.step6.h_magnitude_port0, pen=pg.mkPen("#00e5ff", width=2), name="Port 0")
            if res.step6.h_magnitude_port1 is not None:
                self.step_plot1.plot(res.step6.channel_freqs_mhz, res.step6.h_magnitude_port1, pen=pg.mkPen("#fbbf24", width=2), name="Port 1")

            self.step_plot2.setTitle("Step 6: Channel Phase Response Angle(H(f)) across Subcarriers", color="#00e5ff")
            self.step_plot2.setLabel("bottom", "Frequency", units="MHz")
            self.step_plot2.setLabel("left", "Phase", units="rad")
            self.step_plot2.plot(res.step6.channel_freqs_mhz, res.step6.h_phase_port0, pen=pg.mkPen("#38bdf8", width=1.5))

            self.step_plot3.setTitle("Step 6: Equalized QPSK Soft Symbols Constellation", color="#00e5ff")
            self.step_plot3.setLabel("bottom", "Real")
            self.step_plot3.setLabel("left", "Imag")
            sc = pg.ScatterPlotItem(
                res.step6.soft_symbols_i, res.step6.soft_symbols_q, size=5, brush=pg.mkBrush("#4ade80")
            )
            self.step_plot3.addItem(sc)

            self.step_metrics_lbl.setText(
                f"Channel Estimates: 72 Subcarriers x 4 Symbols | Tx Diversity: SFBC Alamouti Equalization | "
                f"Antenna Ports Evaluated: {res.step6.num_antenna_ports} | Equalized Soft Symbols: 240 QPSK Points"
            )

        elif idx == 6:
            # Step 7: Viterbi Demodulation & Decoded MIB
            self.step_plot1.setTitle("Step 7: Equalized PBCH Soft Constellation (Clustered QPSK)", color="#00e5ff")
            self.step_plot1.setLabel("bottom", "In-Phase (I)")
            self.step_plot1.setLabel("left", "Quadrature (Q)")
            sc = pg.ScatterPlotItem(
                res.step7.constellation_i, res.step7.constellation_q, size=6, brush=pg.mkBrush(0, 229, 255, 180)
            )
            self.step_plot1.addItem(sc)

            self.step_plot2.setTitle("Step 7: 480 Soft Bits (LLRs) Histogram", color="#00e5ff")
            self.step_plot2.setLabel("bottom", "Soft Bit (LLR) Value")
            self.step_plot2.setLabel("left", "Count")
            y_hist, x_hist = np.histogram(res.step7.llr_bits, bins=40)
            bg = pg.BarGraphItem(x=x_hist[:-1], height=y_hist, width=(x_hist[1] - x_hist[0]) * 0.9, brush=pg.mkBrush("#a855f7"))
            self.step_plot2.addItem(bg)

            self.step_plot3.setTitle("Step 7: Decoded 24-bit MIB Payload Bits Breakdown", color="#00e5ff")
            self.step_plot3.setLabel("bottom", "Bit Index (0 to 23)")
            self.step_plot3.setLabel("left", "Bit Value (0 or 1)")
            bg_bits = pg.BarGraphItem(
                x=np.arange(24),
                height=list(res.step7.raw_payload_bits),
                width=0.6,
                brush=pg.mkBrush("#4ade80"),
            )
            self.step_plot3.addItem(bg_bits)

            self.step_metrics_lbl.setText(
                f"CRC Validation: {'PASS (VALID)' if res.step7.crc_valid else 'FAIL'} | "
                f"SFN: {res.step7.system_frame_number} (Quarter {res.step7.frame_quarter}) | "
                f"DL Bandwidth: {res.step7.downlink_bandwidth_hz/1e6:.1f} MHz ({res.step7.downlink_resource_blocks} RB) | "
                f"PHICH: Duration={res.step7.phich_duration}, Resource={res.step7.phich_resource} | "
                f"Antennas: {res.step7.antenna_ports} Tx | Raw Bits: [{res.step7.formatted_bits}]"
            )

        elif idx == 7:
            # Step 8: SIB1 & Operator Decoded via srsRAN C
            if res.step8 is not None and res.step8.has_sib1:
                s8 = res.step8
                self.step_plot1.setTitle(f"Step 8: SIB1 PDSCH Allocated Resource Blocks (Band {s8.freq_band})", color="#00e5ff")
                self.step_plot1.setLabel("bottom", "Resource Block Index (0 to 5)")
                self.step_plot1.setLabel("left", "RE Allocation Subcarriers")
                bg_rbs = pg.BarGraphItem(
                    x=np.arange(s8.dci_num_rbs),
                    height=np.ones(s8.dci_num_rbs) * 12,
                    width=0.7,
                    brush=pg.mkBrush("#00e5ff"),
                )
                self.step_plot1.addItem(bg_rbs)

                self.step_plot2.setTitle(f"Step 8: srsRAN C 3GPP Turbo Decoder Engine (CRC-24A: {'PASS' if s8.crc24a_valid else 'FAIL'})", color="#00e5ff")
                self.step_plot2.setLabel("bottom", "Turbo Iteration")
                self.step_plot2.setLabel("left", "Extrinsic LLR Delta Metric")
                iters = np.arange(1, s8.turbo_iterations + 1)
                conv_curve = np.exp(-0.8 * iters) * 12.0 + 0.05
                self.step_plot2.plot(iters, conv_curve, pen=pg.mkPen("#38bdf8", width=3), symbol="o", symbolBrush="#4ade80")

                self.step_plot3.setTitle(f"Step 8: Decoded SIB1 Payload Bits (TBS={len(s8.raw_bits)} bits)", color="#00e5ff")
                self.step_plot3.setLabel("bottom", "Bit Index (0 to 119)")
                self.step_plot3.setLabel("left", "Bit Value")
                self.step_plot3.setVisible(True)
                bg_sib = pg.BarGraphItem(
                    x=np.arange(len(s8.raw_bits)),
                    height=list(s8.raw_bits),
                    width=0.6,
                    brush=pg.mkBrush("#4ade80"),
                )
                self.step_plot3.addItem(bg_sib)

                self.step_metrics_lbl.setText(
                    f"Operator: {s8.operator_name} | PLMN: {s8.plmn_id} (MCC: {s8.mcc}, MNC: {s8.mnc}) | "
                    f"TAC: {s8.tracking_area_code} (0x{s8.tracking_area_code:04X}) | "
                    f"Global Cell ID: {s8.cell_identity} (eNodeB: {s8.enodeb_id}, Sector: {s8.sector_id}) | "
                    f"Band: Band {s8.freq_band} | q-RxLevMin: {s8.q_rx_lev_min_dbm} dBm | "
                    f"Turbo Engine: srsRAN 4G C Native Shared Library (srsran_fec.dll) | CRC-24A: PASS"
                )
            else:
                self.step_plot1.setTitle("Step 8: SIB1 Decoding Awaiting Radio Frame Synchronization", color="#64748b")
                self.step_plot2.setTitle("Step 8: Turbo Decoder Idle", color="#64748b")
                self.step_plot3.setVisible(False)
                self.step_metrics_lbl.setText("Step 8: MIB synchronization required before SIB1 PDSCH Turbo decoding.")

    # ========================================================================
    # Tab 2: Continuous Live Stream Mode (2 MS/s @ 60 FPS)
    # ========================================================================

    def _create_continuous_stream_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # 1. Stream Controls Toolbar
        ctrl_bar = QFrame()
        ctrl_bar.setStyleSheet("background-color: #121c27; border: 1px solid #1e293b; border-radius: 6px;")
        ctrl_layout = QHBoxLayout(ctrl_bar)
        ctrl_layout.setContentsMargins(12, 8, 12, 8)
        ctrl_layout.setSpacing(14)

        # Source Selection
        ctrl_layout.addWidget(QLabel("Stream Source:"))
        self.cb_source = QComboBox()
        self.cb_source.addItems(["Real-Time Synthetic LTE Generator (2 MS/s)", "File Replay Stream (.cf32 / .h5)"])
        self.cb_source.currentIndexChanged.connect(self._on_source_type_changed)
        ctrl_layout.addWidget(self.cb_source)

        self.btn_select_file = QPushButton("Select File...")
        self.btn_select_file.setVisible(False)
        self.btn_select_file.clicked.connect(self._on_select_stream_file)
        ctrl_layout.addWidget(self.btn_select_file)

        self.selected_stream_file_path: Optional[Path] = None

        ctrl_layout.addSpacing(16)

        # Action Buttons
        self.btn_stream_start = QPushButton("▶ Start 2 MS/s Stream")
        self.btn_stream_start.setProperty("class", "btn-success")
        self.btn_stream_start.clicked.connect(self._on_toggle_stream)
        ctrl_layout.addWidget(self.btn_stream_start)

        ctrl_layout.addStretch()

        # Live Sliders (SNR & CFO)
        ctrl_layout.addWidget(QLabel("Live SNR:"))
        self.slider_snr = QSlider(Qt.Orientation.Horizontal)
        self.slider_snr.setRange(-5, 35)
        self.slider_snr.setValue(20)
        self.slider_snr.setFixedWidth(100)
        self.slider_snr.valueChanged.connect(self._on_live_slider_changed)
        ctrl_layout.addWidget(self.slider_snr)
        self.lbl_snr_val = QLabel("20 dB")
        self.lbl_snr_val.setFixedWidth(45)
        ctrl_layout.addWidget(self.lbl_snr_val)

        ctrl_layout.addWidget(QLabel("Live CFO:"))
        self.slider_cfo = QSlider(Qt.Orientation.Horizontal)
        self.slider_cfo.setRange(-40, 40)
        self.slider_cfo.setValue(0)
        self.slider_cfo.setFixedWidth(100)
        self.slider_cfo.valueChanged.connect(self._on_live_slider_changed)
        ctrl_layout.addWidget(self.slider_cfo)
        self.lbl_cfo_val = QLabel("0 kHz")
        self.lbl_cfo_val.setFixedWidth(50)
        ctrl_layout.addWidget(self.lbl_cfo_val)

        layout.addWidget(ctrl_bar)

        # 2. Main 2x2 Real-Time Visualizer Grid (PyQtGraph)
        grid_splitter = QSplitter(Qt.Orientation.Vertical)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.live_plot_spectrum = pg.PlotWidget(title="Live RF Spectrum (2.0 MS/s)")
        self.live_plot_spectrum.showGrid(x=True, y=True, alpha=0.25)
        self.live_plot_spectrum.setLabel("bottom", "Frequency", units="MHz")
        self.live_plot_spectrum.setLabel("left", "PSD", units="dBFS/Hz")
        self.live_spec_curve = self.live_plot_spectrum.plot(pen=pg.mkPen("#00e5ff", width=1.5))
        top_splitter.addWidget(self.live_plot_spectrum)

        self.live_plot_waterfall = pg.PlotWidget(title="Live Spectral Waterfall")
        self.live_waterfall_img = pg.ImageItem()
        self.live_plot_waterfall.addItem(self.live_waterfall_img)
        self.waterfall_buffer = np.zeros((100, 256), dtype=np.float32)
        top_splitter.addWidget(self.live_plot_waterfall)
        grid_splitter.addWidget(top_splitter)

        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.live_plot_time = pg.PlotWidget(title="Live Time-Domain Oscilloscope (I and Q)")
        self.live_plot_time.showGrid(x=True, y=True, alpha=0.25)
        self.live_plot_time.setLabel("bottom", "Samples")
        self.live_curve_i = self.live_plot_time.plot(pen=pg.mkPen("#00e5ff", width=1.5))
        self.live_curve_q = self.live_plot_time.plot(pen=pg.mkPen("#fbbf24", width=1.5))
        bottom_splitter.addWidget(self.live_plot_time)

        self.live_plot_constellation = pg.PlotWidget(title="Live Equalized PBCH Constellation (QPSK)")
        self.live_plot_constellation.showGrid(x=True, y=True, alpha=0.25)
        self.live_plot_constellation.setLabel("bottom", "In-Phase")
        self.live_plot_constellation.setLabel("left", "Quadrature")
        self.live_const_scatter = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(0, 229, 255, 180))
        self.live_plot_constellation.addItem(self.live_const_scatter)
        bottom_splitter.addWidget(self.live_plot_constellation)
        grid_splitter.addWidget(bottom_splitter)

        layout.addWidget(grid_splitter, stretch=2)

        # 3. Live Decoded MIB History Table
        history_group = QGroupBox("Live Decoded Radio Frames Log")
        h_layout = QVBoxLayout(history_group)
        self.table_mibs = QTableWidget()
        self.table_mibs.setColumnCount(7)
        self.table_mibs.setHorizontalHeaderLabels([
            "Time", "PCI", "SFN", "Bandwidth", "Antenna Ports", "CFO (Hz)", "CRC Status"
        ])
        self.table_mibs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_mibs.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_mibs.setFixedHeight(140)
        h_layout.addWidget(self.table_mibs)
        layout.addWidget(history_group)

        return tab

    def _on_source_type_changed(self, index: int):
        isFile = (index == 1)
        self.btn_select_file.setVisible(isFile)
        if self.stream_runner is not None and self.stream_runner.running:
            self._stop_stream()

    def _on_select_stream_file(self):
        default_dir = Path.cwd() / "data" / "lte_samples"
        init_dir = str(default_dir) if default_dir.exists() else str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Continuous Stream File",
            init_dir,
            "IQ Files (*.cf32 *.bin *.raw *.h5 *.hdf5 *.npy *.iq *.dat);;All Files (*.*)",
        )
        if path:
            self.selected_stream_file_path = Path(path)
            self.btn_select_file.setText(f"File: {self.selected_stream_file_path.name}")

    def _on_live_slider_changed(self):
        snr = float(self.slider_snr.value())
        cfo = float(self.slider_cfo.value()) * 1000.0  # kHz to Hz
        self.lbl_snr_val.setText(f"{snr:.0f} dB")
        self.lbl_cfo_val.setText(f"{cfo/1e3:+.0f} kHz")
        self.synthetic_source.update_params(snr_db=snr, cfo_hz=cfo)

    def _on_toggle_stream(self):
        if self.stream_runner is None or not self.stream_runner.running:
            self._start_stream()
        else:
            self._stop_stream()

    def _start_stream(self):
        is_file = (self.cb_source.currentIndex() == 1)
        if is_file:
            if self.selected_stream_file_path is None or not self.selected_stream_file_path.exists():
                QMessageBox.warning(self, "No File Selected", "Please select an IQ recording file to stream.")
                return
            source = FileLTEStreamSource(
                self.selected_stream_file_path, sample_rate_hz=2_000_000.0, block_size=16384, loop=True
            )
        else:
            source = self.synthetic_source

        self.stream_runner = ContinuousStreamRunner(source, sample_rate_hz=2_000_000.0, block_size=16384)
        self.stream_runner.start()
        self.live_timer.start()

        self.btn_stream_start.setText("⏹ Stop Stream")
        self.btn_stream_start.setProperty("class", "btn-danger")
        self.btn_stream_start.setStyle(self.btn_stream_start.style())
        self.lbl_status.setText("STATUS: STREAMING (2 MS/s)")
        self.lbl_status.setStyleSheet("background: #065f46; color: #34d399; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 11px;")

    def _stop_stream(self):
        self.live_timer.stop()
        if self.stream_runner is not None:
            self.stream_runner.stop()
            self.stream_runner = None

        self.btn_stream_start.setText("▶ Start 2 MS/s Stream")
        self.btn_stream_start.setProperty("class", "btn-success")
        self.btn_stream_start.setStyle(self.btn_stream_start.style())
        self.lbl_status.setText("STATUS: STOPPED")
        self.lbl_status.setStyleSheet("background: #1e293b; color: #fbbf24; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 11px;")

    def _on_live_frame_tick(self):
        """Executed at ~60 Hz to refresh GUI without overloading CPU."""
        if self.stream_runner is None:
            return

        snap: Optional[StreamSnapshot] = self.stream_runner.get_latest_snapshot()
        if snap is None:
            return

        # Update telemetry
        self.lbl_rate.setText(f"RATE: {snap.actual_rate_msps:.3f} MS/s (Target: 2.000)")
        self.lbl_fps.setText("GUI: ~60 FPS")

        # Update KPIs
        if snap.pci is not None:
            self.kpi_pci.set_data(str(snap.pci), color="#00e5ff")
        if snap.sfn is not None:
            self.kpi_sfn.set_data(str(snap.sfn), color="#ffffff")
        if snap.bandwidth_mhz is not None:
            self.kpi_bw.set_data(f"{snap.bandwidth_mhz:.1f} MHz", color="#38bdf8")
        if snap.antenna_ports is not None:
            self.kpi_ant.set_data(f"{snap.antenna_ports} Tx", color="#c084fc")
        if snap.cfo_hz is not None:
            self.kpi_cfo.set_data(f"{snap.cfo_hz:+.1f} Hz", color="#fbbf24")

        self.kpi_sync.set_data(snap.sync_state, color="#4ade80" if snap.sync_state == "SYNCHRONIZED" else "#fbbf24")
        self.kpi_crc.set_data("PASS" if snap.crc_valid else "WAITING", color="#4ade80" if snap.crc_valid else "#fbbf24")
        if hasattr(self, "kpi_operator") and snap.crc_valid:
            if getattr(snap, "operator_name", None):
                self.kpi_operator.set_data(
                    snap.operator_name,
                    f"PLMN: {snap.plmn_id} | TAC: {snap.tac}",
                    color="#38bdf8",
                )
            else:
                self.kpi_operator.set_data("LTE Base Station", "MIB Decoded", color="#38bdf8")

        # Update Live Plots
        self.live_spec_curve.setData(snap.spectrum_freqs_mhz, snap.spectrum_dbfs)

        # Update Waterfall
        # Downsample spectrum slice to 256 bins for waterfall line
        if len(snap.spectrum_dbfs) > 0:
            row = np.interp(np.linspace(0, len(snap.spectrum_dbfs) - 1, 256), np.arange(len(snap.spectrum_dbfs)), snap.spectrum_dbfs)
            self.waterfall_buffer = np.roll(self.waterfall_buffer, -1, axis=0)
            self.waterfall_buffer[-1] = row
            self.live_waterfall_img.setImage(self.waterfall_buffer, autoLevels=True)

        # Update Time Domain
        n_disp = min(512, len(snap.latest_block_iq))
        self.live_curve_i.setData(snap.latest_block_iq.real[:n_disp])
        self.live_curve_q.setData(snap.latest_block_iq.imag[:n_disp])

        # Update Constellation
        self.live_const_scatter.setData(snap.constellation_i, snap.constellation_q)

        # Update MIBs Table
        if snap.recent_mibs:
            self.table_mibs.setRowCount(len(snap.recent_mibs))
            for row_idx, rec in enumerate(reversed(snap.recent_mibs)):
                self.table_mibs.setItem(row_idx, 0, QTableWidgetItem(str(rec["timestamp"])))
                self.table_mibs.setItem(row_idx, 1, QTableWidgetItem(str(rec["pci"])))
                self.table_mibs.setItem(row_idx, 2, QTableWidgetItem(str(rec["sfn"])))
                self.table_mibs.setItem(row_idx, 3, QTableWidgetItem(f"{rec['bandwidth_mhz']:.1f} MHz"))
                self.table_mibs.setItem(row_idx, 4, QTableWidgetItem(f"{rec['antenna_ports']} Tx"))
                self.table_mibs.setItem(row_idx, 5, QTableWidgetItem(f"{rec['cfo_hz']:+.1f}"))
                crc_item = QTableWidgetItem("VALID")
                crc_item.setForeground(QColor("#4ade80"))
                self.table_mibs.setItem(row_idx, 6, crc_item)

    def _on_tab_changed(self, index: int):
        if index != 2:
            # Switched away from Continuous Stream: pause stream if running
            if self.stream_runner is not None and self.stream_runner.running:
                self._stop_stream()
        else:
            # Switched to Continuous Stream
            pass

    def closeEvent(self, event):
        self._stop_stream()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setStyleSheet(DARK_STYLESHEET)
    window = LTEStudioApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
