"""Tk desktop application for testing the streaming pipeline with HDF5 IQ frames."""

from __future__ import annotations

import argparse
import csv
import json
import queue
import threading
import traceback
from pathlib import Path

from .hdf5_runner import (
    DatasetSummary,
    HarnessConfig,
    PipelineRunResult,
    RunCancelled,
    inspect_hdf5_dataset,
    run_hdf5_pipeline,
)

NAVY = "#102a43"
BLUE = "#1677a6"
TEAL = "#10a5a5"
GREEN = "#27865b"
AMBER = "#d9951e"
RED = "#c44949"
INK = "#172b3a"
MUTED = "#61788a"
PANEL = "#f5f8fa"
GRID = "#d8e2e8"
WHITE = "#ffffff"


def _default_dataset() -> Path:
    repo = Path(__file__).resolve().parents[2]
    candidate = repo / "subset_test.h5"
    return candidate if candidate.exists() else Path("subset_test.h5")


def _channel_name(channel: int) -> str:
    return {0: "clean", 1: "multipath"}.get(channel, str(channel))


class PipelineDatasetApp:
    """Responsive Tk application; production processing runs on a background thread."""

    def __init__(self, root, dataset_path: str | Path) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("RF Pipeline Dataset Test Bench")
        self.root.geometry("1480x940")
        self.root.minsize(1180, 720)
        self.root.configure(bg=PANEL)

        self.summary: DatasetSummary | None = None
        self.result: PipelineRunResult | None = None
        self._messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self._cancel_event: threading.Event | None = None
        self._run_thread: threading.Thread | None = None

        self.path_var = tk.StringVar(value=str(Path(dataset_path).resolve()))
        self.modulation_var = tk.StringVar(value="All")
        self.snr_var = tk.StringVar(value="All")
        self.channel_var = tk.StringVar(value="All")
        self.frames_var = tk.StringVar(value="24")
        self.first_index_var = tk.StringVar(value="0")
        self.seed_var = tk.StringVar(value="2026")
        self.random_var = tk.BooleanVar(value=False)
        self.normalize_var = tk.BooleanVar(value=True)
        self.adaptive_var = tk.BooleanVar(value=True)
        self.features_var = tk.BooleanVar(value=True)
        self.fs_var = tk.StringVar(value="1000000")
        self.noise_var = tk.StringVar(value="1e-4")
        self.block_var = tk.StringVar(value="1024")
        self.window_var = tk.StringVar(value="256")
        self.hop_var = tk.StringVar(value="64")
        self.guard_var = tk.StringVar(value="2048")
        self.pre_var = tk.StringVar(value="256")
        self.post_var = tk.StringVar(value="256")
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.metric_frames = tk.StringVar(value="–")
        self.metric_detected = tk.StringVar(value="–")
        self.metric_events = tk.StringVar(value="–")
        self.metric_rate = tk.StringVar(value="–")
        self.metric_features = tk.StringVar(value="–")
        self.metric_throughput = tk.StringVar(value="–")
        self.selected_event_var = tk.StringVar(
            value="Select an event to inspect its spectrum and features."
        )

        self._configure_styles()
        self._build_layout()
        self.root.after(80, self._poll_messages)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(150, self.inspect_dataset)

    def _configure_styles(self) -> None:
        ttk = self.ttk
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=PANEL)
        style.configure("Panel.TFrame", background=WHITE)
        style.configure("Header.TFrame", background=NAVY)
        style.configure(
            "Header.TLabel", background=NAVY, foreground=WHITE, font=("Segoe UI", 18, "bold")
        )
        style.configure(
            "HeaderSmall.TLabel", background=NAVY, foreground="#c9d8e4", font=("Segoe UI", 9)
        )
        style.configure(
            "Section.TLabel", background=WHITE, foreground=NAVY, font=("Segoe UI", 11, "bold")
        )
        style.configure("Body.TLabel", background=WHITE, foreground=INK, font=("Segoe UI", 9))
        style.configure("Muted.TLabel", background=WHITE, foreground=MUTED, font=("Segoe UI", 8))
        style.configure(
            "Metric.TLabel", background=WHITE, foreground=NAVY, font=("Segoe UI", 20, "bold")
        )
        style.configure(
            "MetricName.TLabel", background=WHITE, foreground=MUTED, font=("Segoe UI", 8, "bold")
        )
        style.configure(
            "Accent.TButton",
            background=TEAL,
            foreground=WHITE,
            font=("Segoe UI", 9, "bold"),
            padding=(12, 8),
        )
        style.map("Accent.TButton", background=[("active", "#0d8f8f"), ("disabled", "#9cb7b7")])
        style.configure("Danger.TButton", foreground=RED, padding=(9, 7))
        style.configure(
            "Treeview", rowheight=27, font=("Segoe UI", 8), background=WHITE, fieldbackground=WHITE
        )
        style.configure(
            "Treeview.Heading", background=NAVY, foreground=WHITE, font=("Segoe UI", 8, "bold")
        )
        style.map("Treeview.Heading", background=[("active", BLUE)])
        style.configure("TNotebook", background=PANEL, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 9, "bold"))
        style.map("TNotebook.Tab", foreground=[("selected", BLUE)])

    def _build_layout(self) -> None:
        ttk = self.ttk
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(22, 14))
        header.pack(fill="x")
        ttk.Label(header, text="RF Pipeline Dataset Test Bench", style="Header.TLabel").pack(
            side="left"
        )
        ttk.Label(
            header,
            text="Real positioned IQ • causal detection • transient capture • 56-feature analysis",
            style="HeaderSmall.TLabel",
        ).pack(side="left", padx=(22, 0), pady=(6, 0))
        self.version_label = ttk.Label(
            header, text="signal_processing 2.4.0rc1", style="HeaderSmall.TLabel"
        )
        self.version_label.pack(side="right", pady=(6, 0))

        body = ttk.Frame(self.root, padding=(12, 12))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        sidebar_shell = ttk.Frame(body, style="Panel.TFrame", width=340)
        self.sidebar_shell = sidebar_shell
        sidebar_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        sidebar_shell.grid_propagate(False)
        sidebar_shell.rowconfigure(0, weight=1)
        sidebar_shell.columnconfigure(0, weight=1)
        sidebar_canvas = self.tk.Canvas(
            sidebar_shell,
            width=320,
            background=WHITE,
            highlightthickness=0,
            borderwidth=0,
        )
        sidebar_scroll = ttk.Scrollbar(
            sidebar_shell, orient="vertical", command=sidebar_canvas.yview
        )
        sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        sidebar_canvas.grid(row=0, column=0, sticky="nsew")
        sidebar_scroll.grid(row=0, column=1, sticky="ns")
        sidebar = ttk.Frame(sidebar_canvas, style="Panel.TFrame", padding=14)
        sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")
        sidebar.bind(
            "<Configure>",
            lambda _event: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")),
        )
        sidebar_canvas.bind(
            "<Configure>",
            lambda event: sidebar_canvas.itemconfigure(sidebar_window, width=event.width),
        )
        sidebar_canvas.bind(
            "<MouseWheel>",
            lambda event: sidebar_canvas.yview_scroll(int(-event.delta / 120), "units"),
        )
        content = ttk.Frame(body)
        content.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        self._build_controls(sidebar)
        self._build_tabs(content)

        footer = ttk.Frame(self.root, style="Panel.TFrame", padding=(15, 8))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status_var, style="Body.TLabel").pack(side="left")
        self.progress = ttk.Progressbar(footer, variable=self.progress_var, maximum=100, length=330)
        self.progress.pack(side="right")

    def _entry(self, parent, label: str, variable, row: int, width: int = 12) -> None:
        ttk = self.ttk
        ttk.Label(parent, text=label, style="Body.TLabel").grid(
            row=row, column=0, sticky="w", pady=3
        )
        ttk.Entry(parent, textvariable=variable, width=width).grid(
            row=row, column=1, sticky="ew", pady=3
        )

    def _build_controls(self, parent) -> None:
        ttk = self.ttk
        parent.columnconfigure(0, weight=1)
        ttk.Label(parent, text="Dataset", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        path_frame = ttk.Frame(parent, style="Panel.TFrame")
        path_frame.grid(row=1, column=0, sticky="ew", pady=(6, 4))
        path_frame.columnconfigure(0, weight=1)
        ttk.Entry(path_frame, textvariable=self.path_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(path_frame, text="Browse", command=self._browse).grid(
            row=0, column=1, padx=(5, 0)
        )
        ttk.Button(parent, text="Inspect dataset", command=self.inspect_dataset).grid(
            row=2, column=0, sticky="ew", pady=(2, 8)
        )
        self.dataset_hint = ttk.Label(
            parent,
            text="No dataset inspected",
            style="Muted.TLabel",
            wraplength=295,
            justify="left",
        )
        self.dataset_hint.grid(row=3, column=0, sticky="ew")

        ttk.Separator(parent).grid(row=4, column=0, sticky="ew", pady=10)
        ttk.Label(parent, text="Frame selection", style="Section.TLabel").grid(
            row=5, column=0, sticky="w"
        )
        filters = ttk.Frame(parent, style="Panel.TFrame")
        filters.grid(row=6, column=0, sticky="ew", pady=(5, 0))
        filters.columnconfigure(1, weight=1)
        ttk.Label(filters, text="Modulation", style="Body.TLabel").grid(
            row=0, column=0, sticky="w", pady=3
        )
        self.mod_combo = ttk.Combobox(
            filters, textvariable=self.modulation_var, values=("All",), state="readonly", width=15
        )
        self.mod_combo.grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Label(filters, text="SNR label", style="Body.TLabel").grid(
            row=1, column=0, sticky="w", pady=3
        )
        self.snr_combo = ttk.Combobox(
            filters, textvariable=self.snr_var, values=("All",), state="readonly", width=15
        )
        self.snr_combo.grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Label(filters, text="Channel", style="Body.TLabel").grid(
            row=2, column=0, sticky="w", pady=3
        )
        self.channel_combo = ttk.Combobox(
            filters, textvariable=self.channel_var, values=("All",), state="readonly", width=15
        )
        self.channel_combo.grid(row=2, column=1, sticky="ew", pady=3)
        self._entry(filters, "Frames", self.frames_var, 3)
        self._entry(filters, "First row", self.first_index_var, 4)
        self._entry(filters, "Seed", self.seed_var, 5)
        ttk.Checkbutton(filters, text="Random matching rows", variable=self.random_var).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=2
        )
        ttk.Checkbutton(
            filters, text="Normalize to common noise reference", variable=self.normalize_var
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Separator(parent).grid(row=7, column=0, sticky="ew", pady=10)
        ttk.Label(parent, text="Pipeline", style="Section.TLabel").grid(row=8, column=0, sticky="w")
        settings = ttk.Frame(parent, style="Panel.TFrame")
        settings.grid(row=9, column=0, sticky="ew", pady=(5, 0))
        settings.columnconfigure(1, weight=1)
        self._entry(settings, "Sample rate (Hz)", self.fs_var, 0)
        self._entry(settings, "Noise power", self.noise_var, 1)
        self._entry(settings, "Transport block", self.block_var, 2)
        self._entry(settings, "Window N", self.window_var, 3)
        self._entry(settings, "Hop H", self.hop_var, 4)
        self._entry(settings, "Quiet guard", self.guard_var, 5)
        self._entry(settings, "Pre-trigger", self.pre_var, 6)
        self._entry(settings, "Post-trigger", self.post_var, 7)
        ttk.Checkbutton(settings, text="Adaptive noise floor", variable=self.adaptive_var).grid(
            row=8, column=0, columnspan=2, sticky="w", pady=2
        )
        ttk.Checkbutton(settings, text="Extract all 56 features", variable=self.features_var).grid(
            row=9, column=0, columnspan=2, sticky="w", pady=2
        )

        note = ttk.Label(
            parent,
            text="Frames are independent records. Quiet guards create a declared test stream; normalization is recommended for a common stationary reference.",
            style="Muted.TLabel",
            wraplength=295,
            justify="left",
        )
        note.grid(row=10, column=0, sticky="ew", pady=(10, 6))
        actions = ttk.Frame(parent, style="Panel.TFrame")
        actions.grid(row=11, column=0, sticky="ew", pady=(4, 0))
        actions.columnconfigure(0, weight=1)
        self.run_button = ttk.Button(
            actions, text="Run pipeline test", style="Accent.TButton", command=self.start_run
        )
        self.run_button.grid(row=0, column=0, sticky="ew")
        self.cancel_button = ttk.Button(
            actions,
            text="Cancel",
            style="Danger.TButton",
            command=self.cancel_run,
            state="disabled",
        )
        self.cancel_button.grid(row=0, column=1, padx=(6, 0))

    def _build_tabs(self, parent) -> None:
        ttk = self.ttk
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.overview_tab = ttk.Frame(self.notebook, style="Panel.TFrame", padding=16)
        self.timeline_tab = ttk.Frame(self.notebook, style="Panel.TFrame", padding=10)
        self.events_tab = ttk.Frame(self.notebook, style="Panel.TFrame", padding=10)
        self.features_tab = ttk.Frame(self.notebook, style="Panel.TFrame", padding=10)
        self.dataset_tab = ttk.Frame(self.notebook, style="Panel.TFrame", padding=12)
        self.log_tab = ttk.Frame(self.notebook, style="Panel.TFrame", padding=10)
        for frame, label in (
            (self.overview_tab, "Overview"),
            (self.timeline_tab, "Timeline & spectrum"),
            (self.events_tab, "Captured events"),
            (self.features_tab, "Features"),
            (self.dataset_tab, "Dataset"),
            (self.log_tab, "Run log"),
        ):
            self.notebook.add(frame, text=label)
        self._build_overview()
        self._build_plots()
        self._build_events()
        self._build_features()
        self._build_dataset_view()
        self._build_log()
        from .smoke_panel import SmokePanel

        self.smoke_panel = SmokePanel(self.notebook)
        self.notebook.add(self.smoke_panel, text="Detector ground truth")
        self.notebook.bind("<<NotebookTabChanged>>", self._dataset_tab_changed)

    def _dataset_tab_changed(self, _event=None):
        if self.notebook.select() == str(self.smoke_panel):
            self.sidebar_shell.grid_remove()
        else:
            self.sidebar_shell.grid()

    def _build_overview(self) -> None:
        ttk = self.ttk
        tab = self.overview_tab
        tab.columnconfigure((0, 1, 2), weight=1)
        for col, (title, var) in enumerate(
            (
                ("DATASET FRAMES", self.metric_frames),
                ("MATCHED TRUTHS", self.metric_detected),
                ("CAPTURED EVENTS", self.metric_events),
            )
        ):
            self._metric_card(tab, title, var, 0, col)
        for col, (title, var) in enumerate(
            (
                ("DETECTION RATE", self.metric_rate),
                ("VALID FEATURES", self.metric_features),
                ("PIPELINE THROUGHPUT", self.metric_throughput),
            )
        ):
            self._metric_card(tab, title, var, 1, col)
        ttk.Label(tab, text="What this test means", style="Section.TLabel").grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(18, 6)
        )
        self.overview_text = self.tk.Text(
            tab,
            height=16,
            wrap="word",
            relief="flat",
            bg="#f8fbfc",
            fg=INK,
            font=("Segoe UI", 10),
            padx=14,
            pady=12,
        )
        self.overview_text.grid(row=3, column=0, columnspan=3, sticky="nsew")
        tab.rowconfigure(3, weight=1)
        self._set_text(
            self.overview_text,
            "Load the supplied HDF5 file and run the real StreamingCapturePipeline. "
            "The dashboard will report one-to-one truth matching, capture boundaries, "
            "feature validity and measured unpaced processing throughput.\n\n"
            "The HDF5 records are independent labelled frames, not a continuous RF recording. "
            "The harness creates an explicit continuous test stream using quiet Gaussian guards. "
            "Normalized mode scales each frame to the selected common noise reference using its "
            "nominal SNR label. Results therefore test software behavior under this declared harness; "
            "they are not hardware-acquisition or arbitrary-noise qualification.",
        )

    def _metric_card(self, parent, title, variable, row: int, col: int) -> None:
        ttk = self.ttk
        card = ttk.Frame(parent, style="Panel.TFrame", padding=14, relief="solid", borderwidth=1)
        card.grid(row=row, column=col, sticky="ew", padx=5, pady=5)
        ttk.Label(card, text=title, style="MetricName.TLabel").pack(anchor="w")
        ttk.Label(card, textvariable=variable, style="Metric.TLabel").pack(anchor="w", pady=(4, 0))

    def _build_plots(self) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        ttk = self.ttk
        tab = self.timeline_tab
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        self.figure = Figure(figsize=(9.5, 7.2), dpi=100, facecolor=WHITE)
        self.timeline_axis = self.figure.add_subplot(211)
        self.spectrum_axis = self.figure.add_subplot(212)
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08, hspace=0.38)
        self.canvas = FigureCanvasTkAgg(self.figure, master=tab)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        ttk.Label(tab, textvariable=self.selected_event_var, style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", padx=8, pady=4
        )
        self._empty_plots()

    def _empty_plots(self) -> None:
        for axis in (self.timeline_axis, self.spectrum_axis):
            axis.clear()
            axis.set_facecolor("#fbfdfe")
            axis.grid(True, color=GRID, linewidth=0.6)
        self.timeline_axis.set_title(
            "Stream magnitude and capture intervals", color=NAVY, weight="bold"
        )
        self.timeline_axis.set_xlabel("Absolute sample")
        self.timeline_axis.set_ylabel("|IQ|")
        self.spectrum_axis.set_title("Selected captured event spectrum", color=NAVY, weight="bold")
        self.spectrum_axis.set_xlabel("Frequency (kHz)")
        self.spectrum_axis.set_ylabel("Relative power (dB)")
        self.canvas.draw_idle()

    def _build_events(self) -> None:
        ttk = self.ttk
        tab = self.events_tab
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)
        top = ttk.Frame(tab, style="Panel.TFrame")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(top, text="One-to-one event/truth attribution", style="Section.TLabel").pack(
            side="left"
        )
        ttk.Button(top, text="Export CSV", command=self.export_csv).pack(side="right")
        columns = (
            "event",
            "detected",
            "row",
            "mod",
            "snr",
            "channel",
            "iou",
            "samples",
            "features",
            "quality",
        )
        self.event_tree = ttk.Treeview(tab, columns=columns, show="headings", selectmode="browse")
        headings = {
            "event": "Event",
            "detected": "Detected interval",
            "row": "Dataset row",
            "mod": "Mod",
            "snr": "SNR",
            "channel": "Channel",
            "iou": "IoU",
            "samples": "Capture samples",
            "features": "Features",
            "quality": "Quality",
        }
        widths = {
            "event": 72,
            "detected": 160,
            "row": 85,
            "mod": 70,
            "snr": 55,
            "channel": 85,
            "iou": 65,
            "samples": 100,
            "features": 90,
            "quality": 130,
        }
        for column in columns:
            self.event_tree.heading(column, text=headings[column])
            self.event_tree.column(
                column, width=widths[column], stretch=column in {"detected", "quality"}
            )
        yscroll = ttk.Scrollbar(tab, orient="vertical", command=self.event_tree.yview)
        self.event_tree.configure(yscrollcommand=yscroll.set)
        self.event_tree.grid(row=1, column=0, sticky="nsew")
        yscroll.grid(row=1, column=1, sticky="ns")
        self.event_tree.bind("<<TreeviewSelect>>", self._event_selected)

    def _build_features(self) -> None:
        ttk = self.ttk
        tab = self.features_tab
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)
        self.feature_heading = ttk.Label(
            tab, text="Select a captured event", style="Section.TLabel"
        )
        self.feature_heading.grid(row=0, column=0, sticky="w", pady=(0, 8))
        columns = ("name", "value", "unit", "valid", "reason")
        self.feature_tree = ttk.Treeview(tab, columns=columns, show="headings")
        for column, label, width in (
            ("name", "Feature", 260),
            ("value", "Value", 150),
            ("unit", "Unit", 90),
            ("valid", "Valid", 70),
            ("reason", "Reason / interpretation", 440),
        ):
            self.feature_tree.heading(column, text=label)
            self.feature_tree.column(column, width=width, stretch=column in {"name", "reason"})
        yscroll = ttk.Scrollbar(tab, orient="vertical", command=self.feature_tree.yview)
        self.feature_tree.configure(yscrollcommand=yscroll.set)
        self.feature_tree.grid(row=1, column=0, sticky="nsew")
        yscroll.grid(row=1, column=1, sticky="ns")

    def _build_dataset_view(self) -> None:
        tab = self.dataset_tab
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)
        self.ttk.Label(tab, text="HDF5 schema and label distribution", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.dataset_text = self.tk.Text(
            tab,
            wrap="word",
            relief="flat",
            bg="#f8fbfc",
            fg=INK,
            font=("Consolas", 9),
            padx=14,
            pady=12,
        )
        self.dataset_text.grid(row=1, column=0, sticky="nsew")
        self._set_text(self.dataset_text, "Inspect a dataset to view its schema.")

    def _build_log(self) -> None:
        tab = self.log_tab
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        self.log_text = self.tk.Text(
            tab,
            wrap="word",
            relief="flat",
            bg="#10212f",
            fg="#dce8ef",
            insertbackground=WHITE,
            font=("Consolas", 9),
            padx=12,
            pady=10,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

    def _set_text(self, widget, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _browse(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Select labelled IQ HDF5 dataset",
            filetypes=(("HDF5", "*.h5 *.hdf5"), ("All files", "*.*")),
        )
        if path:
            self.path_var.set(path)
            self.inspect_dataset()

    def inspect_dataset(self) -> None:
        from tkinter import messagebox

        try:
            summary = inspect_hdf5_dataset(self.path_var.get())
        except Exception as exc:
            self.summary = None
            self.dataset_hint.configure(text=f"Dataset error: {exc}")
            messagebox.showerror("Dataset error", str(exc))
            return
        self.summary = summary
        mods = tuple(summary.modulation_to_id)
        snrs = tuple(map(str, summary.snr_counts))
        channels = tuple(f"{key}: {_channel_name(key)}" for key in summary.channel_counts)
        self.mod_combo.configure(values=("All",) + mods)
        self.snr_combo.configure(values=("All",) + snrs)
        self.channel_combo.configure(values=("All",) + channels)
        if self.modulation_var.get() not in ("All",) + mods:
            self.modulation_var.set("All")
        if self.snr_var.get() not in ("All",) + snrs:
            self.snr_var.set("All")
        if self.channel_var.get() not in ("All",) + channels:
            self.channel_var.set("All")
        self.dataset_hint.configure(
            text=f"{summary.frame_count:,} frames × {summary.frame_length:,} samples • {summary.sample_dtype} • subset={summary.subset_name}"
        )
        lines = [
            f"Path: {summary.path}",
            f"Source attribute: {summary.source}",
            f"Subset: {summary.subset_name}",
            f"X shape: ({summary.frame_count:,}, {summary.frame_length:,}, 2)",
            f"X dtype: {summary.sample_dtype}",
            f"Channel meaning: {summary.channel_meaning}",
            "",
            "Modulation labels:",
        ]
        lines.extend(
            f"  {name:<8} id={summary.modulation_to_id[name]}  rows={summary.modulation_counts[name]:,}"
            for name in summary.modulation_to_id
        )
        lines.extend(("", "SNR labels:"))
        lines.extend(f"  {snr:>4} dB  rows={count:,}" for snr, count in summary.snr_counts.items())
        lines.extend(("", "Channel labels:"))
        lines.extend(
            f"  {channel}: {_channel_name(channel):<10} rows={count:,}"
            for channel, count in summary.channel_counts.items()
        )
        lines.extend(
            (
                "",
                "Interpretation boundary:",
                "  Each row is an independent labelled frame. It is not safe to concatenate rows and",
                "  infer one stationary receiver noise process without an explicit harness policy.",
                "  Recommended normalized mode creates a common reference and reports that transformation.",
            )
        )
        self._set_text(self.dataset_text, "\n".join(lines))
        self._log(f"Inspected dataset: {summary.path}")

    def _optional_int(self, value: str) -> int | None:
        return None if value == "All" else int(value.split(":", 1)[0])

    def _config(self) -> HarnessConfig:
        return HarnessConfig(
            sample_rate_hz=float(self.fs_var.get()),
            frame_count=int(self.frames_var.get()),
            modulation=None if self.modulation_var.get() == "All" else self.modulation_var.get(),
            snr_db=self._optional_int(self.snr_var.get()),
            channel=self._optional_int(self.channel_var.get()),
            selection_seed=int(self.seed_var.get()),
            first_index=int(self.first_index_var.get()),
            random_selection=bool(self.random_var.get()),
            normalize_frames=bool(self.normalize_var.get()),
            noise_power_linear=float(self.noise_var.get()),
            transport_block_size=int(self.block_var.get()),
            window_size=int(self.window_var.get()),
            hop_size=int(self.hop_var.get()),
            guard_samples=int(self.guard_var.get()),
            pre_trigger_samples=int(self.pre_var.get()),
            post_signal_samples=int(self.post_var.get()),
            min_post_trigger_samples=int(self.post_var.get()),
            adaptive_noise=bool(self.adaptive_var.get()),
            enable_features=bool(self.features_var.get()),
        )

    def start_run(self) -> None:
        from tkinter import messagebox

        if self._run_thread is not None and self._run_thread.is_alive():
            return
        try:
            config = self._config()
            if self.summary is None or self.summary.path != Path(self.path_var.get()).resolve():
                self.inspect_dataset()
            if self.summary is None:
                return
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return
        self.result = None
        self._cancel_event = threading.Event()
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress_var.set(0)
        self.status_var.set("Starting pipeline test…")
        self._log("=" * 70)
        self._log(f"Run started with configuration: {config}")
        # Capture Tk-owned state before entering the worker thread. Tk variables
        # must never be read from the background processing thread.
        dataset_path = self.path_var.get()

        def work() -> None:
            try:
                result = run_hdf5_pipeline(
                    dataset_path,
                    config,
                    progress=lambda fraction, message: self._messages.put(
                        ("progress", (fraction, message))
                    ),
                    cancel_event=self._cancel_event,
                )
                self._messages.put(("result", result))
            except RunCancelled as exc:
                self._messages.put(("cancelled", str(exc)))
            except Exception as exc:
                self._messages.put(("error", (exc, traceback.format_exc())))

        self._run_thread = threading.Thread(target=work, name="hdf5-pipeline-ui-run", daemon=True)
        self._run_thread.start()

    def cancel_run(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.status_var.set("Cancellation requested; waiting for current block…")
            self.cancel_button.configure(state="disabled")

    def _poll_messages(self) -> None:
        from tkinter import messagebox

        try:
            while True:
                kind, payload = self._messages.get_nowait()
                if kind == "progress":
                    fraction, message = payload
                    self.progress_var.set(float(fraction) * 100)
                    self.status_var.set(str(message))
                elif kind == "result":
                    self._show_result(payload)
                    self._finish_run_controls()
                elif kind == "cancelled":
                    self.status_var.set(str(payload))
                    self._log(str(payload))
                    self._finish_run_controls()
                elif kind == "error":
                    exc, trace = payload
                    self._log(trace)
                    self.status_var.set(f"Run failed: {type(exc).__name__}: {exc}")
                    self._finish_run_controls()
                    messagebox.showerror("Pipeline run failed", f"{type(exc).__name__}: {exc}")
        except queue.Empty:
            pass
        self.root.after(80, self._poll_messages)

    def _finish_run_controls(self) -> None:
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")

    def _show_result(self, result: PipelineRunResult) -> None:
        self.result = result
        valid_features = sum(
            sum(1 for feature in sidecar.features.features.values() if feature.valid)
            for sidecar in result.captures
            if sidecar.features is not None
        )
        possible_features = 56 * sum(sidecar.features is not None for sidecar in result.captures)
        self.metric_frames.set(f"{len(result.truths):,}")
        self.metric_detected.set(f"{result.matched_truths:,}")
        self.metric_events.set(f"{len(result.captures):,}")
        self.metric_rate.set(f"{result.detection_rate * 100:.1f}%")
        self.metric_features.set(
            f"{valid_features:,}/{possible_features:,}" if possible_features else "disabled"
        )
        self.metric_throughput.set(f"{result.throughput_samples_per_second / 1e6:.2f} MS/s")
        mode = (
            "normalized common-reference"
            if result.config.normalize_frames
            else "raw frame amplitude"
        )
        noise_mode = "adaptive robust" if result.config.adaptive_noise else "fixed Gaussian"
        text = (
            f"The production pipeline processed {result.total_stream_samples:,} samples in "
            f"{result.elapsed_seconds:.3f} s ({result.throughput_samples_per_second / 1e6:.3f} MS/s unpaced).\n\n"
            f"One-to-one interval matching found {result.matched_truths}/{len(result.truths)} labelled "
            f"frames, with {result.missed_truths} misses and {result.extra_events} extra events. "
            f"The run used {mode} and a {noise_mode} reference initialized at "
            f"{result.config.noise_power_linear:.6g} linear power. Final estimator power was "
            f"{result.final_noise_power_linear!s}.\n\n"
            "A match means the detector's physical detected interval overlaps a labelled frame and "
            "wins the one-to-one assignment. Review IoU and boundaries in Captured events; a high "
            "frame-level match rate alone does not certify exact event extent, low-SNR operation, "
            "arbitrary noise, or real-time acquisition.\n\n"
            f"Selected HDF5 rows: {', '.join(map(str, result.selected_indices[:30]))}"
            + (" …" if len(result.selected_indices) > 30 else "")
        )
        self._set_text(self.overview_text, text)
        self._populate_events(result)
        self._draw_timeline(result)
        self.status_var.set(
            f"Complete: {result.matched_truths}/{len(result.truths)} matched, "
            f"{result.extra_events} extras, {result.throughput_samples_per_second / 1e6:.2f} MS/s"
        )
        self.progress_var.set(100)
        self._log(self.status_var.get())
        self.notebook.select(self.overview_tab)

    def _populate_events(self, result: PipelineRunResult) -> None:
        for item in self.event_tree.get_children():
            self.event_tree.delete(item)
        for event_index, (sidecar, match) in enumerate(zip(result.captures, result.matches)):
            event = sidecar.event
            truth = result.truths[match.truth_index] if match.truth_index is not None else None
            features = sidecar.features.features if sidecar.features is not None else {}
            valid = sum(value.valid for value in features.values())
            quality = sidecar.termination_reason
            if not event.complete:
                quality += " / incomplete"
            self.event_tree.insert(
                "",
                "end",
                iid=str(event_index),
                values=(
                    event_index + 1,
                    f"[{event.detected_start_sample}, {event.detected_end_sample})",
                    truth.dataset_index if truth else "–",
                    truth.modulation if truth else "extra",
                    f"{truth.snr_db} dB" if truth else "–",
                    _channel_name(truth.channel) if truth else "–",
                    f"{match.interval_iou:.3f}" if truth else "–",
                    f"{event.num_samples:,}",
                    f"{valid}/{len(features)}" if features else sidecar.feature_status,
                    quality,
                ),
            )
        if result.captures:
            self.event_tree.selection_set("0")
            self.event_tree.focus("0")
            self._display_event(0)
        else:
            self._clear_features("No events were captured")

    def _draw_timeline(self, result: PipelineRunResult) -> None:
        from matplotlib.patches import Patch

        axis = self.timeline_axis
        axis.clear()
        axis.set_facecolor("#fbfdfe")
        axis.plot(
            result.display_sample_indices,
            result.display_magnitudes,
            color="#516f82",
            linewidth=0.65,
            alpha=0.85,
        )
        truth_label = False
        for truth in result.truths:
            axis.axvspan(truth.start_sample, truth.end_sample, color=BLUE, alpha=0.16)
            truth_label = True
        event_label = False
        for sidecar in result.captures:
            event = sidecar.event
            axis.axvspan(
                event.detected_start_sample, event.detected_end_sample, color=AMBER, alpha=0.22
            )
            event_label = True
        axis.set_title(
            "Stream magnitude with truth and detected intervals", color=NAVY, weight="bold"
        )
        axis.set_xlabel("Absolute sample")
        axis.set_ylabel("|IQ|")
        axis.grid(True, color=GRID, linewidth=0.6)
        handles = [Patch(facecolor=BLUE, alpha=0.25, label="Dataset frame")]
        if event_label:
            handles.append(Patch(facecolor=AMBER, alpha=0.30, label="Detected interval"))
        if truth_label:
            axis.legend(handles=handles, loc="upper right", frameon=False)
        self.spectrum_axis.clear()
        self.spectrum_axis.set_facecolor("#fbfdfe")
        self.spectrum_axis.set_title(
            "Select an event below or in the Captured events tab", color=NAVY, weight="bold"
        )
        self.spectrum_axis.set_xlabel("Frequency (kHz)")
        self.spectrum_axis.set_ylabel("Relative power (dB)")
        self.spectrum_axis.grid(True, color=GRID, linewidth=0.6)
        self.canvas.draw_idle()

    def _event_selected(self, _event=None) -> None:
        selected = self.event_tree.selection()
        if selected:
            self._display_event(int(selected[0]))

    def _display_event(self, event_index: int) -> None:
        import numpy as np

        if self.result is None or not (0 <= event_index < len(self.result.captures)):
            return
        sidecar = self.result.captures[event_index]
        event = sidecar.event
        match = self.result.matches[event_index]
        truth = self.result.truths[match.truth_index] if match.truth_index is not None else None
        self.feature_heading.configure(
            text=f"Event {event_index + 1}: {event.event_id} • {sidecar.feature_status}"
        )
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)
        if sidecar.features is not None:
            for name, value in sorted(sidecar.features.features.items()):
                displayed = f"{value.value:.10g}" if value.valid else "–"
                detail = value.reason or value.interpretation or ""
                self.feature_tree.insert(
                    "",
                    "end",
                    values=(
                        name,
                        displayed,
                        value.unit or "",
                        "yes" if value.valid else "no",
                        detail,
                    ),
                )
        else:
            self.feature_tree.insert(
                "",
                "end",
                values=("–", "–", "", "no", sidecar.feature_reason or sidecar.feature_status),
            )

        iq = np.asarray(event.signal_iq, dtype=np.complex128)
        nfft = min(8192, max(256, 1 << max(0, (len(iq) - 1).bit_length())))
        window = np.hanning(len(iq)) if len(iq) > 1 else np.ones(len(iq))
        spectrum = np.fft.fftshift(np.fft.fft(iq * window, n=nfft))
        power = np.abs(spectrum) ** 2
        peak = float(power.max()) if len(power) else 0.0
        relative_db = (
            10 * np.log10(np.maximum(power / peak, 1e-12)) if peak > 0 else np.full(nfft, -120.0)
        )
        frequencies = (
            np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / event.stream.sample_rate_hz)) / 1000.0
        )
        axis = self.spectrum_axis
        axis.clear()
        axis.set_facecolor("#fbfdfe")
        axis.plot(frequencies, relative_db, color=TEAL, linewidth=1.0)
        label = f"Event {event_index + 1}"
        if truth is not None:
            label += f" • row {truth.dataset_index} • {truth.modulation} • {truth.snr_db} dB • IoU {match.interval_iou:.3f}"
        else:
            label += " • unmatched extra"
        axis.set_title(label, color=NAVY, weight="bold", fontsize=10)
        axis.set_xlabel("Frequency (kHz)")
        axis.set_ylabel("Relative power (dB)")
        axis.set_ylim(-80, 3)
        axis.grid(True, color=GRID, linewidth=0.6)
        self.selected_event_var.set(
            f"Capture [{event.event_start_sample}, {event.event_end_sample}) • "
            f"detected [{event.detected_start_sample}, {event.detected_end_sample}) • "
            f"{event.num_samples:,} samples • {sidecar.feature_status}"
        )
        self.canvas.draw_idle()

    def _clear_features(self, message: str) -> None:
        self.feature_heading.configure(text=message)
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)

    def export_csv(self) -> None:
        from tkinter import filedialog, messagebox

        if self.result is None:
            messagebox.showinfo("Nothing to export", "Run the pipeline first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export pipeline results",
            defaultextension=".csv",
            filetypes=(("CSV", "*.csv"),),
            initialfile="subset_test_pipeline_results.csv",
        )
        if not path:
            return
        feature_names = sorted(
            {
                name
                for sidecar in self.result.captures
                if sidecar.features is not None
                for name in sidecar.features.features
            }
        )
        fields = [
            "event_index",
            "event_id",
            "event_start",
            "detected_start",
            "detected_end",
            "event_end",
            "dataset_index",
            "modulation",
            "snr_db",
            "channel",
            "interval_iou",
            "feature_status",
            "complete",
        ] + feature_names
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for event_index, (sidecar, match) in enumerate(
                zip(self.result.captures, self.result.matches)
            ):
                event = sidecar.event
                truth = (
                    self.result.truths[match.truth_index] if match.truth_index is not None else None
                )
                row = {
                    "event_index": event_index,
                    "event_id": event.event_id,
                    "event_start": event.event_start_sample,
                    "detected_start": event.detected_start_sample,
                    "detected_end": event.detected_end_sample,
                    "event_end": event.event_end_sample,
                    "dataset_index": truth.dataset_index if truth else "",
                    "modulation": truth.modulation if truth else "",
                    "snr_db": truth.snr_db if truth else "",
                    "channel": truth.channel if truth else "",
                    "interval_iou": match.interval_iou,
                    "feature_status": sidecar.feature_status,
                    "complete": event.complete,
                }
                if sidecar.features is not None:
                    row.update(
                        {
                            name: value.value if value.valid else ""
                            for name, value in sidecar.features.features.items()
                        }
                    )
                writer.writerow(row)
        self._log(f"Exported CSV: {path}")
        messagebox.showinfo("Export complete", f"Saved {path}")

    def _on_close(self) -> None:
        self.smoke_panel.close()
        if self._cancel_event is not None:
            self._cancel_event.set()
        self.root.destroy()


def _headless(path: Path, frames: int) -> int:
    result = run_hdf5_pipeline(
        path,
        HarnessConfig(frame_count=frames),
        progress=lambda fraction, message: print(f"[{fraction:6.1%}] {message}"),
    )
    valid_features = sum(
        sum(value.valid for value in sidecar.features.features.values())
        for sidecar in result.captures
        if sidecar.features is not None
    )
    payload = {
        "dataset": str(result.dataset.path),
        "frames": len(result.truths),
        "captures": len(result.captures),
        "matched": result.matched_truths,
        "missed": result.missed_truths,
        "extra_events": result.extra_events,
        "detection_rate": result.detection_rate,
        "valid_features": valid_features,
        "elapsed_seconds": result.elapsed_seconds,
        "throughput_samples_per_second": result.throughput_samples_per_second,
    }
    print(json.dumps(payload, indent=2))
    return 0 if result.missed_truths == 0 else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=_default_dataset())
    parser.add_argument(
        "--smoke-dataset",
        type=Path,
        help="Detector dataset folder with manifest and supplied truth",
    )
    parser.add_argument("--csv", type=Path, help="Export smoke headless comparisons to CSV")
    parser.add_argument("--detector-profile", default="compatibility",
        choices=("compatibility", "ar1_whitened", "ar1_multires", "ar1_spectral"),
        help="Detector dataset profile; AR1 candidates are offline/model-only")
    parser.add_argument(
        "--headless", action="store_true", help="Run the same harness without opening Tk"
    )
    parser.add_argument("--frames", type=int, default=24, help="Frames used by --headless")
    args = parser.parse_args(argv)
    if args.headless:
        if args.smoke_dataset:
            from .smoke_runner import (
                SmokeConfig,
                compact_smoke_result,
                export_smoke_csv,
                inspect_smoke_dataset,
                run_smoke_record,
            )

            results = []
            failures = 0
            config = SmokeConfig(detector_profile=args.detector_profile,
                window_size=32 if args.detector_profile == "ar1_whitened" else 256,
                hop_size=64 if args.detector_profile == "compatibility" else 8)
            for record in inspect_smoke_dataset(args.smoke_dataset):
                try:
                    result = compact_smoke_result(run_smoke_record(record, config))
                except Exception as exc:
                    failures += 1
                    print(
                        json.dumps(
                            {"record_id": record.record_id, "error": f"{type(exc).__name__}: {exc}"}
                        ),
                        flush=True,
                    )
                    continue
                results.append(result)
                print(json.dumps({"record_id": record.record_id, **result.metrics}), flush=True)
            if args.csv and results:
                export_smoke_csv(args.csv, results)
            if failures:
                return 1
            return (
                2
                if any(
                    r.metrics["misses"] or r.metrics["extras"] or r.metrics["feature_failures"]
                    for r in results
                )
                else 0
            )
        return _headless(args.dataset, args.frames)
    try:
        import tkinter as tk

        import matplotlib  # noqa: F401 - verify optional UI dependency early
    except ImportError as exc:
        raise SystemExit(
            f"UI dependencies unavailable: {exc}. Install with pip install -e .[ui]"
        ) from exc
    root = tk.Tk()
    app = PipelineDatasetApp(root, args.dataset)
    if args.smoke_dataset:
        app.smoke_panel.path.set(str(args.smoke_dataset.resolve()))
        app.smoke_panel.detector_profile.set(args.detector_profile)
        app.smoke_panel._profile_changed()
        app.notebook.select(app.smoke_panel)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
