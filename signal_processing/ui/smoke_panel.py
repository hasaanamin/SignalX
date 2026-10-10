"""Independent-record ground-truth comparison tab for the desktop test bench."""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from signal_processing.ui.hdf5_runner import RunCancelled
from signal_processing.ui.smoke_runner import (
    SmokeConfig,
    compact_smoke_result,
    export_smoke_csv,
    inspect_smoke_dataset,
    run_smoke_record,
)


class SmokePanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=12, style="Panel.TFrame")
        self.records = ()
        self.results = {}
        self.thread = None
        self.cancel = threading.Event()
        self.messages = queue.Queue(maxsize=64)
        self.closed = False
        self.path = tk.StringVar(
            value=str(Path(__file__).resolve().parents[2] / "detector_smoke314")
        )
        self.selected = tk.StringVar(value="All records")
        self.status = tk.StringVar(
            value="Load the detector dataset folder to inspect its supplied truth."
        )
        self.values = {
            name: tk.StringVar(value=str(value))
            for name, value in (
                ("window_size", 256),
                ("hop_size", 64),
                ("block_size", 1024),
                ("pre_trigger_samples", 256),
                ("post_trigger_samples", 256),
                ("max_event_samples", 131072),
                ("minimum_iou", 0.2),
            )
        }
        self.features = tk.BooleanVar(value=True)
        self.detector_profile = tk.StringVar(value="compatibility")
        self.export_mode = tk.StringVar(value="combined")
        self._build()
        self.after(100, self._poll)
        self.after(200, self.load)

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(5, weight=1)
        pathbar = ttk.Frame(self, style="Panel.TFrame")
        pathbar.grid(row=0, column=0, sticky="ew")
        pathbar.columnconfigure(1, weight=1)
        ttk.Label(pathbar, text="Detector dataset folder", style="Section.TLabel").grid(
            row=0, column=0, padx=5
        )
        ttk.Entry(pathbar, textvariable=self.path).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(pathbar, text="Browse folder", command=self.browse).grid(row=0, column=2, padx=5)
        self.load_button = ttk.Button(pathbar, text="Load truth", command=self.load)
        self.load_button.grid(row=0, column=3)
        ttk.Button(
            pathbar, text="Smoke (8)", command=lambda: self.use_dataset("detector_smoke314")
        ).grid(row=0, column=4, padx=4)
        ttk.Button(
            pathbar,
            text="Complete (480)",
            command=lambda: self.use_dataset("detector_dataset_complete"),
        ).grid(row=0, column=5)
        options = ttk.Frame(self, style="Panel.TFrame")
        options.grid(row=1, column=0, sticky="ew", pady=8)
        labels = {
            "window_size": "Window N",
            "hop_size": "Hop H",
            "block_size": "Block",
            "pre_trigger_samples": "Pre-trigger",
            "post_trigger_samples": "Post-trigger",
            "max_event_samples": "Capture cap",
            "minimum_iou": "Match IoU ≥",
        }
        for col, (name, variable) in enumerate(self.values.items()):
            ttk.Label(options, text=labels[name], style="Body.TLabel").grid(
                row=0, column=col, padx=5, sticky="w"
            )
            ttk.Entry(options, textvariable=variable, width=11).grid(row=1, column=col, padx=5)
        ttk.Label(options, text="Detector profile", style="Body.TLabel").grid(row=0, column=7, padx=5)
        profile = ttk.Combobox(options, textvariable=self.detector_profile,
            values=("compatibility", "ar1_whitened", "ar1_multires", "ar1_spectral"),
            width=19, state="readonly")
        profile.grid(row=1, column=7, padx=5)
        profile.bind("<<ComboboxSelected>>", self._profile_changed)
        actions = ttk.Frame(self, style="Panel.TFrame")
        actions.grid(row=2, column=0, sticky="ew", pady=4)
        self.selector = ttk.Combobox(
            actions, textvariable=self.selected, width=40, state="readonly"
        )
        self.selector.pack(side="left", padx=(0, 8))
        ttk.Checkbutton(actions, text="56 features", variable=self.features).pack(side="left")
        self.run_button = ttk.Button(
            actions, text="Run selected / all", command=self.run, style="Accent.TButton"
        )
        self.run_button.pack(side="left", padx=8)
        self.stop_button = ttk.Button(
            actions, text="Cancel", command=self.cancel.set, state="disabled"
        )
        self.stop_button.pack(side="left")
        ttk.Combobox(
            actions,
            textvariable=self.export_mode,
            values=("combined", "truth", "results"),
            width=11,
            state="readonly",
        ).pack(side="right", padx=5)
        ttk.Button(actions, text="Export CSV", command=self.export).pack(side="right")
        ttk.Label(
            self,
            text="Original IQ • fresh pipeline per record • measured noise-only warmup • "
            "ON 1e-4 / OFF 1e-2 model targets • compatibility assumes white noise; AR1 profiles fit the quiet prefix • NOT certified Pfa • offline replay",
            style="Muted.TLabel",
        ).grid(row=3, column=0, sticky="w", pady=5)
        ttk.Label(self, textvariable=self.status, style="Body.TLabel", wraplength=1250).grid(
            row=4, column=0, sticky="ew", pady=5
        )
        self.tabs = ttk.Notebook(self)
        self.tabs.grid(row=5, column=0, sticky="nsew")
        self.record_tree = self._tree(
            "Records",
            (
                ("id", "Record", 115),
                ("scenario", "Scenario", 170),
                ("noise", "Noise", 65),
                ("snr", "Sweep SNR dB", 100),
                ("emitters", "Emitters", 70),
                ("truths", "Activity truth", 90),
                ("captures", "Captures", 80),
                ("match", "Matched", 75),
                ("miss", "Missed", 65),
                ("extra", "Extra", 65),
                ("precision", "Sample precision", 110),
                ("recall", "Sample recall", 100),
                ("event_recall", "Burst detection rate", 125),
                ("event_precision", "Event precision", 110),
                ("state", "Run state", 190),
            ),
        )
        self.record_tree.bind("<<TreeviewSelect>>", self._record_selected)
        self.truth_tree = self._tree(
            "Emitter ground truth",
            (
                ("id", "Emitter ID", 230),
                ("mod", "Truth modulation", 100),
                ("start", "Start sample", 100),
                ("end", "End sample", 100),
                ("snr", "Truth SNR dB", 95),
                ("freq", "Offset Hz", 110),
                ("bw", "Nominal BW Hz", 110),
                ("coverage", "Temporal coverage", 125),
            ),
        )
        self.activity_tree = self._tree(
            "Activity comparison",
            (
                ("id", "Activity", 70),
                ("ts", "Truth start", 95),
                ("te", "Truth end", 95),
                ("event", "Matched event", 230),
                ("ds", "Detected start", 100),
                ("de", "Detected end", 100),
                ("iou", "IoU", 65),
                ("on", "Onset error", 90),
                ("end", "End error", 90),
                ("status", "Status", 100),
            ),
        )
        self.event_tree = self._tree(
            "Package results",
            (
                ("id", "Event ID", 230),
                ("ds", "Detected start", 100),
                ("de", "Detected end", 100),
                ("cs", "Capture start", 100),
                ("ce", "Capture end", 100),
                ("pre", "Pre-trigger", 85),
                ("complete", "Complete", 80),
                ("termination", "Termination", 130),
                ("status", "Analysis", 95),
                ("features", "Valid features", 100),
                ("match", "Activity match", 110),
            ),
        )
        self.event_tree.bind("<<TreeviewSelect>>", self._event_selected)
        self.feature_tree = self._tree(
            "Feature details",
            (
                ("name", "Feature", 260),
                ("value", "Value", 160),
                ("unit", "Unit", 80),
                ("valid", "Valid", 65),
                ("reason", "Reason / interpretation", 490),
            ),
        )
        plot_tab = ttk.Frame(self.tabs, style="Panel.TFrame")
        self.tabs.add(plot_tab, text="Truth / result plot")
        self.figure = Figure(figsize=(11, 5), dpi=100, facecolor="white")
        self.axes = self.figure.subplots(2, 1, sharex=True)
        self.figure.subplots_adjust(left=0.08, right=0.98, bottom=0.12, hspace=0.35)
        self.canvas = FigureCanvasTkAgg(self.figure, plot_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        details = ttk.Frame(self.tabs)
        self.tabs.add(details, text="Provenance / metrics")
        self.details = tk.Text(details, wrap="word", font=("Consolas", 10))
        self.details.pack(fill="both", expand=True)

    def _tree(self, title, columns):
        panel = ttk.Frame(self.tabs)
        self.tabs.add(panel, text=title)
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)
        tree = ttk.Treeview(
            panel, columns=[x[0] for x in columns], show="headings", selectmode="browse"
        )
        for key, label, width in columns:
            tree.heading(key, text=label)
            tree.column(key, width=width, minwidth=60, stretch=False)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(panel, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(panel, orient="horizontal", command=tree.xview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        return tree

    def browse(self):
        if self.thread and self.thread.is_alive():
            return
        path = filedialog.askdirectory(title="Select detector_smoke314 folder")
        if path:
            self.path.set(path)
            self.load()

    def use_dataset(self, name):
        if self.thread and self.thread.is_alive():
            return
        self.path.set(str(Path(__file__).resolve().parents[2] / name))
        self.load()

    def load(self):
        if self.thread and self.thread.is_alive():
            return
        try:
            records = inspect_smoke_dataset(self.path.get())
            annotations = {
                r.record_id: json.loads(r.annotations_path.read_text(encoding="utf-8-sig"))
                for r in records
            }
        except Exception as exc:
            self.status.set(f"Load failed: {exc}")
            return
        self.records = records
        self.annotations = annotations
        self.results.clear()
        self._clear(self.record_tree)
        for record in records:
            a = annotations[record.record_id]
            self.record_tree.insert(
                "",
                "end",
                iid=record.record_id,
                values=(
                    record.record_id,
                    record.scenario,
                    a["background"]["model"],
                    a.get("sweep_snr_db", "—"),
                    len(a["emitters"]),
                    len(a["fullband_activity_intervals"]),
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    "Not run",
                ),
            )
            self.record_tree.set(record.record_id, "state", "Not run")
            self.record_tree.set(record.record_id, "event_recall", "—")
            self.record_tree.set(record.record_id, "event_precision", "—")
        self.selector.configure(values=("All records",) + tuple(r.record_id for r in records))
        self.selected.set("All records")
        self.status.set(
            f"Loaded {len(records)} records. Select a row to inspect supplied truth, then run selected/all."
        )
        self.record_tree.selection_set(records[0].record_id)
        self.show_record(records[0].record_id)

    @staticmethod
    def _clear(tree):
        children = tree.get_children()
        if children:
            tree.delete(*children)

    def _profile_changed(self, _event=None):
        self.values["window_size"].set("32" if self.detector_profile.get() == "ar1_whitened" else "256")
        self.values["hop_size"].set("64" if self.detector_profile.get() == "compatibility" else "8")

    def run(self):
        if self.thread and self.thread.is_alive():
            return
        if not self.records:
            self.load()
        try:
            config = SmokeConfig(
                **{
                    key: float(var.get()) if key == "minimum_iou" else int(var.get())
                    for key, var in self.values.items()
                },
                enable_features=self.features.get(),
                detector_profile=self.detector_profile.get(),
            )
            chosen = tuple(
                r for r in self.records if self.selected.get() in ("All records", r.record_id)
            )
            if not chosen:
                raise ValueError("Select a valid dataset record")
        except Exception as exc:
            messagebox.showerror("Invalid detector settings", str(exc))
            return
        self.cancel = threading.Event()
        for record in chosen:
            self.results.pop(record.record_id, None)
            self.record_tree.set(record.record_id, "state", "Queued")
            for key in ("captures", "match", "miss", "extra", "precision", "recall", "event_recall", "event_precision"):
                self.record_tree.set(record.record_id, key, "—")
        self.run_button.configure(state="disabled")
        self.load_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.show_record(chosen[0].record_id)

        def progress(fraction, message):
            try:
                self.messages.put_nowait(("progress", (fraction, message)))
            except queue.Full:
                pass  # Display updates can coalesce; terminal outcomes cannot.

        def deliver(kind, value):
            while not self.closed:
                try:
                    self.messages.put((kind, value), timeout=0.1)
                    return
                except queue.Full:
                    continue

        def work():
            try:
                for record in chosen:
                    if self.cancel.is_set():
                        deliver("cancel", record.record_id)
                        continue
                    try:
                        result = run_smoke_record(
                            record, config, progress=progress, cancel_event=self.cancel
                        )
                        deliver("result", compact_smoke_result(result))
                        del result
                    except RunCancelled:
                        deliver("cancel", record.record_id)
                    except Exception as exc:
                        deliver("error", (record.record_id, f"{type(exc).__name__}: {exc}"))
            finally:
                deliver("done", None)

        self.thread = threading.Thread(target=work, name="smoke-dataset-ui", daemon=True)
        self.thread.start()

    def _poll(self):
        if self.closed:
            return
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == "progress":
                    self.status.set(value[1])
                elif kind == "result":
                    result = value
                    self.results[result.record.record_id] = result
                    m = result.metrics
                    for key, val in (
                        ("captures", m["captures"]),
                        ("match", m["matched"]),
                        ("miss", m["misses"]),
                        ("extra", m["extras"]),
                        ("precision", self._percent(m["sample_precision"])),
                        ("recall", self._percent(m["sample_recall"])),
                        ("event_recall", self._percent(m["event_recall"])),
                        ("event_precision", self._percent(m["event_precision"])),
                        ("state", "Complete" if not m["feature_failures"] else "Analysis failures"),
                    ):
                        self.record_tree.set(result.record.record_id, key, val)
                    # Respect user selection; do not redraw 480 figures or steal focus.
                    if getattr(self, "current_record_id", None) == result.record.record_id:
                        self.show_record(result.record.record_id)
                    self.status.set(
                        f"Completed {len(self.results)}/{len(self.records)} records. Latest: {result.record.record_id}. Profile: {result.config.detector_profile}. Offline characterization; false-alarm probability not certified."
                    )
                elif kind == "error":
                    self.record_tree.set(value[0], "state", "FAILED: " + value[1])
                    self.status.set(f"{value[0]} failed: {value[1]}")
                elif kind == "cancel":
                    self.record_tree.set(value, "state", "Cancelled; no completed evaluation")
                elif kind == "done":
                    self.run_button.configure(state="normal")
                    self.load_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")

                    def total(name):
                        return sum(r.metrics[name] for r in self.results.values())

                    failed = sum(
                        str(self.record_tree.set(i, "state")).startswith("FAILED")
                        for i in self.record_tree.get_children()
                    )
                    self.status.set(
                        f"{len(self.results)} completed records; {failed} failed. "
                        f"Activity matches {total('matched')}/{total('truth_activities')}; "
                        f"misses {total('misses')}, extras {total('extras')}. Export includes completed evaluations only."
                    )
        except queue.Empty:
            pass
        self.after(100, self._poll)

    @staticmethod
    def _percent(value):
        return "N/A" if value is None else f"{100 * value:.2f}%"

    def _record_selected(self, _event=None):
        selection = self.record_tree.selection()
        if selection:
            self.show_record(selection[0])

    def show_record(self, record_id):
        self.current_record_id = record_id
        a = self.annotations[record_id]
        result = self.results.get(record_id)
        for tree in (self.truth_tree, self.activity_tree, self.event_tree, self.feature_tree):
            self._clear(tree)
        for i, emitter in enumerate(a["emitters"]):
            coverage = self._percent(result.metrics["emitter_coverage"][i]) if result else "Not run"
            self.truth_tree.insert(
                "",
                "end",
                values=(
                    emitter["emitter_id"],
                    emitter["modulation"],
                    emitter["start_sample"],
                    emitter["end_sample"],
                    emitter["requested_snr_fullband_db"],
                    emitter["center_frequency_offset_hz"],
                    emitter["nominal_bandwidth_hz"],
                    coverage,
                ),
            )
        by_truth = {i: (j, score) for i, j, score in result.matches} if result else {}
        by_event = {j: i for i, j, _ in result.matches} if result else {}
        for i, activity in enumerate(a["fullband_activity_intervals"]):
            match = by_truth.get(i)
            event = result.captures[match[0]].event if match else None
            self.activity_tree.insert(
                "",
                "end",
                values=(
                    i,
                    activity["start_sample"],
                    activity["end_sample"],
                    event.event_id if event else "—",
                    event.detected_start_sample if event else "—",
                    event.detected_end_sample if event else "—",
                    f"{match[1]:.4f}" if match else "—",
                    event.detected_start_sample - activity["start_sample"] if event else "—",
                    event.detected_end_sample - activity["end_sample"] if event else "—",
                    "Matched" if match else "Missed" if result else "Not run",
                ),
            )
        if result:
            for j, c in enumerate(result.captures):
                e = c.event
                features = c.features.features if c.features else {}
                self.event_tree.insert(
                    "",
                    "end",
                    iid=str(j),
                    values=(
                        e.event_id,
                        e.detected_start_sample,
                        e.detected_end_sample,
                        e.event_start_sample,
                        e.event_end_sample,
                        e.pre_trigger_samples,
                        e.complete,
                        c.termination_reason,
                        c.feature_status,
                        f"{sum(f.valid for f in features.values())}/{len(features)}",
                        by_event.get(j, "Extra"),
                    ),
                )
            if result.captures:
                self.event_tree.selection_set("0")
                self._event_selected()
        self._plot(a, result)
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert(
            "end",
            "Emitter fields are supplied ground truth. Coverage means detected activity, "
            "not classification or source separation. Capture margins are excluded from detection scoring.\n"
            "All ends are exclusive. Sample metrics exclude the authorized calibration prefix.\n"
            "A missing sample-precision/recall denominator is N/A, not 100%.\n\n",
        )
        if result:
            self.details.insert("end", json.dumps(result.metrics, indent=2) + "\n\n")
        self.details.insert("end", json.dumps(a, indent=2))
        self.details.configure(state="disabled")

    def _plot(self, a, result):
        top, lower = self.axes
        for axis in self.axes:
            axis.clear()
            axis.grid(True, alpha=0.2)
        fs = a["sample_rate_hz"]
        scale = 1000 / fs
        if result:
            top.plot(
                result.display_samples * scale, result.display_magnitude, color="#386780", lw=0.6
            )
        top.axvspan(
            0,
            a["calibration_prefix"]["end_sample"] * scale,
            color="gray",
            alpha=0.2,
            label="Calibration",
        )
        for i, t in enumerate(a["fullband_activity_intervals"]):
            top.axvspan(
                t["start_sample"] * scale,
                t["end_sample"] * scale,
                color="#2281bc",
                alpha=0.18,
                label="Truth activity" if i == 0 else None,
            )
        if result:
            for i, capture in enumerate(result.captures):
                e = capture.event
                top.axvspan(
                    e.detected_start_sample * scale,
                    e.detected_end_sample * scale,
                    color="#e79420",
                    alpha=0.2,
                    label="Detected" if i == 0 else None,
                )
                lower.broken_barh(
                    [
                        (
                            e.detected_start_sample * scale,
                            (e.detected_end_sample - e.detected_start_sample) * scale,
                        )
                    ],
                    (-0.3, 0.6),
                    facecolors="#e79420",
                )
        for i, e in enumerate(a["emitters"]):
            lower.broken_barh(
                [(e["start_sample"] * scale, (e["end_sample"] - e["start_sample"]) * scale)],
                (i + 0.7, 0.6),
                facecolors="#2281bc",
            )
        lower.set_yticks(
            range(len(a["emitters"]) + 1),
            ["Detected activity"]
            + [f"Truth {i}: {e['modulation']}" for i, e in enumerate(a["emitters"])],
        )
        top.set_title(
            f"{a['record_id']} • {a['scenario']} • {a['background']['model']} | blue: truth; amber: output"
        )
        top.set_ylabel("Peak |IQ| / preview bin")
        top.legend(loc="upper right", fontsize=8)
        lower.set_xlabel("Record time (ms)")
        lower.set_xlim(0, a["num_samples"] * scale)
        self.canvas.draw_idle()

    def _event_selected(self, _event=None):
        selection = self.event_tree.selection()
        result = self.results.get(getattr(self, "current_record_id", ""))
        if not selection or not result:
            return
        capture = result.captures[int(selection[0])]
        self._clear(self.feature_tree)
        if capture.features:
            for name, f in sorted(capture.features.features.items()):
                self.feature_tree.insert(
                    "",
                    "end",
                    values=(
                        name,
                        f"{f.value:.10g}" if f.valid else "—",
                        f.unit or "",
                        f.valid,
                        f.reason or f.interpretation or "",
                    ),
                )

    def export(self):
        if not self.results:
            messagebox.showinfo(
                "No completed results", "Run at least one record before exporting comparisons."
            )
            return
        mode = self.export_mode.get()
        path = filedialog.asksaveasfilename(
            title="Export ground truth and pipeline results",
            initialfile=f"{Path(self.path.get()).name}_{mode}.csv",
            defaultextension=".csv",
            filetypes=(("CSV", "*.csv"),),
        )
        if path:
            try:
                export_smoke_csv(path, tuple(self.results.values()), mode)
                self.status.set(f"Exported {mode}: {path}")
            except Exception as exc:
                messagebox.showerror("CSV export failed", str(exc))

    def close(self):
        self.closed = True
        self.cancel.set()
