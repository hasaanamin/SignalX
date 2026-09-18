from __future__ import annotations
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    _pkg_dir = Path(__file__).resolve().parent
    _src_dir = _pkg_dir.parent.parent
    for _p in [str(_src_dir), str(_pkg_dir.parent)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import importlib
    importlib.import_module(f"packages.{_pkg_dir.name}")
    __package__ = f"packages.{_pkg_dir.name}"

from typing import Any
import numpy as np

from .base import BaseClusteringMetric, ClusteringMetricFactory, IQEvent, EmitterTrack, TrackSequenceResult
from ._common import (
    FEATURE_NAMES,
    extract_events_from_input,
)


@ClusteringMetricFactory.register([
    "track_sequence_association",
    "pulse_tracking",
    "track_association",
    "tracksequenceassociation",
    "emitter_tracker",
    "stream_tracker",
])
class TrackSequenceAssociation(BaseClusteringMetric):
    """
    Sequential Pulse Stream Tracking & Association Processor.

    Maintains a continuous emitter track table across streaming pulse detections.
    Applies alpha-beta kinematic/PRI prediction filtering to anticipate subsequent pulse arrival
    times (TOA), establishes track confirmation via M-of-N criteria, and manages track coasting/drop.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        pri_tolerance_percent: float = 15.0,
        freq_tolerance_hz: float | None = 100e3,
        pw_tolerance_percent: float = 25.0,
        alpha: float = 0.6,
        beta: float = 0.3,
        confirmation_m_of_n: tuple[int, int] = (3, 4),
        max_coasting_misses: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)

        self.pri_tolerance_percent = float(pri_tolerance_percent)
        if self.pri_tolerance_percent <= 0:
            raise ValueError(f"pri_tolerance_percent must be > 0, got {pri_tolerance_percent}")

        self.freq_tolerance_hz = float(freq_tolerance_hz) if freq_tolerance_hz is not None else None
        if self.freq_tolerance_hz is not None and self.freq_tolerance_hz <= 0:
            raise ValueError(f"freq_tolerance_hz must be > 0, got {freq_tolerance_hz}")

        self.pw_tolerance_percent = float(pw_tolerance_percent)
        if self.pw_tolerance_percent <= 0:
            raise ValueError(f"pw_tolerance_percent must be > 0, got {pw_tolerance_percent}")

        self.alpha = float(alpha)
        self.beta = float(beta)
        if not (0.0 < self.alpha <= 1.0) or not (0.0 <= self.beta <= 1.0):
            raise ValueError("alpha and beta filter gains must be in (0, 1] and [0, 1].")

        m_req, n_win = confirmation_m_of_n
        if not (1 <= m_req <= n_win):
            raise ValueError(f"confirmation_m_of_n must satisfy 1 <= M <= N, got ({m_req}, {n_win})")

        self.confirmation_m_of_n = (int(m_req), int(n_win))
        self.max_coasting_misses = int(max_coasting_misses)

    @property
    def name(self) -> str:
        return "track_sequence_association"

    def compute(self, data: Any, **kwargs: Any) -> TrackSequenceResult:
        """
        Executes sequential pulse stream tracking.

        Parameters
        ----------
        data : complex ndarray, list of PulseCandidate, list of IQEvent, dict, or 2D feature matrix.

        Returns
        -------
        TrackSequenceResult
        """
        sample_rate = float(kwargs.get("sample_rate", self.sample_rate))
        pri_tol_pct = float(kwargs.get("pri_tolerance_percent", self.pri_tolerance_percent))
        freq_tol = kwargs.get("freq_tolerance_hz", self.freq_tolerance_hz)
        freq_tol_val = float(freq_tol) if freq_tol is not None else None
        pw_tol_pct = float(kwargs.get("pw_tolerance_percent", self.pw_tolerance_percent))
        alpha = float(kwargs.get("alpha", self.alpha))
        beta = float(kwargs.get("beta", self.beta))
        m_req, n_win = kwargs.get(
            "confirmation_m_of_n",
            kwargs.get("m_of_n_confirm", self.config.get("m_of_n_confirm", self.confirmation_m_of_n)),
        )
        max_misses = int(
            kwargs.get(
                "max_coasting_misses",
                kwargs.get(
                    "max_consecutive_misses",
                    self.config.get("max_consecutive_misses", self.max_coasting_misses),
                ),
            )
        )

        events, _, _ = extract_events_from_input(
            data=data,
            sample_rate=sample_rate,
            feature_names=FEATURE_NAMES,
            **kwargs,
        )

        n_events = len(events)
        if n_events == 0:
            return TrackSequenceResult(
                events=[],
                tracks=[],
                pulse_track_map=np.empty(0, dtype=int),
                num_active_tracks=0,
                num_confirmed_tracks=0,
                sample_rate=sample_rate,
            )

        # Issue 8: Check for real TOA values in temporal tracking
        for ev in events:
            if getattr(ev, "is_synthetic_time", False):
                raise ValueError("TrackSequenceAssociation requires real TOA values.")

        # Sort pulses by TOA
        sorted_indices = sorted(range(n_events), key=lambda i: events[i].start_time_sec)

        pulse_track_map = np.full(n_events, -1, dtype=int)
        tracks: list[EmitterTrack] = []
        track_id_counter = 0

        # Sequential Pulse Association Loop
        for p_idx in sorted_indices:
            ev = events[p_idx]
            toa = ev.start_time_sec
            # Prefer carrier_freq_hz over spectral_centroid_hz
            fc = ev.carrier_freq_hz if ev.carrier_freq_hz is not None else ev.spectral_centroid_hz
            pw = ev.duration_sec

            # 1. Update coasting and misses for opportunities that elapsed before current pulse
            for trk in tracks:
                if trk.state != "terminated" and len(trk.toas) >= 2 and trk.estimated_pri_sec > 0:
                    pri_est = trk.estimated_pri_sec
                    elapsed = toa - trk.last_toa_sec
                    expected_misses = int(np.floor((elapsed - 0.2 * pri_est) / max(pri_est, 1e-12)))
                    new_misses = expected_misses - trk.misses
                    if new_misses > 0:
                        for _ in range(new_misses):
                            trk.misses += 1
                            trk.total_updates += 1
                            trk.extra.setdefault("rolling_history", []).append(False)
                            if trk.state == "confirmed":
                                trk.state = "coasting"
                            if trk.misses >= max_misses:
                                trk.state = "terminated"
                                break

            best_track_idx = -1
            best_cost = float("inf")

            for t_i, trk in enumerate(tracks):
                if trk.state == "terminated":
                    continue

                # Check frequency match
                if freq_tol_val is not None and abs(fc - trk.estimated_fc_hz) > freq_tol_val:
                    continue

                # Check pulse width match
                pw_err_pct = (abs(pw - trk.estimated_pw_sec) / max(trk.estimated_pw_sec, 1e-12)) * 100.0
                if pw_err_pct > pw_tol_pct:
                    continue

                # Check TOA/PRI consistency
                if len(trk.toas) == 1:
                    # Second pulse: sets initial PRI estimate
                    dt = toa - trk.last_toa_sec
                    if dt > 0:
                        cost = abs(fc - trk.estimated_fc_hz) / max(freq_tol_val if freq_tol_val else 1.0, 1.0)
                        if cost < best_cost:
                            best_cost = cost
                            best_track_idx = t_i
                else:
                    # Third or later pulse: test predicted TOA and recent stagger intervals
                    pri_est = max(trk.estimated_pri_sec, 1e-12)
                    dt = toa - trk.last_toa_sec
                    recent_pris = [trk.toas[i] - trk.toas[i - 1] for i in range(1, len(trk.toas))]
                    candidates = list(set(recent_pris[-6:] + [pri_est]))

                    best_cand_err = float("inf")
                    best_cand_tol = (2.5 * pri_tol_pct / 100.0) * pri_est

                    for p_cand in candidates:
                        k_h = max(1, int(round(dt / max(p_cand, 1e-12))))
                        cand_err = abs(dt - k_h * p_cand)
                        cand_tol = max((2.5 * pri_tol_pct / 100.0) * (k_h * p_cand), 1e-6)
                        if cand_err <= cand_tol and cand_err < best_cand_err:
                            best_cand_err = cand_err
                            best_cand_tol = cand_tol

                    if best_cand_err <= best_cand_tol:
                        f_cost = (abs(fc - trk.estimated_fc_hz) / max(freq_tol_val, 1.0)) if freq_tol_val else 0.0
                        cost = (best_cand_err / best_cand_tol) + f_cost
                        if cost < best_cost:
                            best_cost = cost
                            best_track_idx = t_i

            # Association or New Track Initiation
            if best_track_idx >= 0:
                trk = tracks[best_track_idx]
                dt = toa - trk.last_toa_sec
                trk.pulse_indices.append(p_idx)
                trk.toas.append(toa)
                pulse_track_map[p_idx] = trk.track_id
                trk.hits += 1
                trk.misses = 0
                trk.total_updates += 1
                trk.extra.setdefault("rolling_history", []).append(True)

                if len(trk.toas) == 2:
                    trk.estimated_pri_sec = float(dt)
                    trk.predicted_next_toa_sec = toa + trk.estimated_pri_sec
                else:
                    # Alpha-Beta Update
                    k_harmonic = max(1, int(round(dt / max(trk.estimated_pri_sec, 1e-12))))
                    pred_toa = trk.last_toa_sec + k_harmonic * trk.estimated_pri_sec
                    res = toa - pred_toa

                    trk.predicted_next_toa_sec = toa + trk.estimated_pri_sec + alpha * res
                    trk.estimated_pri_sec = max(1e-12, trk.estimated_pri_sec + (beta * res / k_harmonic))

                # Update PRI Jitter
                pris = np.diff(trk.toas)
                if len(pris) > 1:
                    trk.estimated_pri_jitter = float((np.std(pris) / max(np.mean(pris), 1e-12)) * 100.0)

                # Update running RF feature averages
                n_pts = len(trk.pulse_indices)
                trk.estimated_fc_hz = trk.estimated_fc_hz + (fc - trk.estimated_fc_hz) / n_pts
                trk.estimated_pw_sec = trk.estimated_pw_sec + (pw - trk.estimated_pw_sec) / n_pts
                trk.last_toa_sec = toa

                # True rolling M-of-N confirmation & reacquisition
                hist = trk.extra.get("rolling_history", [])
                win_hits = sum(hist[-n_win:])
                if win_hits >= m_req:
                    trk.state = "confirmed"
                elif trk.state == "coasting":
                    trk.state = "confirmed"
            else:
                # Initiate new tentative track
                new_track = EmitterTrack(
                    track_id=track_id_counter,
                    state="confirmed" if m_req <= 1 else "tentative",
                    pulse_indices=[p_idx],
                    toas=[toa],
                    estimated_pri_sec=0.0,
                    estimated_pri_jitter=0.0,
                    estimated_fc_hz=float(fc),
                    estimated_pw_sec=float(pw),
                    last_toa_sec=float(toa),
                    predicted_next_toa_sec=float(toa),
                    hits=1,
                    misses=0,
                    total_updates=1,
                    extra={"rolling_history": [True]},
                )
                pulse_track_map[p_idx] = track_id_counter
                tracks.append(new_track)
                track_id_counter += 1

        # Post-loop: evaluate elapsed misses up to the final event timestamp
        if len(sorted_indices) > 0:
            max_sim_time = events[sorted_indices[-1]].start_time_sec
            for trk in tracks:
                if trk.state != "terminated" and len(trk.toas) >= 2 and trk.estimated_pri_sec > 0:
                    dt = max_sim_time - trk.last_toa_sec
                    expected_misses = int(np.floor((dt + 0.2 * trk.estimated_pri_sec) / max(trk.estimated_pri_sec, 1e-12)))
                    new_misses = expected_misses - trk.misses
                    if new_misses > 0:
                        for _ in range(new_misses):
                            trk.misses += 1
                            trk.total_updates += 1
                            trk.extra.setdefault("rolling_history", []).append(False)
                            if trk.state == "confirmed":
                                trk.state = "coasting"
                            if trk.misses >= max_misses:
                                trk.state = "terminated"
                                break

        active_tracks = [t for t in tracks if t.state in ("tentative", "confirmed", "coasting")]
        confirmed_tracks = [t for t in tracks if t.state == "confirmed" or (t.state == "coasting" and t.hits >= m_req)]

        return TrackSequenceResult(
            events=events,
            tracks=tracks,
            pulse_track_map=pulse_track_map,
            num_active_tracks=len(active_tracks),
            num_confirmed_tracks=len(confirmed_tracks),
            sample_rate=sample_rate,
        )


def compute_track_sequence_association(
    data: Any,
    sample_rate: float = 1.0,
    pri_tolerance_percent: float = 15.0,
    freq_tolerance_hz: float | None = 100e3,
    pw_tolerance_percent: float = 25.0,
    alpha: float = 0.6,
    beta: float = 0.3,
    confirmation_m_of_n: tuple[int, int] = (3, 4),
    max_coasting_misses: int = 5,
    **kwargs: Any,
) -> TrackSequenceResult:
    """Convenience functional API for sequential pulse stream tracking."""
    processor = TrackSequenceAssociation(
        sample_rate=sample_rate,
        pri_tolerance_percent=pri_tolerance_percent,
        freq_tolerance_hz=freq_tolerance_hz,
        pw_tolerance_percent=pw_tolerance_percent,
        alpha=alpha,
        beta=beta,
        confirmation_m_of_n=confirmation_m_of_n,
        max_coasting_misses=max_coasting_misses,
        **kwargs,
    )
    return processor.compute(data)


def main() -> None:
    print("=" * 60)
    print("  TrackSequenceAssociation: Demonstration")
    print("=" * 60)
    pulses = [
        {"start_time_sec": 0.001 + i * 0.001, "duration_sec": 20e-6, "spectral_centroid_hz": 300e3}
        for i in range(6)
    ]
    metric = TrackSequenceAssociation()
    result = metric.compute(pulses)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
