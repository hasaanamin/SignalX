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

from .base import BaseClusteringMetric, ClusteringMetricFactory, IQEvent, EmitterGroupResult
from ._common import (
    FEATURE_NAMES,
    extract_events_from_input,
    normalize_features,
    resolve_feature_weights,
    perform_pulse_clustering,
    analyze_pri_sequence,
)


@ClusteringMetricFactory.register([
    "emitter_grouping",
    "emitter_group",
    "emitter_association",
    "emittergrouping",
    "emitter_classifier",
    "radar_emitter_grouping",
])
class EmitterGrouping(BaseClusteringMetric):
    """
    Radar & RF Communications Emitter Grouping Processor.

    Partitions interleaved pulse streams into discrete emitter sources and extracts
    individual emitter characterizations including PRI modulation mode (constant, staggered,
    jittered, agile), PRI sequences, carrier frequency agility, and pulse width spread.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        distance_threshold: float = 1.5,
        n_emitters: int | None = None,
        linkage: str = "average",
        feature_weights: dict[str, float] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)

        self.distance_threshold = float(distance_threshold)
        if self.distance_threshold <= 0 or not np.isfinite(self.distance_threshold):
            raise ValueError(f"distance_threshold must be a positive finite float, got {distance_threshold}")

        self.n_emitters = int(n_emitters) if n_emitters is not None else None
        if self.n_emitters is not None and self.n_emitters < 1:
            raise ValueError(f"n_emitters must be >= 1 when specified, got {n_emitters}")

        self.linkage = str(linkage).lower().strip()
        if self.linkage not in ("average", "complete", "single", "ward"):
            raise ValueError(f"Unsupported linkage '{linkage}'. Expected one of: 'average', 'complete', 'single', 'ward'.")

        self.feature_weights = feature_weights

    @property
    def name(self) -> str:
        return "emitter_grouping"

    def compute(self, data: Any, **kwargs: Any) -> EmitterGroupResult:
        """
        Groups pulse events into emitter entities and calculates emitter radar descriptors.

        Parameters
        ----------
        data : complex ndarray, list of PulseCandidate, list of IQEvent, dict, or 2D feature matrix.

        Returns
        -------
        EmitterGroupResult
        """
        sample_rate = float(kwargs.get("sample_rate", self.sample_rate))
        distance_threshold = float(kwargs.get("distance_threshold", self.distance_threshold))
        n_emitters = kwargs.get("n_emitters", self.n_emitters)
        linkage_method = str(kwargs.get("linkage", self.linkage)).lower().strip()
        custom_weights = kwargs.get("feature_weights", self.feature_weights)

        events, raw_features, feat_names = extract_events_from_input(
            data=data,
            sample_rate=sample_rate,
            feature_names=FEATURE_NAMES,
            **kwargs,
        )

        n_events = len(events)
        if n_events == 0:
            return EmitterGroupResult(
                events=[],
                emitter_ids=np.empty(0, dtype=int),
                num_emitters=0,
                emitter_profiles=[],
                raw_features=np.empty((0, len(feat_names)), dtype=np.float64),
                normalized_features=np.empty((0, len(feat_names)), dtype=np.float64),
                confidence=np.empty(0, dtype=np.float64),
                sample_rate=sample_rate,
            )

        norm_features = normalize_features(raw_features, sample_rate=sample_rate, feature_names=feat_names)
        weights_arr = resolve_feature_weights(feat_names, custom_weights)

        cluster_res = perform_pulse_clustering(
            events=events,
            raw_features=raw_features,
            norm_features=norm_features,
            feature_names=feat_names,
            feature_weights=weights_arr,
            distance_threshold=distance_threshold,
            n_clusters=n_emitters,
            linkage_method=linkage_method,
            sample_rate=sample_rate,
        )

        emitter_ids = cluster_res.cluster_labels.copy()
        num_emitters = cluster_res.num_clusters

        # Issue 13: Timing-assisted temporal PRI deinterleaving if static features are identical
        can_deinterleave = (n_emitters is not None and n_emitters > num_emitters and n_events >= n_emitters) or (num_emitters == 1 and n_events >= 8)
        if can_deinterleave:
            # Check if timing information is available to deinterleave
            all_toas = np.array([ev.start_time_sec for ev in events], dtype=np.float64)
            has_real_timing = not any(getattr(ev, "is_synthetic_time", False) for ev in events) and len(np.unique(all_toas)) > 1

            if has_real_timing:
                # Find the largest cluster to temporally deinterleave
                unique_clusters, cluster_counts = np.unique(emitter_ids, return_counts=True)
                largest_c = unique_clusters[np.argmax(cluster_counts)]
                c_indices = [i for i, lbl in enumerate(emitter_ids) if lbl == largest_c]
                sorted_c_idx = sorted(c_indices, key=lambda i: events[i].start_time_sec)
                c_toas = np.array([events[i].start_time_sec for i in sorted_c_idx], dtype=np.float64)

                # Search for candidate PRIs via pairwise difference histogram
                diffs = []
                for i in range(len(c_toas)):
                    for j in range(i + 1, min(i + 15, len(c_toas))):
                        dt = c_toas[j] - c_toas[i]
                        if dt > 1e-6:
                            diffs.append(dt)

                if len(diffs) > 0:
                    diffs_arr = np.array(diffs, dtype=np.float64)
                    q_diffs = np.round(diffs_arr, decimals=6)
                    u_vals, u_counts = np.unique(q_diffs, return_counts=True)

                    # Sort candidates by frequency of occurrence and filter out harmonics
                    sorted_by_count = [u_vals[idx] for idx in np.argsort(-u_counts) if u_counts[idx] >= 3 and u_vals[idx] > 1e-6]
                    fund_candidates = []
                    for cand in sorted_by_count:
                        is_harm = False
                        for f in fund_candidates:
                            k_h1 = round(cand / f)
                            if k_h1 >= 2 and abs(cand - k_h1 * f) <= 0.08 * cand:
                                is_harm = True
                                break
                            k_h2 = round(f / cand)
                            if k_h2 >= 2 and abs(f - k_h2 * cand) <= 0.08 * f:
                                is_harm = True
                                break
                        if not is_harm:
                            fund_candidates.append(cand)

                    target_k = n_emitters if n_emitters is not None else min(len(fund_candidates), 4)
                    if len(fund_candidates) >= 2 and target_k >= 2:
                        sub_labels = np.full(len(sorted_c_idx), -1, dtype=int)
                        cur_sub_id = 0

                        for pri_cand in fund_candidates:
                            pri_tol = max(0.12 * pri_cand, 1e-6)
                            for start_k in range(len(sorted_c_idx)):
                                if sub_labels[start_k] != -1:
                                    continue
                                train_members = [start_k]
                                last_t = c_toas[start_k]
                                for next_k in range(start_k + 1, len(sorted_c_idx)):
                                    if sub_labels[next_k] != -1:
                                        continue
                                    dt = c_toas[next_k] - last_t
                                    k_harm = max(1, int(round(dt / pri_cand)))
                                    if abs(dt - k_harm * pri_cand) <= (k_harm * pri_tol):
                                        train_members.append(next_k)
                                        last_t = c_toas[next_k]

                                if len(train_members) >= 3:
                                    for m in train_members:
                                        sub_labels[m] = cur_sub_id
                                    cur_sub_id += 1
                                    break
                            if cur_sub_id >= target_k:
                                break

                        # Assign any remaining unassigned pulses to closest sub-train
                        for k_idx, s_lbl in enumerate(sub_labels):
                            if s_lbl == -1:
                                t_k = c_toas[k_idx]
                                best_s = 0
                                best_res = float("inf")
                                for s in range(cur_sub_id):
                                    s_toas = [c_toas[m] for m in range(len(sorted_c_idx)) if sub_labels[m] == s]
                                    if len(s_toas) >= 2:
                                        s_pri = float(np.median(np.diff(s_toas)))
                                        min_dt = min(abs((t_k - st) - round((t_k - st) / max(s_pri, 1e-12)) * s_pri) for st in s_toas)
                                        if min_dt < best_res:
                                            best_res = min_dt
                                            best_s = s
                                sub_labels[k_idx] = best_s

                        if len(set(sub_labels)) > 1:
                            new_em_ids = emitter_ids.copy()
                            for k_idx, orig_i in enumerate(sorted_c_idx):
                                new_em_ids[orig_i] = int(sub_labels[k_idx])
                            u_new = sorted(list(set(new_em_ids)))
                            remap = {old: new for new, old in enumerate(u_new)}
                            emitter_ids = np.array([remap[l] for l in new_em_ids], dtype=int)
                            num_emitters = len(u_new)

        emitter_profiles: list[dict[str, Any]] = []

        fc_idx = feat_names.index("spectral_centroid_hz") if "spectral_centroid_hz" in feat_names else 0
        bw_idx = feat_names.index("bandwidth_hz") if "bandwidth_hz" in feat_names else 1
        pw_idx = feat_names.index("duration_sec") if "duration_sec" in feat_names else 2
        pwr_idx = feat_names.index("mean_power_db") if "mean_power_db" in feat_names else 3

        for em_id in range(num_emitters):
            pulse_indices = [i for i, lbl in enumerate(emitter_ids) if lbl == em_id]
            em_events = [events[i] for i in pulse_indices]
            em_toas = [ev.start_time_sec for ev in em_events]

            # PRI Sequence Analysis
            pri_info = analyze_pri_sequence(em_toas)

            # Feature statistics
            em_raw = raw_features[pulse_indices]
            fc_mean = float(np.mean(em_raw[:, fc_idx]))
            fc_spread = float(np.std(em_raw[:, fc_idx])) if len(pulse_indices) > 1 else 0.0

            carrier_freqs = [ev.carrier_freq_hz for ev in em_events if ev.carrier_freq_hz is not None]
            if len(carrier_freqs) > 0:
                carrier_fc_mean = float(np.mean(carrier_freqs))
                carrier_fc_spread = float(np.std(carrier_freqs)) if len(carrier_freqs) > 1 else 0.0
            else:
                carrier_fc_mean = fc_mean
                carrier_fc_spread = fc_spread

            centroids = [ev.spectral_centroid_hz for ev in em_events if ev.spectral_centroid_hz is not None]
            if len(centroids) > 0:
                centroid_fc_mean = float(np.mean(centroids))
                centroid_fc_spread = float(np.std(centroids)) if len(centroids) > 1 else 0.0
            else:
                centroid_fc_mean = fc_mean
                centroid_fc_spread = fc_spread

            bw_mean = float(np.mean(em_raw[:, bw_idx]))
            pw_mean = float(np.mean(em_raw[:, pw_idx]))
            pw_spread = float(np.std(em_raw[:, pw_idx])) if len(pulse_indices) > 1 else 0.0

            pwr_mean = float(np.mean(em_raw[:, pwr_idx]))

            profile = {
                "emitter_id": int(em_id),
                "pulse_count": int(len(pulse_indices)),
                "pulse_indices": pulse_indices,
                "fc_mean_hz": carrier_fc_mean,
                "fc_spread_hz": carrier_fc_spread,
                "carrier_freq_mean_hz": carrier_fc_mean,
                "carrier_freq_spread_hz": carrier_fc_spread,
                "spectral_centroid_mean_hz": centroid_fc_mean,
                "spectral_centroid_spread_hz": centroid_fc_spread,
                "bw_mean_hz": bw_mean,
                "pw_mean_sec": pw_mean,
                "pw_spread_sec": pw_spread,
                "power_mean_db": pwr_mean,
                "pri_mean_sec": float(pri_info["pri_mean_sec"]),
                "pri_std_sec": float(pri_info["pri_std_sec"]),
                "pri_min_sec": float(pri_info["pri_min_sec"]),
                "pri_max_sec": float(pri_info["pri_max_sec"]),
                "pri_jitter_percent": float(pri_info["pri_jitter_percent"]),
                "pri_mode": str(pri_info["pri_mode"]),
                "pri_sequence": pri_info["pri_sequence"],
            }
            emitter_profiles.append(profile)

        return EmitterGroupResult(
            events=events,
            emitter_ids=emitter_ids,
            num_emitters=num_emitters,
            emitter_profiles=emitter_profiles,
            raw_features=raw_features,
            normalized_features=norm_features,
            confidence=cluster_res.confidence,
            sample_rate=sample_rate,
        )


def compute_emitter_grouping(
    data: Any,
    sample_rate: float = 1.0,
    distance_threshold: float = 1.5,
    n_emitters: int | None = None,
    linkage: str = "average",
    feature_weights: dict[str, float] | np.ndarray | None = None,
    **kwargs: Any,
) -> EmitterGroupResult:
    """Convenience functional API for emitter grouping."""
    processor = EmitterGrouping(
        sample_rate=sample_rate,
        distance_threshold=distance_threshold,
        n_emitters=n_emitters,
        linkage=linkage,
        feature_weights=feature_weights,
        **kwargs,
    )
    return processor.compute(data)


def main() -> None:
    print("=" * 60)
    print("  EmitterGrouping: Demonstration")
    print("=" * 60)
    pulses = [
        {"toa": 0.001, "freq": 1e9, "pw": 1e-6, "pri": 100e-6, "amp": 1.0},
        {"toa": 0.0011, "freq": 1e9, "pw": 1e-6, "pri": 100e-6, "amp": 1.0},
        {"toa": 0.002, "freq": 2e9, "pw": 2e-6, "pri": 200e-6, "amp": 0.8},
        {"toa": 0.0022, "freq": 2e9, "pw": 2e-6, "pri": 200e-6, "amp": 0.8},
    ]
    metric = EmitterGrouping()
    result = metric.compute(pulses)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
