"""SignalContext coordinator and memoization container for RF signal analysis."""

from typing import Any

import numpy as np

from signal_processing.context.config import AnalysisConfig
from signal_processing.context.exceptions import ContextError
from signal_processing.core import (
    AutocorrelationResult,
    CaptureQuality,
    PhaseResult,
    PSDResult,
    SignalEvent,
    StreamInfo,
)
from signal_processing.transforms import (
    FFTResult,
    compute_autocorrelation,
    compute_envelope,
    compute_fft,
    compute_instantaneous_frequency,
    compute_mean_power_dbfs,
    compute_mean_power_linear,
    compute_phase,
    compute_power,
    compute_psd_from_fft,
    compute_welch_psd,
)


class SignalContext:
    """Coordinator and memoization container for DSP transforms on a SignalEvent.

    A SignalContext is bound to exactly one immutable SignalEvent instance and
    an immutable AnalysisConfig. It lazily calculates and caches intermediate
    DSP transformations (envelope, power, phase, instantaneous frequency, FFT,
    PSD, autocorrelation) operating strictly on the detected signal region
    (`event.signal_iq`).

    Attributes/Properties:
        event: The underlying immutable SignalEvent.
        config: The AnalysisConfig specifying transform parameters.
        stream: Stream metadata (StreamInfo) from the event.
        event_id: Unique event identifier string.
        sample_rate_hz: Sampling frequency in Hz from StreamInfo.
        center_frequency_hz: Center RF frequency in Hz from StreamInfo.
        signal_iq: Detected signal IQ array (read-only view of active signal).
        event_iq: Entire captured event IQ array (including pre/post margins).
        signal_num_samples: Number of samples in the detected signal region.
        signal_duration_s: Duration in seconds of the detected signal region.
        cached_keys: Read-only frozenset of transform keys currently present in cache.
    """

    def __init__(
        self,
        event: SignalEvent,
        config: AnalysisConfig | None = None,
        capture_quality: CaptureQuality | None = None,
    ) -> None:
        """Initialize a SignalContext for a SignalEvent.

        Args:
            event: An immutable SignalEvent instance.
            config: Optional AnalysisConfig specifying transform settings.
                Defaults to AnalysisConfig() if None.
            capture_quality: Optional CaptureQuality metadata. If None, derived
                automatically from event.complete and event.dropped_samples.

        Raises:
            ContextError: If event is not a SignalEvent, config is not an AnalysisConfig,
                or capture_quality is not a CaptureQuality.
        """
        if not isinstance(event, SignalEvent):
            raise ContextError(f"Expected SignalEvent instance, got {type(event).__name__}")

        if config is None:
            resolved_config = AnalysisConfig()
        elif isinstance(config, AnalysisConfig):
            resolved_config = config
        else:
            raise ContextError(
                f"Expected AnalysisConfig instance or None, got {type(config).__name__}"
            )

        if capture_quality is not None:
            if not isinstance(capture_quality, CaptureQuality):
                raise ContextError(
                    f"Expected CaptureQuality instance or None, got {type(capture_quality).__name__}"
                )
            self._capture_quality = capture_quality
        else:
            is_cont = bool(event.dropped_samples == 0)
            warns = ()
            if not event.complete or not is_cont:
                warns = (f"Incomplete event capture: dropped_samples={event.dropped_samples}",)
            self._capture_quality = CaptureQuality(
                complete=event.complete,
                dropped_samples=event.dropped_samples,
                is_continuous=is_cont,
                warnings=warns,
            )

        self._event = event
        self._config = resolved_config
        self._cache: dict[str, Any] = {}

    @property
    def event(self) -> SignalEvent:
        """The underlying immutable SignalEvent."""
        return self._event

    @property
    def config(self) -> AnalysisConfig:
        """The immutable analysis transform configuration."""
        return self._config

    @property
    def capture_quality(self) -> CaptureQuality:
        """The capture completeness, continuity, and quality metadata."""
        return self._capture_quality

    @property
    def stream(self) -> StreamInfo:
        """Stream metadata contract from the SignalEvent."""
        return self._event.stream

    @property
    def event_id(self) -> str:
        """Unique event identifier string."""
        return self._event.event_id

    @property
    def sample_rate_hz(self) -> float:
        """Sampling rate in Hertz from the stream metadata."""
        return self._event.stream.sample_rate_hz

    @property
    def center_frequency_hz(self) -> float:
        """Receiver tuned center frequency in Hertz from the stream metadata."""
        return self._event.stream.center_frequency_hz

    @property
    def signal_iq(self) -> np.ndarray:
        """Read-only view of the complex IQ samples belonging strictly to detected signal."""
        return self._event.signal_iq

    @property
    def event_iq(self) -> np.ndarray:
        """Read-only array of the full captured event buffer (including pre/post margins)."""
        return self._event.iq

    @property
    def signal_num_samples(self) -> int:
        """Number of samples in the detected active signal region."""
        return self._event.signal_samples

    @property
    def signal_duration_s(self) -> float:
        """Duration in seconds of the detected active signal region."""
        return self._event.signal_duration_s

    @property
    def cached_keys(self) -> frozenset[str]:
        """Read-only view of currently cached transform keys (for inspection/testing)."""
        return frozenset(self._cache.keys())

    def envelope(self) -> np.ndarray:
        """Compute or retrieve cached instantaneous amplitude envelope of signal_iq.

        Returns:
            1-D real-valued read-only float array of envelope samples: |signal_iq[n]|.
        """
        if "envelope" not in self._cache:
            self._cache["envelope"] = compute_envelope(self.signal_iq)
        return self._cache["envelope"]

    def power(self) -> np.ndarray:
        """Compute or retrieve cached instantaneous linear power of signal_iq.

        Returns:
            1-D real-valued read-only float array of instantaneous linear power: |signal_iq[n]|^2.
        """
        if "power" not in self._cache:
            self._cache["power"] = compute_power(self.signal_iq)
        return self._cache["power"]

    def mean_power_linear(self) -> float:
        """Compute or retrieve cached average (mean) linear power of signal_iq.

        Reuses the cached instantaneous power array if already computed, or computes
        via compute_mean_power_linear.

        Returns:
            Scalar finite float representing average linear power: (1/N) * sum(|signal_iq|^2).
        """
        if "mean_power_linear" not in self._cache:
            if "power" in self._cache:
                self._cache["mean_power_linear"] = float(np.mean(self._cache["power"]))
            else:
                self._cache["mean_power_linear"] = compute_mean_power_linear(self.signal_iq)
        return self._cache["mean_power_linear"]

    def mean_power_dbfs(self) -> float:
        """Compute or retrieve cached average (mean) power of signal_iq in dBFS.

        Reuses `self.mean_power_linear()` and converts using configured power floor.

        Returns:
            Scalar finite float representing average power in dBFS relative to full-scale 1.0.
        """
        if "mean_power_dbfs" not in self._cache:
            p_lin = self.mean_power_linear()
            self._cache["mean_power_dbfs"] = compute_mean_power_dbfs(
                self.signal_iq,
                floor_dbfs=self._config.power_floor_dbfs,
                mean_power_linear=p_lin,
            )
        return self._cache["mean_power_dbfs"]

    def phase(self) -> PhaseResult:
        """Compute or retrieve cached wrapped and unwrapped phase of signal_iq.

        Returns:
            Immutable PhaseResult contract containing wrapped_rad and unwrapped_rad.
        """
        if "phase" not in self._cache:
            self._cache["phase"] = compute_phase(
                self.signal_iq,
                amplitude_threshold=self._config.phase_amplitude_threshold,
            )
        return self._cache["phase"]

    def instantaneous_frequency(self) -> np.ndarray:
        """Compute or retrieve cached instantaneous baseband frequency offset in Hz.

        Reuses the cached PhaseResult from `self.phase()` to differentiate unwrapped phase.

        Returns:
            1-D real-valued read-only float array of instantaneous frequency in Hz.
        """
        if "instantaneous_frequency" not in self._cache:
            phase_result = self.phase()
            self._cache["instantaneous_frequency"] = compute_instantaneous_frequency(
                phase_result,
                self.sample_rate_hz,
                allow_gaps=True,
            )
        return self._cache["instantaneous_frequency"]

    def fft(self) -> FFTResult:
        """Compute or retrieve cached Fast Fourier Transform (FFT) of signal_iq.

        Uses parameters configured in `self.config` (nfft, spectral_window, centered_spectrum).

        Returns:
            Immutable FFTResult contract containing frequency axis, spectrum, and metadata.
        """
        if "fft" not in self._cache:
            self._cache["fft"] = compute_fft(
                self.signal_iq,
                self.sample_rate_hz,
                nfft=self._config.nfft,
                window=self._config.spectral_window,
                centered=self._config.centered_spectrum,
            )
        return self._cache["fft"]

    def psd(self) -> PSDResult:
        """Compute or retrieve cached Power Spectral Density (PSD) of signal_iq.

        Depending on `self.config.psd_mode`, either computes periodogram from cached FFT
        or performs Welch segment averaging.

        Returns:
            Immutable PSDResult contract containing frequency coordinates, linear PSD,
            and logarithmic PSD in dBFS/Hz.
        """
        if "psd" not in self._cache:
            if self._config.psd_mode == "welch":
                self._cache["psd"] = compute_welch_psd(
                    self.signal_iq,
                    self.sample_rate_hz,
                    nperseg=self._config.psd_nperseg,
                    noverlap=self._config.psd_noverlap,
                    nfft=self._config.nfft,
                    window=self._config.psd_welch_window,
                    centered=self._config.centered_spectrum,
                    floor_dbfs_per_hz=self._config.psd_floor_dbfs_per_hz,
                )
            else:
                fft_result = self.fft()
                self._cache["psd"] = compute_psd_from_fft(
                    fft_result,
                    floor_dbfs_per_hz=self._config.psd_floor_dbfs_per_hz,
                )
        return self._cache["psd"]

    def autocorrelation(self) -> AutocorrelationResult:
        """Compute or retrieve cached complex autocorrelation for non-negative lags.

        Uses parameters configured in `self.config` (autocorrelation_max_lag_samples,
        autocorrelation_normalized).

        Returns:
            Immutable AutocorrelationResult contract containing lag_samples and values.
        """
        if "autocorrelation" not in self._cache:
            self._cache["autocorrelation"] = compute_autocorrelation(
                self.signal_iq,
                max_lag_samples=self._config.autocorrelation_max_lag_samples,
                normalized=self._config.autocorrelation_normalized,
            )
        return self._cache["autocorrelation"]
