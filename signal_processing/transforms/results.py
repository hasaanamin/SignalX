"""Result contracts for transform operations."""

import math
from dataclasses import dataclass

import numpy as np

from signal_processing.transforms.exceptions import TransformError


@dataclass(frozen=True)
class FFTResult:
    """Complex Fast Fourier Transform (FFT) result container.

    Attributes:
        frequency_offset_hz: Frequency bin coordinates in Hz relative to tuned center.
            Spans [-Fs/2, +Fs/2) when centered=True.
        spectrum: Complex-valued Fourier coefficients corresponding to frequency bins.
        sample_rate_hz: Sampling frequency in Hertz (finite, strictly > 0).
        nfft: Total FFT transform size (>= input_samples).
        input_samples: Number of actual input samples before zero-padding (>= 1).
        window_name: Name of the applied window function (e.g. 'rectangular', 'hann').
        window_energy: Sum of squared window coefficients: sum(window^2).
        centered: Whether zero-frequency is centered at DC (fftshift applied).
    """

    frequency_offset_hz: np.ndarray
    spectrum: np.ndarray
    sample_rate_hz: float
    nfft: int
    input_samples: int
    window_name: str
    window_energy: float
    centered: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.sample_rate_hz, bool) or not isinstance(
            self.sample_rate_hz, (int, float, np.floating, np.integer)
        ):
            raise TransformError("sample_rate_hz must be a numeric float")
        sr = float(self.sample_rate_hz)
        if not math.isfinite(sr) or sr <= 0.0:
            raise TransformError(
                f"sample_rate_hz must be finite and strictly positive, got {self.sample_rate_hz}"
            )
        object.__setattr__(self, "sample_rate_hz", sr)

        for name, val in [("nfft", self.nfft), ("input_samples", self.input_samples)]:
            if isinstance(val, bool) or not isinstance(val, (int, np.integer)) or val <= 0:
                raise TransformError(f"{name} must be an integer > 0, got {val!r}")

        if self.nfft < self.input_samples:
            raise TransformError(
                f"nfft ({self.nfft}) must be >= input_samples ({self.input_samples})"
            )

        if not isinstance(self.window_name, str) or not self.window_name.strip():
            raise TransformError("window_name must be a non-blank string")

        if isinstance(self.window_energy, bool) or not isinstance(
            self.window_energy, (int, float, np.floating, np.integer)
        ):
            raise TransformError("window_energy must be a numeric float")
        we = float(self.window_energy)
        if not math.isfinite(we) or we <= 0.0:
            raise TransformError(
                f"window_energy must be finite and strictly positive, got {self.window_energy}"
            )
        object.__setattr__(self, "window_energy", we)

        if not isinstance(self.centered, bool):
            raise TransformError("centered must be a boolean")

        # Validate frequency array
        try:
            freq_arr = np.asarray(self.frequency_offset_hz)
        except Exception as exc:
            raise TransformError(f"frequency_offset_hz cannot be converted to array: {exc}") from exc

        if freq_arr.ndim != 1:
            raise TransformError(
                f"frequency_offset_hz must be 1-dimensional, got {freq_arr.ndim} dimensions"
            )
        if not (
            np.issubdtype(freq_arr.dtype, np.floating) or np.issubdtype(freq_arr.dtype, np.integer)
        ) or np.iscomplexobj(freq_arr):
            raise TransformError(
                f"frequency_offset_hz must be a real-valued numeric array, got dtype {freq_arr.dtype}"
            )
        if not np.all(np.isfinite(freq_arr)):
            raise TransformError("frequency_offset_hz contains non-finite values")

        # Validate spectrum array
        try:
            spec_arr = np.asarray(self.spectrum)
        except Exception as exc:
            raise TransformError(f"spectrum cannot be converted to array: {exc}") from exc

        if spec_arr.ndim != 1:
            raise TransformError(
                f"spectrum must be 1-dimensional, got {spec_arr.ndim} dimensions"
            )
        if not (
            np.issubdtype(spec_arr.dtype, np.floating)
            or np.issubdtype(spec_arr.dtype, np.complexfloating)
            or np.issubdtype(spec_arr.dtype, np.integer)
        ):
            raise TransformError(
                f"spectrum must be a numeric array, got dtype {spec_arr.dtype}"
            )
        if not np.all(np.isfinite(spec_arr)):
            raise TransformError("spectrum contains non-finite values")

        if len(freq_arr) != len(spec_arr):
            raise TransformError(
                f"frequency_offset_hz length ({len(freq_arr)}) does not match spectrum length ({len(spec_arr)})"
            )

        if len(freq_arr) != self.nfft:
            raise TransformError(
                f"Array length ({len(freq_arr)}) does not match nfft ({self.nfft})"
            )

        c_f = np.ascontiguousarray(freq_arr)
        f_frozen = np.frombuffer(c_f.tobytes(), dtype=c_f.dtype).reshape(c_f.shape)
        object.__setattr__(self, "frequency_offset_hz", f_frozen)

        c_s = np.ascontiguousarray(spec_arr)
        s_frozen = np.frombuffer(c_s.tobytes(), dtype=c_s.dtype).reshape(c_s.shape)
        object.__setattr__(self, "spectrum", s_frozen)
