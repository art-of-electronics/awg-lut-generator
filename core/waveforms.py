"""Legacy-compatible waveform generation, independent of Tkinter.

Algorithms intentionally preserve the original sampling conventions.
"""
import numpy as np

SAMPLE_COUNTS = {"1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192,
                 "16k": 16384, "32k": 32768, "64k": 65536}
WAVEFORM_NAMES = ("None", "Sine", "Square", "Gaussian", "Exponential", "Wave")


def generate_waveform(name: str, samples: int, replicate: int = 1,
                      parameter: float = 0.1) -> tuple[np.ndarray | None, str, str]:
    """Return (samples, repetition_suffix, parameter_suffix) as in legacy code."""
    if name.lower() == "none":
        return None, "", ""
    if name.lower() not in {"sine", "square", "gaussian", "exponential", "wave"}:
        raise ValueError(f"Unknown waveform: {name}")
    if samples <= 0 or replicate <= 0 or samples % replicate != 0:
        raise ValueError("Sample count must be positive and divisible by replicate")
    count = samples // replicate
    xpos = np.linspace(0, 1, count)
    suffix = f"_{replicate}"
    param_suffix = f"-{parameter}".replace(".", "")
    match name.lower():
        case "square":
            period = np.where(xpos < parameter, -1.0, 1.0)
        case "sine":
            indices = np.arange(count, dtype=np.float64)
            period = np.sin(indices * 2 * np.pi / count)
        case "wave":
            period = np.zeros(count)
            for harmonic in np.arange(1, 15, 2):
                period += (4 / (np.pi * harmonic)) * np.cos(2 * np.pi * harmonic * xpos)
            period = period / max(period)
        case "gaussian":
            period = np.exp(-((xpos - 0.5) / parameter) ** 2)
        case "exponential":
            period = np.exp(-xpos / parameter)
    if name.lower() in {"gaussian", "exponential"}:
        minimum = np.min(period)
        maximum = np.max(period)
        amplitude = maximum - minimum
        if amplitude > 0.0:
            period = (period - minimum) / amplitude
        else:
            period = np.zeros_like(period)
    if name.lower() in {"sine", "wave"}:
        param_suffix = ""
    return np.tile(period, replicate), suffix, param_suffix
