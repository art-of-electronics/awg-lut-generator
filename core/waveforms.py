"""AWG waveform generation and sample count definitions.

Generate periodic waveforms using NumPy without dependencies on the GUI.
Preserve the established sampling conventions for CSV compatibility.
"""

import numpy as np


SAMPLE_COUNTS = {
    "1k": 1024,
    "2k": 2048,
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
    "64k": 65536,
}

WAVEFORM_NAMES = (
    "None",
    "Sine",
    "Square",
    "Gaussian",
    "Exponential",
    "Wave",
)


def generate_waveform(
    name: str,
    samples: int,
    replicate: int = 1,
    parameter: float = 0.1,
) -> tuple[np.ndarray | None, str, str]:
    """Generate waveform samples and their filename suffixes.

    Generate a single waveform period and repeat it to fill the
    requested sample count. Gaussian and exponential waveforms are
    normalized to the [0.0, 1.0] range.

    Args:
        name: Waveform type. Supported values are "None", "Sine",
            "Square", "Gaussian", "Exponential", and "Wave".
            Names are case-insensitive.
        samples: Total number of output samples.
        replicate: Number of waveform repetitions.
        parameter: Shape parameter used by square, Gaussian,
            and exponential waveforms.

    Returns:
        A tuple containing:
            - Generated waveform samples, or None if no waveform
              is selected.
            - Filename suffix identifying the repetition count.
            - Filename suffix identifying the shape parameter,
              or an empty string when not applicable.

    Raises:
        ValueError: If the waveform type is unsupported, sample
            count or repetition count is invalid, or the sample
            count is not divisible by the repetition count.
    """
    waveform = name.lower()

    if waveform == "none":
        return None, "", ""

    if waveform not in {"sine", "square", "gaussian", "exponential", "wave"}:
        raise ValueError(f"Unknown waveform: {name}")

    if samples <= 0 or replicate <= 0 or samples % replicate != 0:
        raise ValueError(
            "Sample count must be positive and divisible by replicate"
        )

    count = samples // replicate
    xpos = np.linspace(0, 1, count)

    suffix = f"_{replicate}"
    param_suffix = f"-{parameter}".replace(".", "")

    match waveform:
        case "square":
            period = np.where(xpos < parameter, -1.0, 1.0)

        case "sine":
            indices = np.arange(count, dtype=np.float64)
            period = np.sin(indices * 2 * np.pi / count)

        case "wave":
            period = np.zeros(count)

            for harmonic in np.arange(1, 15, 2):
                period += (
                    (4 / (np.pi * harmonic))
                    * np.cos(2 * np.pi * harmonic * xpos)
                )

            period = period / max(period)

        case "gaussian":
            period = np.exp(-((xpos - 0.5) / parameter) ** 2)

        case "exponential":
            period = np.exp(-xpos / parameter)

    if waveform in {"gaussian", "exponential"}:
        minimum = np.min(period)
        maximum = np.max(period)
        amplitude = maximum - minimum

        if amplitude > 0.0:
            period = (period - minimum) / amplitude
        else:
            period = np.zeros_like(period)

    if waveform in {"sine", "wave"}:
        param_suffix = ""

    return np.tile(period, replicate), suffix, param_suffix
