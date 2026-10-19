"""AWG waveform modulation and amplitude normalization."""

import numpy as np


def normalize(signal: np.ndarray) -> np.ndarray:
    """Normalize a waveform using its maximum absolute amplitude.

    Scale nonzero samples so that the maximum absolute amplitude equals
    1.0. Preserve the original polarity and relative sample amplitudes.

    Args:
        signal: Nonempty array containing waveform samples.

    Returns:
        A new array containing normalized samples. An all-zero signal
        is returned unchanged as a copy.
    """
    peak = np.max(np.abs(signal))

    if peak > 0.0:
        return signal / peak

    return signal.copy()


def combine(
    operation: str,
    envelope: np.ndarray | None,
    carrier: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    """Combine two waveforms and normalize the resulting signal.

    Apply the selected mathematical operation to the envelope and
    carrier samples. Missing operands are replaced with the neutral
    element of the selected operation.

    Args:
        operation: Mathematical operation: "multiply", "add", or
            "subtract". Operation names are case-insensitive.
        envelope: Envelope samples, or None for a neutral operand.
        carrier: Carrier samples, or None for a neutral operand.

    Returns:
        A tuple containing:
            - Normalized waveform samples as a NumPy array.
            - Filename suffix identifying the selected operation.

    Raises:
        ValueError: If the requested operation is not supported.
    """
    match operation.lower():
        case "multiply":
            lhs = 1.0 if envelope is None else envelope
            rhs = 1.0 if carrier is None else carrier
            output = lhs * rhs
            suffix = "_x"

        case "add":
            lhs = 0.0 if envelope is None else envelope
            rhs = 0.0 if carrier is None else carrier
            output = lhs + rhs
            suffix = "_+"

        case "subtract":
            lhs = 0.0 if envelope is None else envelope
            rhs = 0.0 if carrier is None else carrier
            output = lhs - rhs
            suffix = "_-"

        case _:
            raise ValueError(f"Unknown operation: {operation}")

    return normalize(np.asarray(output)), suffix
    