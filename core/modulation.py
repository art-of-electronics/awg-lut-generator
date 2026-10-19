"""Combine AWG waveforms and scale their peak magnitude to 1."""
import numpy as np


def normalize(signal: np.ndarray) -> np.ndarray:
    """Preserve signal polarity and shape; scale only nonzero signals."""
    peak = np.max(np.abs(signal))
    if peak > 0.0:
        return signal / peak
    return signal.copy()


def combine(operation: str, envelope: np.ndarray | None,
            carrier: np.ndarray | None) -> tuple[np.ndarray, str]:
    match operation.lower():
        case "multiply":
            lhs = 1 if envelope is None else envelope
            rhs = 1 if carrier is None else carrier
            output = lhs * rhs
            suffix = "_x"
        case "add":
            lhs = 0 if envelope is None else envelope
            rhs = 0 if carrier is None else carrier
            output = lhs + rhs
            suffix = "_+"
        case "subtract":
            lhs = 0 if envelope is None else envelope
            rhs = 0 if carrier is None else carrier
            output = lhs - rhs
            suffix = "_-"
        case _:
            raise ValueError(f"Unknown operation: {operation}")
    return normalize(np.asarray(output)), suffix
