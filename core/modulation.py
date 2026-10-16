"""Legacy-compatible signal combination. Normalization intentionally unchanged."""
import numpy as np


def combine(operation: str, envelope: np.ndarray | None,
            carrier: np.ndarray | None) -> tuple[np.ndarray, str]:
    match operation.lower():
        case "multiply":
            lhs = 1 if envelope is None else envelope
            rhs = 1 if carrier is None else carrier
            output = lhs * rhs
            output = output / max(output)
            suffix = "_x"
        case "add":
            lhs = 0 if envelope is None else envelope
            rhs = 0 if carrier is None else carrier
            output = lhs + rhs
            output = output / max(output)
            suffix = "_+"
        case "subtract":
            lhs = 0 if envelope is None else envelope
            rhs = 0 if carrier is None else carrier
            output = lhs - rhs
            if np.any(output):
                output = output / max(output)
            suffix = "_-"
        case _:
            raise ValueError(f"Unknown operation: {operation}")
    return output, suffix
