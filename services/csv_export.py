"""Legacy-compatible AWG CSV export."""
from pathlib import Path
import numpy as np
import pandas as pd


def discover_headers(directory: Path) -> list[str]:
    return ["RAW"] + [file.stem.removeprefix("Header_") for file in directory.iterdir()
                      if file.is_file() and file.suffix == ".csv" and file.stem.startswith("Header_")]


def export_lut(signal: np.ndarray, filename: str, header: str,
               output_dir: Path, headers_dir: Path) -> Path:
    destination = output_dir / header
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / filename
    if header != "RAW":
        source = pd.read_csv(headers_dir / f"Header_{header}.csv", skip_blank_lines=False, header=None)
        lines = source[0] + "," + source[1]
        lines.to_csv(output, mode="w", sep=",", encoding="utf-8", index=False, header=None, na_rep=None)
    data = pd.DataFrame(data=signal)
    data.index += 1
    data.to_csv(output, mode="a+", sep=",", encoding="utf-8", header=False, float_format="%.6f")
    return output
