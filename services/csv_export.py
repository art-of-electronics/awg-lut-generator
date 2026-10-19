"""AWG CSV export using the Python standard library."""

import csv
from pathlib import Path

import numpy as np


def discover_headers(directory: Path) -> list[str]:
    return ["RAW"] + [
        file.stem.removeprefix("Header_")
        for file in directory.iterdir()
        if file.is_file() and file.suffix == ".csv" and file.stem.startswith("Header_")
    ]


def export_lut(signal: np.ndarray, filename: str, header: str,
               output_dir: Path, headers_dir: Path) -> Path:
    destination = output_dir / header
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / filename

    with output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")

        if header != "RAW":
            header_path = headers_dir / f"Header_{header}.csv"
            with header_path.open("r", encoding="utf-8", newline="") as source:
                reader = csv.reader(source)
                for row in reader:
                    # The legacy Pandas exporter wrote each header row as
                    # one quoted CSV field (including empty header rows).
                    if len(row) >= 2:
                        value = f"{row[0]},{row[1]}"
                    elif len(row) == 1:
                        value = row[0]
                    else:
                        value = ""
                    target.write('"' + value.replace('"', '""') + '"\n')

        for index, value in enumerate(signal, start=1):
            writer.writerow((index, f"{float(value):.6f}"))

    return output
