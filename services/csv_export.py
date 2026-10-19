"""AWG waveform CSV export and device header management."""

import csv
from pathlib import Path

import numpy as np


def discover_headers(directory: Path) -> list[str]:
    """Discover available device header templates.

    Args:
        directory: Directory containing device header CSV files.

    Returns:
        A list of available export formats, starting with "RAW".
        Device formats are derived from Header_*.csv filenames.
    """
    return ["RAW"] + [
        file.stem.removeprefix("Header_")
        for file in directory.iterdir()
        if file.is_file()
        and file.suffix == ".csv"
        and file.stem.startswith("Header_")
    ]


def read_header_sample_count(
    header: str,
    headers_dir: Path,
) -> int | None:
    """Read the required sample count from a device header.

    Search for a "data length" declaration in the selected header.
    RAW exports and headers without this declaration allow manual
    sample count selection.

    Args:
        header: Export format name, or "RAW".
        headers_dir: Directory containing device header templates.

    Returns:
        Required sample count, or None if no restriction is declared.

    Raises:
        ValueError: If a data length declaration is malformed,
            nonpositive, or duplicated.
        OSError: If the header file cannot be read.
    """
    if header == "RAW":
        return None

    path = headers_dir / f"Header_{header}.csv"
    count = None

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.reader(source):
            if not row or row[0].strip().casefold() != "data length":
                continue

            if len(row) != 2:
                raise ValueError(f"Invalid data length in {path.name}")

            value = row[1].strip()

            if not value.isdecimal() or int(value) <= 0:
                raise ValueError(
                    f"Invalid data length in {path.name}: {value!r}"
                )

            if count is not None:
                raise ValueError(f"Duplicate data length in {path.name}")

            count = int(value)

    return count


def export_lut(
    signal: np.ndarray,
    filename: str,
    header: str,
    output_dir: Path,
    headers_dir: Path,
) -> Path:
    """Export waveform samples to a CSV lookup table.

    Create the destination directory if necessary and write samples
    using one-based indices and six decimal places.

    Device-specific exports prepend the selected header template.
    Header rows preserve the established CSV serialization format.

    Args:
        signal: Waveform samples to export.
        filename: Name of the output CSV file.
        header: Export format name, or "RAW".
        output_dir: Root directory for generated files.
        headers_dir: Directory containing device header templates.

    Returns:
        Path to the generated CSV file.

    Raises:
        OSError: If an input or output file cannot be accessed.
    """
    destination = output_dir / header
    destination.mkdir(parents=True, exist_ok=True)

    output = destination / filename

    with output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")

        if header != "RAW":
            header_path = headers_dir / f"Header_{header}.csv"

            with header_path.open(
                "r", encoding="utf-8", newline=""
            ) as source:
                reader = csv.reader(source)

                for row in reader:
                    # Preserve the established header serialization:
                    # each header row is written as a single CSV field.
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
    