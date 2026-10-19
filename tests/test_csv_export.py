import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from core.waveforms import generate_waveform
from core.modulation import combine
from services.csv_export import export_lut

ROOT = Path(__file__).resolve().parents[1]


class TestCSVExport(unittest.TestCase):
    def test_raw_export(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal = np.array([-1., 0., 0.125, 1.])
            result = export_lut(signal, 'test.csv', 'RAW', root, root)
            self.assertEqual(result, root / 'RAW' / 'test.csv')
            with result.open(newline='', encoding='utf-8') as source:
                rows = list(csv.reader(source))
            self.assertEqual(rows, [['1', '-1.000000'], ['2', '0.000000'],
                                    ['3', '0.125000'], ['4', '1.000000']])

    def test_sag1021_header_and_samples(self):
        header_file = ROOT / 'headers' / 'Header_SAG1021.csv'
        if not header_file.is_file():
            self.skipTest('SAG1021 header file is not available')
        envelope, _, _ = generate_waveform('Gaussian', 16384, 2, 0.1)
        carrier, _, _ = generate_waveform('Sine', 16384, 8, 0.1)
        signal, _ = combine('Multiply', envelope, carrier)
        with TemporaryDirectory() as tmp:
            result = export_lut(signal, 'reference.csv', 'SAG1021', Path(tmp), ROOT / 'headers')
            with result.open(newline='', encoding='utf-8') as source:
                rows = list(csv.reader(source))
            self.assertEqual(rows[0], ['data length,16384'])
            self.assertEqual(rows[12], ['xpos,value'])
            self.assertEqual(len(rows), 13 + 16384)
            self.assertEqual([int(row[0]) for row in rows[13:]], list(range(1, 16385)))
            values = np.array([float(row[1]) for row in rows[13:]])
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertLessEqual(float(np.max(np.abs(values))), 1.0 + 1e-6)
