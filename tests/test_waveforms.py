import unittest
import numpy as np
from core.waveforms import SAMPLE_COUNTS, WAVEFORM_NAMES, generate_waveform


class TestWaveforms(unittest.TestCase):
    def test_all_sizes_and_types(self):
        for samples in SAMPLE_COUNTS.values():
            for name in WAVEFORM_NAMES[1:]:
                with self.subTest(samples=samples, name=name):
                    signal, _, _ = generate_waveform(name, samples, 2, 0.1)
                    self.assertEqual(signal.shape, (samples,))
                    self.assertTrue(np.all(np.isfinite(signal)))
                    self.assertLessEqual(float(np.max(np.abs(signal))), 1.0 + 1e-12)

    def test_unipolar_full_range(self):
        for name in ('Gaussian', 'Exponential'):
            for parameter in (0.1, 0.2, 0.3, 0.9):
                for repetitions in (1, 2, 4):
                    with self.subTest(name=name, parameter=parameter, repetitions=repetitions):
                        signal, _, _ = generate_waveform(name, 1024, repetitions, parameter)
                        self.assertAlmostEqual(float(np.min(signal)), 0.0, places=12)
                        self.assertAlmostEqual(float(np.max(signal)), 1.0, places=12)

    def test_period_replication(self):
        for name in ('Sine', 'Square', 'Gaussian', 'Exponential', 'Wave'):
            with self.subTest(name=name):
                signal, _, _ = generate_waveform(name, 1024, 4, 0.2)
                for index in range(1, 4):
                    np.testing.assert_array_equal(signal[:256], signal[index * 256:(index + 1) * 256])

    def test_none_and_invalid_arguments(self):
        self.assertIsNone(generate_waveform('None', 1024)[0])
        for args in (('Unknown', 1024, 1, 0.1), ('Sine', 0, 1, 0.1),
                     ('Sine', 1024, 0, 0.1), ('Sine', 1024, 3, 0.1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                generate_waveform(*args)
