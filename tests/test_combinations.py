import unittest
import numpy as np
from core.waveforms import generate_waveform
from core.modulation import combine


class TestCombinations(unittest.TestCase):
    def test_all_waveform_pairs_and_operations(self):
        names = ('Sine', 'Square', 'Gaussian', 'Exponential', 'Wave')
        for samples, repetitions in ((1024, 4), (16384, 8)):
            signals = {name: generate_waveform(name, samples, repetitions, 0.2)[0]
                       for name in names}
            for envelope_name in names:
                for carrier_name in names:
                    for operation in ('Multiply', 'Add', 'Subtract'):
                        with self.subTest(samples=samples, envelope=envelope_name,
                                          carrier=carrier_name, operation=operation):
                            result, _ = combine(operation, signals[envelope_name],
                                                signals[carrier_name])
                            self.assertEqual(result.shape, (samples,))
                            self.assertTrue(np.all(np.isfinite(result)))
                            self.assertLessEqual(float(np.max(np.abs(result))), 1.0 + 1e-12)

    def test_low_samples_per_period(self):
        envelope, _, _ = generate_waveform('Gaussian', 1024, 4, 0.1)
        carrier, _, _ = generate_waveform('Sine', 1024, 128, 0.1)
        result, _ = combine('Multiply', envelope, carrier)
        self.assertEqual(len(result), 1024)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertAlmostEqual(float(np.max(np.abs(result))), 1.0, places=12)

    def test_golden_configuration_contract(self):
        envelope, _, _ = generate_waveform('Gaussian', 16384, 2, 0.1)
        carrier, _, _ = generate_waveform('Sine', 16384, 8, 0.1)
        result, suffix = combine('Multiply', envelope, carrier)
        self.assertEqual(suffix, '_x')
        self.assertEqual(len(result), 16384)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertAlmostEqual(float(np.max(np.abs(result))), 1.0, places=12)
