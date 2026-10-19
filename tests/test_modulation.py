import unittest
import numpy as np
from core.modulation import combine, normalize
from core.waveforms import generate_waveform


class TestModulation(unittest.TestCase):
    def test_zero_and_negative_normalization(self):
        np.testing.assert_array_equal(normalize(np.zeros(4)), np.zeros(4))
        np.testing.assert_allclose(normalize(np.array([-4., -2., -1.])), [-1., -0.5, -0.25])
        np.testing.assert_allclose(normalize(np.array([-4., -2., 1.])), [-1., -0.5, 0.25])

    def test_identical_subtraction(self):
        for name in ('Sine', 'Square', 'Gaussian', 'Exponential', 'Wave'):
            with self.subTest(name=name):
                signal, _, _ = generate_waveform(name, 1024, 4, 0.2)
                result, _ = combine('Subtract', signal, signal)
                np.testing.assert_array_equal(result, np.zeros(1024))

    def test_operations_preserve_sign_and_inputs(self):
        left = np.array([-0.8, -0.2, 0.4, 0.9])
        right = np.array([0.1, -0.3, 0.7, -0.5])
        for operation, raw in (('Multiply', left * right),
                               ('Add', left + right), ('Subtract', left - right)):
            with self.subTest(operation=operation):
                before_left, before_right = left.copy(), right.copy()
                result, _ = combine(operation, left, right)
                np.testing.assert_allclose(result, raw / np.max(np.abs(raw)))
                np.testing.assert_array_equal(left, before_left)
                np.testing.assert_array_equal(right, before_right)
                self.assertTrue(np.all(np.isfinite(result)))
                self.assertLessEqual(float(np.max(np.abs(result))), 1.0 + 1e-12)

    def test_none_identity(self):
        signal = np.array([0., 0.5, 1.])
        for operation in ('Multiply', 'Add'):
            for left, right in ((None, signal), (signal, None)):
                with self.subTest(operation=operation, left_none=left is None):
                    result, _ = combine(operation, left, right)
                    np.testing.assert_array_equal(result, signal)
        result, _ = combine('Subtract', signal, None)
        np.testing.assert_array_equal(result, signal)
        result, _ = combine('Subtract', None, signal)
        np.testing.assert_array_equal(result, -signal)

    def test_invalid_operation(self):
        with self.assertRaises(ValueError):
            combine('Divide', np.ones(4), np.ones(4))
