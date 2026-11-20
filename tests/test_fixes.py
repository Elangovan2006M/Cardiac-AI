import unittest
import os

import dsp_pipeline


class DSPTests(unittest.TestCase):
    def test_mel_filterbank_shape(self):
        fb = dsp_pipeline.get_mel_filterbank()
        self.assertEqual(fb.shape, (32, 129))

    def test_spectrogram_shape(self):
        spec = dsp_pipeline.compute_log_mel_spectrogram([0.0] * 4000)
        self.assertEqual(spec.shape, (32, 32))


class AppTests(unittest.TestCase):
    def test_routes_exist(self):
        from app import app
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        self.assertIn('/api/record/export', rules)
        self.assertIn('/api/train', rules)


if __name__ == '__main__':
    unittest.main()
