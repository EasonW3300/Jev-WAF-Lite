"""Calibration and threshold reporting stay separate from model training."""

import math
import unittest

from security_jev_v2.policy import calibrate, choose_policy, report_predictions


class V2PolicyTests(unittest.TestCase):
    def test_calibration_is_finite_and_choice_stays_normalized(self):
        block_logits = [-3., -2., 2., 3.]
        type_logits = [[3., 0., 0., 0.], [0., 3., 0., 0.],
                       [0., 0., 3., 0.], [0., 0., 0., 3.]]
        policy = calibrate(block_logits, [0, 0, 1, 1], type_logits, [0, 1, 2, 3])
        self.assertTrue(math.isfinite(policy["block_temperature"]))
        self.assertGreater(policy["type_temperature"], 0)
        self.assertTrue(math.isfinite(policy["block_bias"]))

    def test_low_false_positive_policy_does_not_use_test_labels(self):
        calibration = choose_policy([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], max_fpr=.01)
        self.assertLessEqual(calibration["fpr"], .01)
        self.assertGreaterEqual(calibration["threshold"], .8)
        report = report_predictions([.1, .2, .8, .9], [0, 0, 1, 1],
                                    [[.7, .1, .1, .1], [.1, .7, .1, .1]],
                                    [0, 1], ["site-a", "site-b", "site-c", "site-d"],
                                    calibration["threshold"])
        self.assertEqual(report["block"]["confusion"]["fp"], 0)
        self.assertEqual(report["choice"]["known_support"], 2)
        self.assertIn("fpr_wilson_95", report["block"])


if __name__ == "__main__":
    unittest.main()
