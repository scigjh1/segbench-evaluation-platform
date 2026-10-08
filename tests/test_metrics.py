"""Focused metric and API regression tests for SegBench."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from urllib.request import urlopen

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server import evaluate_pair, sample_evaluation  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_perfect_mask_scores_one(self) -> None:
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        result = evaluate_pair(mask, mask, tolerance=2)
        self.assertEqual(result["dice"], 1.0)
        self.assertEqual(result["iou"], 1.0)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["boundaryF1"], 1.0)
        self.assertEqual(result["hd95"], 0.0)

    def test_disjoint_masks_score_zero_overlap(self) -> None:
        prediction = np.zeros((32, 32), dtype=np.uint8)
        target = np.zeros((32, 32), dtype=np.uint8)
        prediction[2:10, 2:10] = 255
        target[22:30, 22:30] = 255
        result = evaluate_pair(prediction, target, tolerance=1)
        self.assertEqual(result["dice"], 0.0)
        self.assertEqual(result["iou"], 0.0)
        self.assertGreater(result["hd95"], 0.0)

    def test_fixed_suite_candidate_improves_core_metrics(self) -> None:
        result = sample_evaluation(tolerance=2)
        self.assertEqual(result["candidate"]["summary"]["samples"], 24)
        self.assertGreater(result["delta"]["dice"], 0)
        self.assertGreater(result["delta"]["iou"], 0)
        self.assertGreater(result["delta"]["boundaryF1"], 0)
        self.assertGreater(result["delta"]["hd95"], 0)


class LiveApiTests(unittest.TestCase):
    def test_health_and_sample_endpoints(self) -> None:
        base_url = os.environ.get("SEG_BENCH_TEST_URL", "http://127.0.0.1:4174").rstrip("/")
        try:
            with urlopen(f"{base_url}/api/health", timeout=3) as response:
                health = json.load(response)
            with urlopen(
                f"{base_url}/api/sample-evaluation?tolerance=2",
                timeout=10,
            ) as response:
                payload = json.load(response)
        except OSError as exc:
            self.skipTest(f"SegBench server is not running: {exc}")
        self.assertEqual(health["status"], "ok")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["result"]["candidate"]["summary"]["samples"], 24)


if __name__ == "__main__":
    unittest.main()
