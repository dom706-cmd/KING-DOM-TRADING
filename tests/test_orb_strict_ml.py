from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ml import orb_model_service as oms


class StrictOrbMLRoutingTests(unittest.TestCase):
    def test_bucket_resolution_uses_ab_split(self) -> None:
        self.assertEqual(oms.bucket_name_for_price(29.99), "under30")
        self.assertEqual(oms.bucket_name_for_price(30.00), "liquid")
        self.assertTrue(str(oms.bucket_model_path_for_price(29.99)).endswith("models/model_b_outlier.pkl"))
        self.assertTrue(str(oms.bucket_model_path_for_price(30.00)).endswith("models/model_a_liquid.pkl"))

    def test_missing_strict_bucket_raises_instead_of_falling_back(self) -> None:
        missing = Path("/tmp/definitely_missing_strict_orb_model.pkl")
        cand = SimpleNamespace(symbol="ABCD", last_price=12.34)
        with patch.dict(os.environ, {"ORB_STRICT_ML": "1", "ORB_MODEL_B_PATH": str(missing)}, clear=False):
            with self.assertRaises(oms.OrbModelMissingError):
                oms.score_orb_candidates([cand], provider=object())

    def test_scanner_routes_under30_and_ge30_to_separate_models(self) -> None:
        seen = []

        class DummyRanker:
            def __init__(self, cfg, provider=None):
                self.cfg = cfg
                self.provider = provider
                self.failures = []

            def load(self):
                seen.append(("load", Path(self.cfg.model_path).name))
                return True

            def score_candidates(self, symbols):
                seen.append(("score", Path(self.cfg.model_path).name, tuple(symbols)))
                return {sym: 0.77 for sym in symbols}

        under = SimpleNamespace(symbol="ABCD", last_price=12.34)
        liquid = SimpleNamespace(symbol="BIGX", last_price=45.67)

        with patch.object(oms, "ORBRanker", DummyRanker):
            out = oms.score_orb_candidates([under, liquid], provider=object())

        self.assertEqual(out["bucket_by_symbol"]["ABCD"], "under30")
        self.assertEqual(out["bucket_by_symbol"]["BIGX"], "liquid")
        self.assertIn(("score", "model_b_outlier.pkl", ("ABCD",)), seen)
        self.assertIn(("score", "model_a_liquid.pkl", ("BIGX",)), seen)
        self.assertNotIn(("score", "orb_ranker.pkl", ("ABCD",)), seen)
        self.assertNotIn(("score", "orb_ranker_b_under20.pkl", ("ABCD",)), seen)


if __name__ == "__main__":
    unittest.main(verbosity=2)
