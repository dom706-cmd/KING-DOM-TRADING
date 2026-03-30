from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from runtime.state_store import RuntimeStateStore


class RuntimeStateStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "runtime_state.db"
        self.store = RuntimeStateStore(self.db_path)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_saved_preset_metadata_round_trip_and_touch(self) -> None:
        self.store.save_scan_preset(
            "Open Drive",
            {"strategy": "orb", "min_rvol": "2.0"},
            favorite=True,
            notes="Use for strong open-drive tape.",
            category="momentum",
        )
        presets = self.store.recent_scan_presets(limit=10)
        self.assertEqual(len(presets), 1)
        row = presets[0]
        self.assertTrue(row["favorite"])
        self.assertEqual(row["notes"], "Use for strong open-drive tape.")
        self.assertEqual(row["category"], "momentum")
        self.assertIsNone(row["last_used_at"])

        touched = self.store.touch_scan_preset("Open Drive", last_used_at=1234.5)
        self.assertTrue(touched)
        updated = self.store.recent_scan_presets(limit=10)[0]
        self.assertEqual(updated["last_used_at"], 1234.5)

    def test_recent_news_preserves_source_url(self) -> None:
        self.store.append_news_event(
            {
                "news_id": "abc123",
                "symbol": "ABCD",
                "headline": "ABCD announces partnership",
                "source": "Newswire",
                "url": "https://example.com/story",
                "published_at": "2026-03-30T14:00:00+00:00",
                "received_at": 1234.0,
                "sentiment_score": 0.7,
                "catalyst_score": 0.8,
                "freshness_sec": 1200.0,
                "tags": ["deal"],
            }
        )
        items = self.store.recent_news(symbol="ABCD", limit=5)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["url"], "https://example.com/story")


if __name__ == "__main__":
    unittest.main(verbosity=2)
