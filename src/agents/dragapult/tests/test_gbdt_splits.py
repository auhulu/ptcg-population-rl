"""Protect fixed holdout games when expanding the Dragapult teacher corpus."""

from __future__ import annotations

import unittest

from agents.dragapult.gbdt.features import SCHEMA_ID
from agents.dragapult.gbdt.scripts.extract import preserve_splits


class PreserveSplitsTest(unittest.TestCase):
    """Validate retention, leakage rejection, and missing-game rejection."""

    def test_expansion_preserves_every_holdout(self) -> None:
        """Keep train, validation, test, and calibration memberships unchanged."""
        splits = {
            name: {"games": [name]} for name in ("train", "validation", "test", "calibration")
        }
        evidence = [
            {"sha256": name} for name in ("new", "test", "train", "calibration", "validation")
        ]
        names, assignments = preserve_splits(evidence, {"schema_id": SCHEMA_ID, "splits": splits})
        self.assertEqual(set(names), set(splits))
        self.assertEqual(assignments, ["train", "test", "train", "calibration", "validation"])

    def test_rejects_overlap(self) -> None:
        """Reject a source manifest that leaks the same game between splits."""
        previous = {
            "schema_id": SCHEMA_ID,
            "splits": {"train": {"games": ["a"]}, "validation": {"games": ["a"]}},
        }
        with self.assertRaisesRegex(ValueError, "overlap"):
            preserve_splits([{"sha256": "a"}], previous)

    def test_rejects_missing_games_and_schema_changes(self) -> None:
        """Do not silently drop previous holdouts or mix feature contracts."""
        previous = {
            "schema_id": SCHEMA_ID,
            "splits": {"train": {"games": ["a"]}, "validation": {"games": ["b"]}},
        }
        with self.assertRaisesRegex(ValueError, "every previous"):
            preserve_splits([{"sha256": "a"}], previous)
        previous["schema_id"] = "other"
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            preserve_splits([{"sha256": "a"}, {"sha256": "b"}], previous)


if __name__ == "__main__":
    unittest.main()
