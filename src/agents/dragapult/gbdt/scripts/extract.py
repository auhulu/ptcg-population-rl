"""Extract aligned rule labels, audit C++ features, split whole games."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import json

import numpy as np

from agents.dragapult.gbdt.features import NAMES, SCHEMA, SCHEMA_ID, action_queries, check_deck
from agents.dragapult.replay import decisions, inspect_replay, replay_schema, sha256

if TYPE_CHECKING:
    from typing import Any


def preserve_splits(
    evidence: list[dict[str, Any]], previous: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Keep every prior game in its original split and assign only new games to training."""
    if previous["schema_id"] != SCHEMA_ID:
        raise ValueError("Previous split schema mismatch")
    fixed = {}
    split_names = list(previous["splits"])
    if not {"train", "validation"} <= set(split_names) or set(split_names) - {
        "train",
        "validation",
        "test",
        "calibration",
    }:
        raise ValueError("Invalid previous split names")
    for split, details in previous["splits"].items():
        for digest in details["games"]:
            if digest in fixed:
                raise ValueError("Previous game split overlap")
            fixed[digest] = split
    if not set(fixed) <= {entry["sha256"] for entry in evidence}:
        raise ValueError("Current corpus must contain every previous split game")
    assignments = [fixed.get(entry["sha256"], "train") for entry in evidence]
    return split_names, assignments


def extract(
    replay_dir: Path,
    output: Path,
    validation_fraction: float = 0.25,
    seed: int = 42,
    test_fraction: float = 0.0,
    calibration_fraction: float = 0.0,
    split_manifest: Path | None = None,
) -> dict[str, Any]:
    """Split complete games and write aligned candidate features, labels, and manifests."""
    if split_manifest is None and (
        not 0 < validation_fraction < 1
        or not 0 <= test_fraction < 1
        or not 0 <= calibration_fraction < 1
        or validation_fraction + test_fraction + calibration_fraction >= 1
    ):
        raise ValueError("Validation fraction must be between 0 and 1")
    paths = sorted(replay_dir.rglob("game-*.jsonl"))
    evidence = []
    seen = set()
    for path in paths:
        header, terminal = inspect_replay(path)
        if replay_schema(header) != SCHEMA_ID:
            raise ValueError(f"Replay feature/deck schema mismatch: {path}")
        if "dragapult" not in header["agents"]:
            continue
        for seat, agent in enumerate(header["agents"]):
            if agent == "dragapult":
                check_deck(header["decks"][seat])
        digest = sha256(path)
        if digest in seen:
            raise ValueError(f"Duplicate replay: {path}")
        seen.add(digest)
        evidence.append(
            dict(
                path=str(path.relative_to(replay_dir)),
                sha256=digest,
                metadata_sha256=sha256(path.parent / "card_metadata.json"),
                result=terminal["result"],
                seed=header["seed"],
            )
        )
    if len(evidence) < (3 if test_fraction and split_manifest is None else 2):
        raise ValueError("At least two successful teacher games are required")
    if split_manifest is not None:
        previous = json.loads(split_manifest.read_text())
        split_names, assignments = preserve_splits(evidence, previous)
    else:
        indices = np.random.default_rng(seed).permutation(len(evidence))
        n_valid = max(1, min(len(evidence) - 1, round(len(evidence) * validation_fraction)))
        n_test = max(1, round(len(evidence) * test_fraction)) if test_fraction else 0
        n_calibration = (
            max(1, round(len(evidence) * calibration_fraction)) if calibration_fraction else 0
        )
        if n_valid + n_test + n_calibration >= len(evidence):
            raise ValueError("Not enough games for disjoint train/validation/test splits")
        validation = set(indices[:n_valid].tolist())
        test = set(indices[n_valid : n_valid + n_test].tolist())
        calibration = set(indices[n_valid + n_test : n_valid + n_test + n_calibration].tolist())
        assignments = [
            "validation"
            if i in validation
            else "test"
            if i in test
            else "calibration"
            if i in calibration
            else "train"
            for i in range(len(evidence))
        ]
        split_names = ["train", "validation"]
        split_names += ["test"] if test_fraction else []
        split_names += ["calibration"] if calibration_fraction else []
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Feature output directory must be empty")
    report = dict(
        schema_id=SCHEMA_ID,
        feature_names=NAMES,
        deck_counts=SCHEMA["deck_counts"],
        split_seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        calibration_fraction=calibration_fraction,
        games=evidence,
        splits={},
        feature_max_abs_error=0.0,
        teacher="dragapult",
        selection=SCHEMA["selection"],
    )
    if split_manifest is not None:
        report.update(
            split_seed=None,
            validation_fraction=None,
            test_fraction=None,
            calibration_fraction=None,
            previous_split_manifest_sha256=sha256(split_manifest),
            new_game_split="train",
        )
    for split in split_names:
        matrices, labels, groups, meta, split_games = [], [], [], [], []
        forced = 0
        for file_index, entry in enumerate(evidence):
            if assignments[file_index] != split:
                continue
            split_games.append(entry["sha256"])
            path = replay_dir / entry["path"]
            table = {
                c["cardId"]: c for c in json.loads((path.parent / "card_metadata.json").read_text())
            }
            for record in decisions(path):
                if record["agent"] != "dragapult":
                    continue
                queries = list(action_queries(record["observation"], record["action"], table))
                native = record.get("gbdt_queries")
                if native is None or len(native) != len(queries):
                    raise ValueError(f"Missing native feature evidence: {path}")
                for query, cpp in zip(queries, native, strict=True):
                    for key in ("prefix", "options", "target"):
                        if query[key] != cpp[key]:
                            raise ValueError(
                                f"Selection contract mismatch: {path} step={record['step']}"
                            )
                    x = np.asarray(query["features"], dtype=np.float64)
                    cx = np.asarray(cpp["features"], dtype=np.float64)
                    if x.shape != cx.shape or not np.isfinite(x).all() or not np.array_equal(x, cx):
                        raise ValueError(
                            f"Python/C++ feature mismatch: {path} step={record['step']}"
                        )
                    if len(query["options"]) == 1:
                        forced += 1
                        continue
                    matrices.append(x)
                    labels.extend(int(o == query["target"]) for o in query["options"])
                    groups.append(len(query["options"]))
                    meta.append(
                        dict(
                            game=entry["sha256"],
                            step=record["step"],
                            seat=record["seat"],
                            context=record["observation"]["select"]["context"],
                            prefix=query["prefix"],
                            options=query["options"],
                            target=query["target"],
                        )
                    )
            if len(split_games) % 25 == 0:
                print(f"{split}: extracted {len(split_games)} games", flush=True)
        if not groups:
            raise ValueError(f"No trainable ranking groups in {split}")
        np.savez_compressed(
            output / f"{split}.npz",
            x=np.concatenate(matrices),
            y=np.asarray(labels, dtype=np.int32),
            groups=np.asarray(groups, dtype=np.int32),
        )
        (output / f"{split}.groups.json").write_text(json.dumps(meta, separators=(",", ":")) + "\n")
        report["splits"][split] = dict(
            games=split_games,
            groups=len(groups),
            rows=len(labels),
            forced_groups=forced,
            arrays_sha256=sha256(output / f"{split}.npz"),
            groups_sha256=sha256(output / f"{split}.groups.json"),
        )
    all_games = [game for split in report["splits"].values() for game in split["games"]]
    if len(all_games) != len(set(all_games)):
        raise ValueError("Game split overlap")
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: {n: v for n, v in s.items() if n != "games"} for k, s in report["splits"].items()}
        ),
        flush=True,
    )
    return report


def main() -> None:
    """Parse command-line arguments and run this tool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-dir", type=Path, default=ROOT / "outputs/dragapult/rule_replays")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/dragapult/gbdt_features")
    p.add_argument("--validation-fraction", type=float, default=0.25)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--calibration-fraction", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split-manifest",
        type=Path,
        help="Preserve all splits from an earlier feature manifest; put new games in train. Overrides split fractions and seed.",
    )
    a = p.parse_args()
    extract(
        a.replay_dir,
        a.output_dir,
        a.validation_fraction,
        a.seed,
        a.test_fraction,
        a.calibration_fraction,
        a.split_manifest,
    )


if __name__ == "__main__":
    main()
