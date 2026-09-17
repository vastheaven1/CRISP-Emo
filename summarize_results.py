"""Aggregate holdout evaluations as mean +/- sample SD."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METRICS = ("accuracy", "macro_f1", "auroc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", type=Path, nargs="+")
    return parser.parse_args()


def _scores(payload: dict) -> dict[str, float]:
    primary = (
        payload.get("primary_test")
        or payload.get("primary_development_test")
        or payload.get("primary_development_holdout")
        or {}
    )
    report = (
        payload.get("test")
        or payload.get("development_test")
        or payload.get("development_holdout")
    )
    if report is None:
        raise ValueError("Evaluation JSON has no test report")
    return {
        "accuracy": float(primary["accuracy"]),
        "macro_f1": float(
            primary.get("macro_f1", report["selection_scores"]["macro_f1"])
        ),
        "auroc": float(primary.get("auroc", report["selection_scores"]["auroc"])),
    }


def aggregate(paths: list[Path]) -> dict:
    seeds = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seeds.append({"seed": int(payload["seed"]), **_scores(payload)})
    seeds.sort(key=lambda item: item["seed"])
    output = {"seeds": seeds, "aggregate": {}, "paper_percent": {}}
    for metric in METRICS:
        values = [item[metric] for item in seeds]
        mean = statistics.fmean(values)
        sample_sd = statistics.stdev(values)
        output["aggregate"][metric] = {"mean": mean, "sample_sd": sample_sd}
        output["paper_percent"][metric] = (
            f"{100.0 * mean:.2f} +/- {100.0 * sample_sd:.2f}"
        )
    return output


def main() -> None:
    args = parse_args()
    if len(args.evaluations) < 2:
        raise ValueError("At least two seeds are required for a sample SD")
    result = aggregate(args.evaluations)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
