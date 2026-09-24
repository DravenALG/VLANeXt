"""
Summarize LIBERO-Plus rollout results.

Usage:
    python -m src.evaluation.libero_plus_bench.results_summary \
        --video_dir /path/to/videos \
        --suite libero_spatial \
        --classification_json third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json \
        --output summary.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_CATEGORY_NAME_MAP = {
    "Background Textures": "Background",
    "Robot Initial States": "Robot",
    "Camera Viewpoints": "Camera",
    "Language Instructions": "Language",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
    "Light Conditions": "Light",
}


def _parse_filename(path: Path) -> Tuple[int, bool, str]:
    """Parse episode id, success flag, and task slug from rollout filename."""
    m = re.match(r"episode=(\d+)--success=(True|False)--task=(.+)\.mp4", path.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {path.name}")
    episode_id = int(m.group(1))
    success = m.group(2) == "True"
    task_slug = m.group(3)
    return episode_id, success, task_slug


def _load_suite_categories(classification_path: Path, suite: str) -> List[str]:
    with open(classification_path, "r") as f:
        data = json.load(f)
    if suite not in data:
        raise ValueError(f"Suite '{suite}' not found in classification file.")
    return [entry.get("category", "Unknown") for entry in data[suite]]


def summarize(video_dir: Path, classification_path: Path, suite: str) -> Dict:
    categories = _load_suite_categories(classification_path, suite)
    per_task = defaultdict(lambda: {"success": 0, "total": 0, "category": "Unknown"})
    per_category = defaultdict(lambda: {"success": 0, "total": 0})
    unknown_category = "Unknown"

    for mp4 in video_dir.glob("*.mp4"):
        try:
            ep_id, success, _ = _parse_filename(mp4)
        except ValueError:
            continue

        idx = ep_id - 1
        category_raw = categories[idx] if 0 <= idx < len(categories) else unknown_category
        category = _CATEGORY_NAME_MAP.get(category_raw, category_raw or unknown_category)

        per_task[ep_id]["total"] += 1
        per_task[ep_id]["category"] = category
        per_task[ep_id]["success"] += int(success)

        per_category[category]["total"] += 1
        per_category[category]["success"] += int(success)

    task_results = {
        ep_id: {
            "category": vals["category"],
            "success": vals["success"],
            "total": vals["total"],
            "success_rate": vals["success"] / vals["total"] if vals["total"] > 0 else 0.0,
        }
        for ep_id, vals in sorted(per_task.items())
    }
    category_results = {
        cat: {
            "success": vals["success"],
            "total": vals["total"],
            "success_rate": vals["success"] / vals["total"] if vals["total"] > 0 else 0.0,
        }
        for cat, vals in sorted(per_category.items())
    }
    overall_success = sum(v["success"] for v in per_task.values())
    overall_total = sum(v["total"] for v in per_task.values())

    return {
        "suite": suite,
        "overall": {
            "success": overall_success,
            "total": overall_total,
            "success_rate": overall_success / overall_total if overall_total > 0 else 0.0,
        },
        "per_task": task_results,
        "per_category": category_results,
    }


def main():
    video_dir = Path("/mnt/draven/checkpoints/VLANeXt/VLANeXt_from_huggingface/libero_spatial_checkpoint_depth2_exec8_diff2_libero_plus_SR16.57")
    suite = "libero_spatial"
    classification_json = Path("third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json")
    output = None

    summary = summarize(video_dir, classification_json, suite)

    print(f"Suite: {summary['suite']}")
    print(f"Overall: {summary['overall']['success']}/{summary['overall']['total']} "
          f"({summary['overall']['success_rate']*100:.1f}%)")

    print("\nPer Category:")
    for cat, vals in summary["per_category"].items():
        print(f"  {cat}: {vals['success']}/{vals['total']} ({vals['success_rate']*100:.1f}%)")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to {output}")


if __name__ == "__main__":
    main()
