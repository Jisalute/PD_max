"""Evaluation gates for candidate image-detection models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from app.ai_detection.core.dataset_policy import is_v3_baseline_sample, is_v3_candidate_gate_sample


METRIC_NAMES = ("balanced_accuracy", "normal_recall", "tampered_recall")
MINIMUM_ROI_COVERAGE = 0.90
MAX_ALLOWED_TEST_REGRESSION = 0.02
MINIMUM_TEST_CLASS_SAMPLES = 20


def evaluate_regression_gate(predictions: Iterable[tuple[str, int, str]]) -> Dict[str, Any]:
    failures = []
    total = 0
    for path, expected_label, actual in predictions:
        total += 1
        expected = "正常" if int(expected_label) == 0 else "篡改"
        if actual != expected:
            failures.append({"path": str(path), "expected": expected, "actual": actual})
    return {"passed": not failures, "total": total, "failures": failures}


def compare_validation_metrics(
    candidate: Optional[Dict[str, Any]],
    active: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    candidate = candidate or {}
    active = active or {}
    if not candidate.get("available"):
        return {"passed": False, "reason": "候选模型缺少可用的分组验证指标", "comparisons": {}}
    comparisons: Dict[str, Any] = {}
    passed = True
    for name in METRIC_NAMES:
        current_value = active.get(name)
        candidate_value = candidate.get(name)
        metric_passed = candidate_value is not None and (
            current_value is None or float(candidate_value) >= float(current_value)
        )
        comparisons[name] = {
            "candidate": candidate_value,
            "active": current_value,
            "passed": metric_passed,
        }
        passed = passed and metric_passed
    return {"passed": passed, "comparisons": comparisons}


def compare_same_test_metrics(
    candidate: Optional[Dict[str, Any]],
    active: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    candidate = candidate or {}
    active = active or {}
    if not candidate.get("available") or not active.get("available"):
        return {"passed": False, "reason": "活跃模型或候选模型缺少同一冻结 test 指标", "comparisons": {}}
    normal_count = int(candidate.get("normal_sample_count") or 0)
    tampered_count = int(candidate.get("tampered_sample_count") or 0)
    if normal_count < MINIMUM_TEST_CLASS_SAMPLES or tampered_count < MINIMUM_TEST_CLASS_SAMPLES:
        return {
            "passed": False,
            "reason": f"冻结 test 每类至少需要 {MINIMUM_TEST_CLASS_SAMPLES} 张",
            "normal_sample_count": normal_count,
            "tampered_sample_count": tampered_count,
            "comparisons": {},
        }
    if (
        int(active.get("normal_sample_count") or -1) != normal_count
        or int(active.get("tampered_sample_count") or -1) != tampered_count
        or int(active.get("sample_count") or -1) != int(candidate.get("sample_count") or -2)
    ):
        return {"passed": False, "reason": "活跃模型与候选模型未使用完全相同的冻结 test", "comparisons": {}}
    comparisons = {}
    passed = True
    for name in METRIC_NAMES:
        candidate_value = candidate.get(name)
        active_value = active.get(name)
        metric_passed = (
            candidate_value is not None
            and active_value is not None
            and float(candidate_value) >= float(active_value) - MAX_ALLOWED_TEST_REGRESSION
        )
        comparisons[name] = {
            "candidate": candidate_value,
            "active": active_value,
            "allowed_regression": MAX_ALLOWED_TEST_REGRESSION,
            "passed": metric_passed,
        }
        passed = passed and metric_passed
    return {
        "passed": passed,
        "normal_sample_count": normal_count,
        "tampered_sample_count": tampered_count,
        "comparisons": comparisons,
    }


def compare_roi_coverage(
    candidate: Optional[Dict[str, Any]],
    active: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    candidate = candidate or {}
    active = active or {}
    value = candidate.get("roi_coverage", candidate.get("coverage"))
    active_value = active.get("roi_coverage", active.get("coverage"))
    passed = (
        value is not None
        and active_value is not None
        and float(value) >= MINIMUM_ROI_COVERAGE
        and float(value) >= float(active_value) - MAX_ALLOWED_TEST_REGRESSION
    )
    return {
        "passed": passed,
        "candidate": value,
        "active": active_value,
        "minimum": MINIMUM_ROI_COVERAGE,
        "allowed_regression": MAX_ALLOWED_TEST_REGRESSION,
        "sample_count": int(candidate.get("sample_count") or 0),
    }


def build_candidate_gates(
    *,
    regression_predictions: Iterable[tuple[str, int, str]],
    candidate_metrics: Optional[Dict[str, Any]],
    active_metrics: Optional[Dict[str, Any]],
    training_replay_predictions: Iterable[tuple[str, int, str]] = (),
    holdout_metrics: Optional[Dict[str, Any]] = None,
    roi_coverage: Optional[Dict[str, Any]] = None,
    candidate_test_metrics: Optional[Dict[str, Any]] = None,
    active_test_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    regression = evaluate_regression_gate(regression_predictions)
    replay = evaluate_regression_gate(training_replay_predictions)
    validation = compare_validation_metrics(candidate_metrics, active_metrics)
    candidate_test = candidate_test_metrics or holdout_metrics
    same_test = compare_same_test_metrics(candidate_test, active_test_metrics)
    coverage = compare_roi_coverage(candidate_test or roi_coverage, active_test_metrics)
    return {
        "passed": bool(
            regression["passed"]
            and replay["passed"]
            and same_test["passed"]
            and coverage["passed"]
        ),
        "fixed_regression": regression,
        "training_replay_regression": replay,
        "training_profile_validation": validation,
        "same_test_metrics": same_test,
        "roi_coverage": coverage,
    }


def _manifest_samples(base: Path, marker: str):
    manifest_path = base / "dataset_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    seen = set()
    for row in manifest.get("entries", []):
        if not row.get(marker) or not is_v3_baseline_sample(row):
            continue
        path = (base / str(row.get("path") or "")).resolve()
        if path.is_file() and path not in seen:
            seen.add(path)
            yield path, int(row.get("label", 1))


def fixed_regression_samples(base_dir: str | Path, _pptest_dir: Optional[str | Path] = None):
    base = Path(base_dir)
    manifest_path = base / "dataset_manifest.json"
    if manifest_path.is_file():
        yield from _manifest_samples(base, "fixed_regression")
        return
    for class_name, label in (("normal", 0), ("tampered", 1)):
        class_dir = base / class_name
        if class_dir.is_dir():
            for path in sorted(class_dir.glob("*")):
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    yield path, label


def training_replay_samples(base_dir: str | Path):
    yield from _manifest_samples(Path(base_dir), "training_replay_regression")


def holdout_samples(base_dir: str | Path):
    base = Path(base_dir)
    manifest_path = base / "dataset_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    seen = set()
    for row in manifest.get("entries", []):
        if row.get("split") != "test" or not is_v3_candidate_gate_sample(row):
            continue
        path = (base / str(row.get("path") or "")).resolve()
        if path.is_file() and path not in seen:
            seen.add(path)
            yield path, int(row.get("label", 1))
