"""数据集角色判定，供训练和评估共享。"""

from __future__ import annotations

from typing import Any, Mapping


AIGC_DOCUMENT_SOURCE = "ai_generated_document"


def is_ai_generated_document(row: Mapping[str, Any]) -> bool:
    """识别需要与 v3 局部篡改基线隔离的 AIGC 文档。"""
    source = str(row.get("source") or "").strip().lower()
    fraud_type = str(row.get("fraud_type") or "").strip().lower()
    aigc = row.get("aigc")
    return (
        source == AIGC_DOCUMENT_SOURCE
        or fraud_type == AIGC_DOCUMENT_SOURCE
        or isinstance(aigc, Mapping)
    )


def is_v3_baseline_sample(row: Mapping[str, Any]) -> bool:
    """返回可用于 v3 局部篡改训练和常规评估的规范原图。"""
    if bool(row.get("is_derived")) or str(row.get("split") or "") == "derived":
        return False
    if bool(row.get("exclude_from_v3_training")) or bool(row.get("exclude_from_v3_evaluation")):
        return False
    # AIGC 目前没有金额、姓名、时间篡改真值，不能让全图痕迹污染 ROI 基线。
    return not is_ai_generated_document(row)


def is_v3_candidate_gate_sample(row: Mapping[str, Any]) -> bool:
    """返回可进入活跃/候选 v3 同集门禁的样本。"""
    return is_v3_baseline_sample(row) and not bool(row.get("exclude_from_v3_candidate_gate"))
