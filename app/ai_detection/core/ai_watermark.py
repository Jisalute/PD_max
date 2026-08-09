"""基于 OCR 文本识别明确的 AIGC 平台水印。"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np

from app.ai_detection.core.amount_candidates import OCRToken, group_tokens_by_line, normalize_text


DOUBAO_WATERMARK_TEXT = "豆包AI生成"
AI_GENERATED_DOCUMENT_EVIDENCE = "ai_generated_document"
DOUBAO_WATERMARK_SOURCE = "doubao_ai_watermark_ocr"
DOUBAO_WATERMARK_REASON = "OCR识别到“豆包AI生成”水印，按业务规则直接判定篡改"
DOUBAO_WATERMARK_TEMPLATE_SOURCE = "doubao_ai_watermark_template"
DOUBAO_WATERMARK_TEMPLATE_REASON = "识别到“豆包AI生成”水印轮廓，按业务规则直接判定篡改"
WATERMARK_TEMPLATE_THRESHOLD = 0.70
_WATERMARK_TEMPLATE_PATH = Path(__file__).with_name("assets") / "doubao_watermark_templates.npz"

# 仅容忍 OCR 对 I 的常见混淆；不能把普通的“豆包”或“AI”单独作为强判依据。
_DOUBAO_WATERMARK_PATTERN = re.compile(r"豆包A[IL1]生成", re.IGNORECASE)


def _normalize_watermark_text(text: str) -> str:
    normalized = normalize_text(str(text or ""))
    normalized = re.sub(r"\s+", "", normalized).upper()
    return normalized.replace("|", "I").replace("!", "I").replace("Ｉ", "I").replace("１", "1")


def _merge_bbox(tokens: Sequence[OCRToken]) -> list[int]:
    return [
        min(int(token.bbox[0]) for token in tokens),
        min(int(token.bbox[1]) for token in tokens),
        max(int(token.bbox[2]) for token in tokens),
        max(int(token.bbox[3]) for token in tokens),
    ]


def _build_evidence(tokens: Sequence[OCRToken]) -> Dict[str, Any]:
    matched_text = "".join(str(token.text) for token in tokens)
    confidence = sum(float(token.conf) for token in tokens) / max(1, len(tokens))
    return {
        "detected": True,
        "hard_tamper": True,
        "evidence_type": AI_GENERATED_DOCUMENT_EVIDENCE,
        "source": DOUBAO_WATERMARK_SOURCE,
        "watermark_text": DOUBAO_WATERMARK_TEXT,
        "matched_text": matched_text,
        "bbox_xyxy": _merge_bbox(tokens),
        "ocr_confidence": round(confidence, 4),
        "reason": DOUBAO_WATERMARK_REASON,
    }


def detect_doubao_ai_watermark(tokens: Sequence[OCRToken]) -> Optional[Dict[str, Any]]:
    """在单 token 或相邻 OCR token 中定位“豆包AI生成”水印。"""
    normalized_tokens = [(token, _normalize_watermark_text(token.clean_text or token.text)) for token in tokens]
    for token, text in normalized_tokens:
        if _DOUBAO_WATERMARK_PATTERN.search(text):
            return _build_evidence([token])

    for line in group_tokens_by_line(tokens):
        ranges: list[tuple[int, int, OCRToken]] = []
        pieces: list[str] = []
        offset = 0
        for token in line:
            text = _normalize_watermark_text(token.clean_text or token.text)
            if not text:
                continue
            pieces.append(text)
            next_offset = offset + len(text)
            ranges.append((offset, next_offset, token))
            offset = next_offset
        match = _DOUBAO_WATERMARK_PATTERN.search("".join(pieces))
        if not match:
            continue
        matched_tokens = [
            token
            for start, end, token in ranges
            if start < match.end() and end > match.start()
        ]
        if matched_tokens:
            return _build_evidence(matched_tokens)
    return None


@lru_cache(maxsize=1)
def _load_doubao_watermark_templates() -> tuple[np.ndarray, ...]:
    """加载固定水印轮廓模板；缺失资源时只保留 OCR 精确匹配。"""
    if not _WATERMARK_TEMPLATE_PATH.is_file():
        return ()
    try:
        with np.load(_WATERMARK_TEMPLATE_PATH, allow_pickle=False) as data:
            templates = [
                np.asarray(data[name], dtype=np.uint8)
                for name in sorted(data.files)
                if data[name].ndim == 2
            ]
    except (OSError, ValueError):
        return ()
    return tuple(template for template in templates if template.size)


def _watermark_template_feature(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    x1 = min(max(0, int(width * 0.74)), max(0, width - 1))
    y1 = min(max(0, int(height * 0.93)), max(0, height - 1))
    x2 = min(width, max(x1 + 1, int(width * 0.999)))
    y2 = min(height, max(y1 + 1, int(height * 0.999)))
    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    outline = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel) - gray
    return np.clip(outline, 0, 80).astype(np.uint8), x1, y1


def detect_doubao_ai_watermark_template(image: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    """以固定位置的水印轮廓补足半透明文字的 OCR 漏检。"""
    if image is None or image.size == 0 or image.ndim < 2:
        return None
    templates = _load_doubao_watermark_templates()
    if not templates:
        return None

    feature, origin_x, origin_y = _watermark_template_feature(image)
    if feature.shape[0] < 12 or feature.shape[1] < 20:
        return None

    best_score = -1.0
    best_location: Optional[tuple[int, int]] = None
    best_size: Optional[tuple[int, int]] = None
    for template in templates:
        for scale in np.linspace(0.55, 1.80, 26):
            target_h = max(8, int(template.shape[0] * float(scale)))
            target_w = max(12, int(template.shape[1] * float(scale)))
            if target_h >= feature.shape[0] or target_w >= feature.shape[1]:
                continue
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            candidate = cv2.resize(template, (target_w, target_h), interpolation=interpolation)
            mask = np.where(candidate > 10, 255, 0).astype(np.uint8)
            if not np.any(mask):
                continue
            scores = cv2.matchTemplate(feature, candidate, cv2.TM_CCORR_NORMED, mask=mask)
            finite_scores = np.where(np.isfinite(scores), scores, -1.0)
            _, score, _, location = cv2.minMaxLoc(finite_scores)
            if float(score) > best_score:
                best_score = float(score)
                best_location = (int(location[0]), int(location[1]))
                best_size = (target_w, target_h)

    if best_score < WATERMARK_TEMPLATE_THRESHOLD or best_location is None or best_size is None:
        return None
    x1 = origin_x + best_location[0]
    y1 = origin_y + best_location[1]
    return {
        "detected": True,
        "hard_tamper": True,
        "evidence_type": AI_GENERATED_DOCUMENT_EVIDENCE,
        "source": DOUBAO_WATERMARK_TEMPLATE_SOURCE,
        "watermark_text": DOUBAO_WATERMARK_TEXT,
        "matched_text": DOUBAO_WATERMARK_TEXT,
        "bbox_xyxy": [x1, y1, x1 + best_size[0], y1 + best_size[1]],
        "watermark_score": round(best_score, 6),
        "detection_method": "watermark_template_fallback",
        "reason": DOUBAO_WATERMARK_TEMPLATE_REASON,
    }
