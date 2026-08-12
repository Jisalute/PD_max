"""每日 AI 预测使用的报价指纹。"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable


def build_smm_price_fingerprint(items: Iterable[object]) -> str:
    """根据报价日期和均价生成稳定指纹，报价未变化时指纹保持一致。"""
    payload = []
    for item in items:
        price_date = getattr(item, "定价日期", None)
        average_price = getattr(item, "均价", None)
        payload.append(
            [
                price_date.isoformat() if hasattr(price_date, "isoformat") else str(price_date),
                str(average_price),
            ]
        )
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_smm_price_fingerprint_from_rows(rows: Iterable[tuple[object, object]]) -> str:
    """根据数据库返回的 (报价日期, 均价) 行生成报价指纹。"""
    payload = [
        [
            price_date.isoformat() if hasattr(price_date, "isoformat") else str(price_date),
            str(average_price),
        ]
        for price_date, average_price in rows
    ]
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
