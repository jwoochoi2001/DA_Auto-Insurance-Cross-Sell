# -*- coding: utf-8 -*-
"""
원본(raw) 변수 -> 모델 입력 형태로 바꾸는 공용 변환 함수.
자산 생성 스크립트와 Streamlit 대시보드가 동일한 규칙을 쓰도록 한곳에 둔다.

모델 입력 열(순서 고정):
  Gender, Age, Region_Code, Previously_Insured,
  Vehicle_Age_ord, Vehicle_Damage, Annual_Premium,
  Policy_Sales_Channel, Vintage
"""
from __future__ import annotations
import pandas as pd

# 전처리 단계(analysis/preprocess.py)와 동일한 정의
VEHICLE_AGE_MAP = {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}
VEHICLE_AGE_INV = {v: k for k, v in VEHICLE_AGE_MAP.items()}
PREV_INSURED_LABEL = {0: "미가입", 1: "기가입"}

MODEL_FEATURES = [
    "Gender", "Age", "Region_Code", "Previously_Insured",
    "Vehicle_Age_ord", "Vehicle_Damage", "Annual_Premium",
    "Policy_Sales_Channel", "Vintage",
]

TIER_LABELS = ["매우 낮음", "낮음", "보통", "높음", "매우 높음"]  # 1~5


def region_label(code) -> str:
    return f"R{int(round(float(code)))}"


def channel_label(code, top_channels: list[str]) -> str:
    """원본 채널 코드 -> 'CH26' 등. 상위 목록에 없으면 'CHOTHER'."""
    lab = f"CH{int(round(float(code)))}"
    return lab if lab in top_channels else "CHOTHER"


def build_model_frame(raw: pd.DataFrame, top_channels: list[str]) -> pd.DataFrame:
    """
    raw 는 다음 원본 열을 가진다고 가정:
      Gender(str), Age(int), Region_Code(number),
      Previously_Insured(0/1), Vehicle_Age(str) 또는 Vehicle_Age_ord(int),
      Vehicle_Damage(str), Annual_Premium(number),
      Policy_Sales_Channel(number) 또는 이미 'CH..' 문자열, Vintage(int)
    """
    df = raw.copy()

    if "Vehicle_Age_ord" not in df.columns:
        df["Vehicle_Age_ord"] = df["Vehicle_Age"].map(VEHICLE_AGE_MAP).astype("int8")

    if not str(df["Region_Code"].iloc[0]).startswith("R"):
        df["Region_Code"] = df["Region_Code"].map(region_label)

    ch = df["Policy_Sales_Channel"]
    if not str(ch.iloc[0]).startswith("CH"):
        df["Policy_Sales_Channel"] = ch.map(lambda c: channel_label(c, top_channels))
    else:
        df["Policy_Sales_Channel"] = ch.where(ch.isin(top_channels), "CHOTHER")

    return df[MODEL_FEATURES].reset_index(drop=True)


def probability_to_tier(prob, cutpoints: list[float]) -> str:
    """cutpoints = [q20, q40, q60, q80] (오름차순). 반환: TIER_LABELS 중 하나."""
    q20, q40, q60, q80 = cutpoints
    p = float(prob)
    if p >= q80:
        return TIER_LABELS[4]
    if p >= q60:
        return TIER_LABELS[3]
    if p >= q40:
        return TIER_LABELS[2]
    if p >= q20:
        return TIER_LABELS[1]
    return TIER_LABELS[0]
