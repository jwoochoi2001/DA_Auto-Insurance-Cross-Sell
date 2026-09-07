# -*- coding: utf-8 -*-
"""
분할된 원본 데이터(train/valid/test_raw)를 모델 입력용으로 전처리

적용 규칙
- 원본(data/raw/)은 읽지 않는다. split_data.py 가 만든 *_raw.csv 만 사용.
- Driving_License        : 제외 (준상수, 값 1 비율 99.79%)
- Vehicle_Age            : 순서형 인코딩 (< 1 Year=0, 1-2 Year=1, > 2 Years=2)
- Region_Code            : 문자열 라벨('R28')로 변환 (범주형)
- Policy_Sales_Channel   : 빈도 상위 N개만 유지, 나머지는 'CHOTHER' 로 통합
                           --> 상위 목록은 **학습(train) 데이터에서만** 산출하고
                               검증/테스트에 동일 적용 (데이터 누수 방지)
- id, Response, 그 외 변수(Gender, Age, Previously_Insured,
  Vehicle_Damage, Annual_Premium, Vintage)는 원본 값 유지

출력
- data/processed/{train,valid,test}.csv
- data/processed/preprocess_params.json  (상위 채널 목록 등 학습셋 기준값)
"""

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PARAMS_PATH = PROC / "preprocess_params.json"

TOP_N_CHANNEL = 10
VEHICLE_AGE_MAP = {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}
ORDERED = [
    "id", "Gender", "Age", "Region_Code", "Previously_Insured",
    "Vehicle_Age_ord", "Vehicle_Damage", "Annual_Premium",
    "Policy_Sales_Channel", "Vintage", "Response",
]


def fit_params(train_raw: pd.DataFrame) -> dict:
    """학습 데이터에서만 전처리 기준값을 산출한다."""
    ch = train_raw["Policy_Sales_Channel"].round().astype("int64").astype("string")
    top = ch.value_counts().head(TOP_N_CHANNEL).index.tolist()
    return {
        "top_channels": ["CH" + c for c in top],
        "top_n_channel": TOP_N_CHANNEL,
        "vehicle_age_map": VEHICLE_AGE_MAP,
        "dropped_columns": ["Driving_License"],
        "train_channel_cardinality": int(ch.nunique()),
    }


def transform(raw: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = raw.drop(columns=params["dropped_columns"]).copy()

    unmapped = set(df["Vehicle_Age"].unique()) - set(VEHICLE_AGE_MAP)
    assert not unmapped, f"매핑되지 않은 Vehicle_Age 값: {unmapped}"
    df["Vehicle_Age_ord"] = df["Vehicle_Age"].map(VEHICLE_AGE_MAP).astype("int8")
    df = df.drop(columns=["Vehicle_Age"])

    df["Region_Code"] = "R" + df["Region_Code"].round().astype("int64").astype("string")

    top = set(params["top_channels"])
    ch = "CH" + df["Policy_Sales_Channel"].round().astype("int64").astype("string")
    df["Policy_Sales_Channel"] = ch.where(ch.isin(top), other="CHOTHER")

    df = df[ORDERED]
    assert df.isnull().sum().sum() == 0, "결측치가 발생했습니다."
    return df


def main() -> None:
    train_raw = pd.read_csv(PROC / "train_raw.csv")
    params = fit_params(train_raw)
    PARAMS_PATH.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"학습셋 기준 상위 {TOP_N_CHANNEL}개 채널: {', '.join(params['top_channels'])}")
    print(f"(학습셋 채널 고유값 {params['train_channel_cardinality']}개 -> 나머지는 CHOTHER)")
    print()

    for name in ("train", "valid", "test"):
        raw = pd.read_csv(PROC / f"{name}_raw.csv")
        out = transform(raw, params)
        out.to_csv(PROC / f"{name}.csv", index=False, encoding="utf-8")
        n_other = int((out["Policy_Sales_Channel"] == "CHOTHER").sum())
        print(f"{name:<6}: {len(out):>7,}행 / CHOTHER {n_other:>6,}행 "
              f"({n_other/len(out)*100:.1f}%) / 결측치 0")

    print("\n저장: data/processed/{train,valid,test}.csv, preprocess_params.json")


if __name__ == "__main__":
    main()
