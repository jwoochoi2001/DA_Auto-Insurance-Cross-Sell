# -*- coding: utf-8 -*-
"""
원본 데이터를 학습/검증/테스트로 분할 (전처리보다 먼저 수행)

- 비율        : train 60% / valid 20% / test 20%
- random_state : 42 고정
- Response(불균형, 양성 12.26%) 기준 층화(stratify) 분할
- 입력 : data/raw/insurance_cross_sell.csv   (원본, 수정하지 않음)
- 출력 : data/processed/{train,valid,test}_raw.csv

전처리 기준(예: 판매채널 상위 목록)을 학습 데이터에서만 정하도록,
분할을 전처리보다 앞에 둔다.  이후 analysis/preprocess.py 실행.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "insurance_cross_sell.csv"
OUT_DIR = ROOT / "data" / "processed"

RANDOM_STATE = 42
TARGET = "Response"


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    n = len(df)

    train_df, temp_df = train_test_split(
        df, test_size=0.40, random_state=RANDOM_STATE, stratify=df[TARGET],
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_df[TARGET],
    )

    splits = {"train": train_df, "valid": valid_df, "test": test_df}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(OUT_DIR / f"{name}_raw.csv", index=False, encoding="utf-8")

    assert sum(len(p) for p in splits.values()) == n, "행 수 합계 불일치"
    ids = set()
    for p in splits.values():
        ids |= set(p["id"])
    assert len(ids) == n, "분할 간 id 중복 존재"

    print(f"전체 행 수: {n:,}")
    print(f"{'split':<8}{'행 수':>12}{'비율':>10}{'Response=1':>14}{'양성률(%)':>12}")
    for name, part in splits.items():
        print(
            f"{name:<8}{len(part):>12,}{len(part)/n:>10.4f}"
            f"{int(part[TARGET].sum()):>14,}{part[TARGET].mean()*100:>12.4f}"
        )
    print("\n저장: data/processed/{train,valid,test}_raw.csv")


if __name__ == "__main__":
    main()
