# -*- coding: utf-8 -*-
"""
운영 대시보드용 자산 생성

산출물
- outputs/model/xgb_pipeline.joblib   : 전처리 + XGBoost 파이프라인
- outputs/model/metadata.json         : 운영 임계값, 5단계 등급 컷포인트/실제 관심율,
                                        테스트 성능, 상위 채널 목록, 입력 스키마
- outputs/dashboard/variable_analysis.csv : 변수별 관심율(가입 관심도) 분석표 (train 기준)
- data/processed/sales_customers.csv   : 영업 대상 리스트 시드(테스트셋 상위 확률 고객)

모델: XGBClassifier(n_estimators=1000, max_depth=4, learning_rate=0.05,
      random_state=42), 검증 ROC-AUC 30라운드 조기 종료
평가/임계값/등급 컷포인트는 valid 로 산출한다(테스트셋으로 튜닝하지 않음).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from features import (
    MODEL_FEATURES, TIER_LABELS, VEHICLE_AGE_INV, PREV_INSURED_LABEL,
    probability_to_tier,
)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "outputs" / "model"
DASH_DIR = ROOT / "outputs" / "dashboard"

RANDOM_STATE = 42
TARGET = "Response"
NOMINAL = ["Gender", "Region_Code", "Vehicle_Damage", "Policy_Sales_Channel"]
SEED_LIST_SIZE = 6000

# 비용/이익 가정 (원). 실제 콜센터 단가·리드 가치로 바꿔 쓰는 예시값.
COST_PER_CALL = 3000            # 아웃바운드 통화 1건 비용
VALUE_PER_INTERESTED_LEAD = 40000  # '관심 있음' 리드 1건의 기대 가치


def load(name):
    df = pd.read_csv(PROC / f"{name}.csv")
    X = df[["id"] + MODEL_FEATURES].copy()
    y = df[TARGET].to_numpy()
    return df, X, y


def best_f1_threshold(y, proba):
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f1s = [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def best_profit_threshold(y, proba):
    """기대이익 = TP*가치 - (TP+FP)*통화비용 을 최대화하는 임계값 (valid 기준)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    best_t, best_profit = 0.5, -np.inf
    for t in grid:
        p = (proba >= t).astype(int)
        tp = int(((p == 1) & (y == 1)).sum())
        called = int(p.sum())
        profit = tp * VALUE_PER_INTERESTED_LEAD - called * COST_PER_CALL
        if profit > best_profit:
            best_profit, best_t = profit, float(t)
    return best_t, int(best_profit)


def variable_analysis(train_df: pd.DataFrame) -> pd.DataFrame:
    """train 기준 변수별 관심율(Response=1 비율) 표 (long format)."""
    d = train_df.copy()
    d["나이대"] = pd.cut(d["Age"], [19, 25, 30, 35, 40, 50, 60, 85],
                        labels=["20-25", "26-30", "31-35", "36-40", "41-50", "51-60", "61+"])
    d["연간보험료대"] = pd.qcut(d["Annual_Premium"], 5,
                          labels=["최저 20%", "하위 20-40%", "중간 40-60%", "상위 60-80%", "최고 20%"])
    d["관계기간대"] = pd.cut(d["Vintage"], [9, 73, 146, 219, 299],
                        labels=["10-73일", "74-146일", "147-219일", "220-299일"])
    d["차량연식"] = d["Vehicle_Age_ord"].map(VEHICLE_AGE_INV)
    d["기존_자동차보험"] = d["Previously_Insured"].map(PREV_INSURED_LABEL)

    specs = {
        "성별(Gender)": "Gender",
        "나이대(Age)": "나이대",
        "지역코드(Region_Code)": "Region_Code",
        "기존 자동차보험(Previously_Insured)": "기존_자동차보험",
        "차량 연식(Vehicle_Age)": "차량연식",
        "차량 손상 이력(Vehicle_Damage)": "Vehicle_Damage",
        "연간보험료 구간(Annual_Premium)": "연간보험료대",
        "판매채널(Policy_Sales_Channel)": "Policy_Sales_Channel",
        "관계기간 구간(Vintage)": "관계기간대",
    }
    rows = []
    base = d[TARGET].mean()
    for disp, col in specs.items():
        g = d.groupby(col, observed=True)[TARGET].agg(n="count", rate="mean").reset_index()
        g = g.rename(columns={col: "구간"})
        for _, r in g.iterrows():
            rows.append({
                "변수": disp,
                "구간": str(r["구간"]),
                "고객수": int(r["n"]),
                "관심율(%)": round(r["rate"] * 100, 2),
                "전체평균대비": round(r["rate"] / base, 2),
            })
    return pd.DataFrame(rows)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DASH_DIR.mkdir(parents=True, exist_ok=True)

    train_df, X_train, y_train = load("train")
    valid_df, X_valid, y_valid = load("valid")
    test_df, X_test, y_test = load("test")

    pp = json.loads((PROC / "preprocess_params.json").read_text(encoding="utf-8"))
    top_channels = sorted(pp["top_channels"])

    Xtr = X_train[MODEL_FEATURES]
    Xva = X_valid[MODEL_FEATURES]
    Xte = X_test[MODEL_FEATURES]

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL)],
        remainder="passthrough",
    )
    Xtr_enc = pre.fit_transform(Xtr)
    Xva_enc = pre.transform(Xva)

    xgb = XGBClassifier(
        n_estimators=1000, max_depth=4, learning_rate=0.05,
        random_state=RANDOM_STATE, eval_metric="auc",
        early_stopping_rounds=30, n_jobs=-1,
    )
    xgb.fit(Xtr_enc, y_train, eval_set=[(Xva_enc, y_valid)], verbose=False)
    n_trees = int(xgb.best_iteration + 1)

    pipe = Pipeline([("pre", pre), ("xgb", xgb)])
    joblib.dump(pipe, MODEL_DIR / "xgb_pipeline.joblib")

    # ---- 변수 중요도 ----
    # (1) 순열 중요도: valid 표본에서 변수를 섞었을 때 ROC-AUC 하락폭 (원본 변수 단위)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(Xva), size=min(15000, len(Xva)), replace=False)
    perm = permutation_importance(
        pipe, Xva.iloc[idx], y_valid[idx],
        scoring="roc_auc", n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1,
    )
    # (2) XGB gain 중요도: 원-핫 컬럼별 gain 을 원본 변수로 합산
    enc_names = list(pipe.named_steps["pre"].get_feature_names_out())
    gain = pipe.named_steps["xgb"].feature_importances_
    gain_by_var = {f: 0.0 for f in MODEL_FEATURES}
    for name, g in zip(enc_names, gain):
        core = name.split("__", 1)[-1]
        for f in MODEL_FEATURES:
            if core == f or core.startswith(f + "_"):
                gain_by_var[f] += float(g)
                break
    g_total = sum(gain_by_var.values()) or 1.0

    fi = pd.DataFrame({
        "변수": MODEL_FEATURES,
        "순열중요도": perm.importances_mean.round(4),
        "순열중요도_표준편차": perm.importances_std.round(4),
        "gain중요도(%)": [round(gain_by_var[f] / g_total * 100, 1) for f in MODEL_FEATURES],
    }).sort_values("순열중요도", ascending=False).reset_index(drop=True)
    fi.to_csv(DASH_DIR / "feature_importance.csv", index=False, encoding="utf-8")

    # ---- 5겹 교차검증 ROC-AUC (train, 조기종료 없이 400그루) ----
    cv_pipe = Pipeline([
        ("pre", ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL)],
            remainder="passthrough")),
        ("xgb", XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                              random_state=RANDOM_STATE, eval_metric="auc", n_jobs=-1)),
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(cv_pipe, Xtr, y_train, scoring="roc_auc", cv=skf, n_jobs=-1)
    cv_summary = {"평균": round(float(cv_auc.mean()), 4), "표준편차": round(float(cv_auc.std()), 4)}

    # ---- 운영 임계값 & 등급 컷포인트 (valid) ----
    pv = pipe.predict_proba(Xva)[:, 1]
    thr = best_f1_threshold(y_valid, pv)
    thr_profit, profit_at = best_profit_threshold(y_valid, pv)
    # 5단계 등급 컷포인트: 영업 행동과 직결되도록 확률 절대값 기준으로 고정
    #   매우낮음 <0.05 | 낮음 0.05~기준율 | 보통 기준율~임계값 | 높음 임계값~0.35 | 매우높음 >=0.35
    base_rate = round(float(y_train.mean()), 3)
    cutpoints = sorted({0.05, base_rate, round(thr, 3), 0.35})
    while len(cutpoints) < 4:
        cutpoints.append(cutpoints[-1] + 0.05)
    cutpoints = cutpoints[:4]

    vdf = pd.DataFrame({"p": pv, "y": y_valid})
    vdf["tier"] = vdf["p"].apply(lambda p: probability_to_tier(p, cutpoints))
    tier_stats = {}
    for t in TIER_LABELS:
        sub = vdf[vdf["tier"] == t]
        tier_stats[t] = {
            "고객비율(%)": round(len(sub) / len(vdf) * 100, 1),
            "평균_관심확률(%)": round(float(sub["p"].mean()) * 100, 1) if len(sub) else 0.0,
            "실제_관심율(%)": round(float(sub["y"].mean()) * 100, 1) if len(sub) else 0.0,
        }

    # ---- 테스트 성능 (임계값 valid 결정값 적용) ----
    pt = pipe.predict_proba(Xte)[:, 1]
    pred_t = (pt >= thr).astype(int)
    test_metrics = {
        "ROC_AUC": round(roc_auc_score(y_test, pt), 4),
        "정확도": round(accuracy_score(y_test, pred_t), 4),
        "정밀도": round(precision_score(y_test, pred_t, zero_division=0), 4),
        "재현율": round(recall_score(y_test, pred_t, zero_division=0), 4),
        "F1": round(f1_score(y_test, pred_t, zero_division=0), 4),
    }

    # ---- 변수 분석표 ----
    variable_analysis(train_df).to_csv(
        DASH_DIR / "variable_analysis.csv", index=False, encoding="utf-8"
    )

    # ---- 영업 대상 리스트 시드 (테스트셋 상위 확률 고객) ----
    seed = test_df[["id"] + MODEL_FEATURES].copy()
    seed["관심확률"] = pt
    seed["등급"] = seed["관심확률"].apply(lambda p: probability_to_tier(p, cutpoints))
    seed["관심확률(%)"] = (seed["관심확률"] * 100).round(1)
    seed["차량연식"] = seed["Vehicle_Age_ord"].map(VEHICLE_AGE_INV)
    seed["기존_자동차보험"] = seed["Previously_Insured"].map(PREV_INSURED_LABEL)
    seed["접촉상태"] = "미접촉"
    seed["메모"] = ""
    # 영업 대상 = '보통' 등급 이상(관심확률 >= 기준 관심율) 고객에서 등급별 표본 추출
    #   (실제 콜 풀의 등급 구성을 대시보드에서 보이도록 매우높음/높음/보통을 섞는다)
    pool = seed[seed["관심확률"] >= base_rate]
    quota = {"매우 높음": 2500, "높음": 2000, "보통": 1500}
    parts = []
    for t, q in quota.items():
        sub = pool[pool["등급"] == t]
        parts.append(sub.sample(min(len(sub), q), random_state=RANDOM_STATE))
    seed = (pd.concat(parts)
            .sort_values("관심확률", ascending=False)
            .drop(columns=["관심확률"])
            .reset_index(drop=True))
    seed.to_csv(PROC / "sales_customers.csv", index=False, encoding="utf-8")

    # ---- 메타데이터 ----
    meta = {
        "model": "XGBoost",
        "params": {"n_estimators_max": 1000, "trees_used": n_trees,
                   "max_depth": 4, "learning_rate": 0.05,
                   "early_stopping_rounds": 30, "random_state": RANDOM_STATE},
        "operating_threshold": thr,
        "threshold_f1_max": thr,
        "threshold_profit_max": thr_profit,
        "profit_assumptions": {
            "통화비용_원": COST_PER_CALL,
            "관심리드_가치_원": VALUE_PER_INTERESTED_LEAD,
            "valid_기대이익_원": profit_at,
        },
        "cv_roc_auc_train": cv_summary,
        "tier_labels": TIER_LABELS,
        "tier_cutpoints_prob": cutpoints,
        "tier_stats_valid": tier_stats,
        "test_metrics": test_metrics,
        "base_rate": round(float(y_train.mean()), 4),
        "top_channels": top_channels,
        "sales_list_source": "test split (학습·임계값 선정에 쓰이지 않은 홀드아웃), 신규 고객 대용",
        "model_features": MODEL_FEATURES,
        "seed_list_size": int(len(seed)),
        "input_options": {
            "Gender": ["Male", "Female"],
            "Vehicle_Age": list(VEHICLE_AGE_INV.values()),
            "Vehicle_Damage": ["Yes", "No"],
            "Previously_Insured": [0, 1],
            "Region_Code_range": [0, 52],
            "Age_range": [20, 85],
            "Vintage_range": [10, 299],
            "Annual_Premium_range": [2630, 540165],
        },
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"트리 {n_trees}그루 / 5겹 CV ROC-AUC {cv_summary['평균']}±{cv_summary['표준편차']}")
    print(f"운영 임계값(F1 최대) {thr:.2f} / 이익최대 임계값 {thr_profit:.2f} "
          f"(valid 기대이익 {profit_at:,}원, 가정: 통화 {COST_PER_CALL:,}원 / 리드 {VALUE_PER_INTERESTED_LEAD:,}원)")
    print("등급별(valid):")
    for t, s in tier_stats.items():
        print(f"  {t:>6}: 비율 {s['고객비율(%)']}% / 평균확률 {s['평균_관심확률(%)']}% / 실제관심율 {s['실제_관심율(%)']}%")
    print("테스트 성능:", test_metrics)
    print("저장 완료: outputs/model/, outputs/dashboard/, data/processed/sales_customers.csv")


if __name__ == "__main__":
    main()
