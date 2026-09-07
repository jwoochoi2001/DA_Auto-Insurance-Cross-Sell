# -*- coding: utf-8 -*-
"""
최종 테스트셋 검증 : 랜덤 포레스트 vs XGBoost

절차
1) train 으로 두 모델 학습 (앞 단계와 동일한 설정, random_state=42)
2) 운영 임계값은 valid 에서 F1 최대 지점으로 결정 (test 로는 튜닝하지 않음)
3) 결정된 임계값을 test 에 그대로 적용해 최종 성능 산출
4) valid 대비 test 성능 차이로 과적합 여부 확인

전처리: 문자형 범주 4개를 train 기준 원-핫 인코딩 -> valid/test 동일 적용.

출력
- outputs/figures/test_evaluation.png
- outputs/tables/test_metrics.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "analysis"))
from plotting import setup_korean_font  # noqa: E402

PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
TBL_DIR = ROOT / "outputs" / "tables"

RANDOM_STATE = 42
TARGET = "Response"
DROP = ["id"]
NOMINAL = ["Gender", "Region_Code", "Vehicle_Damage", "Policy_Sales_Channel"]

setup_korean_font()


def load(name):
    df = pd.read_csv(PROC / f"{name}.csv")
    return df.drop(columns=DROP + [TARGET]), df[TARGET].to_numpy()


def best_f1_threshold(y, proba):
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f1s = [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def metric_row(y, proba, thr):
    p = (proba >= thr).astype(int)
    return {
        "정확도(Accuracy)": accuracy_score(y, p),
        "정밀도(Precision)": precision_score(y, p, zero_division=0),
        "재현율(Recall)": recall_score(y, p, zero_division=0),
        "F1": f1_score(y, p, zero_division=0),
        "ROC-AUC": roc_auc_score(y, proba),
    }


def main():
    X_train, y_train = load("train")
    X_valid, y_valid = load("valid")
    X_test, y_test = load("test")

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL)],
        remainder="passthrough",
    )
    Xtr = pre.fit_transform(X_train)
    Xva = pre.transform(X_valid)
    Xte = pre.transform(X_test)

    # ---- 모델 학습 ----
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_split=20,
        random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(Xtr, y_train)

    xgb = XGBClassifier(
        n_estimators=1000, max_depth=4, learning_rate=0.05,
        random_state=RANDOM_STATE, eval_metric="auc",
        early_stopping_rounds=30, n_jobs=-1,
    )
    xgb.fit(Xtr, y_train, eval_set=[(Xva, y_valid)], verbose=False)
    print(f"XGBoost 조기 종료: 트리 {xgb.best_iteration + 1}그루")

    models = {"랜덤 포레스트": rf, "XGBoost": xgb}
    rows = []
    curves = {}
    for name, mdl in models.items():
        pv = mdl.predict_proba(Xva)[:, 1]
        pt = mdl.predict_proba(Xte)[:, 1]
        thr = best_f1_threshold(y_valid, pv)          # 임계값은 valid 에서 결정

        mv = metric_row(y_valid, pv, thr)
        mt = metric_row(y_test, pt, thr)
        cm = confusion_matrix(y_test, (pt >= thr).astype(int))
        curves[name] = {
            "roc": roc_curve(y_test, pt),
            "pr": precision_recall_curve(y_test, pt),
            "ap": average_precision_score(y_test, pt),
            "cm": cm, "thr": thr,
        }
        for split, m in [("valid", mv), ("test", mt)]:
            rows.append({"모델": name, "임계값": thr, "구분": split, **m})

        print(f"\n[{name}] 운영 임계값(valid F1 최대) = {thr:.2f}")
        print(f"  valid: " + " / ".join(f"{k} {v:.4f}" for k, v in mv.items()))
        print(f"  test : " + " / ".join(f"{k} {v:.4f}" for k, v in mt.items()))
        print(f"  test 혼동행렬 [[TN,FP],[FN,TP]]:\n{cm}")

    res = pd.DataFrame(rows)
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(TBL_DIR / "test_metrics.csv", index=False, encoding="utf-8")

    # ---- 시각화 ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {"랜덤 포레스트": "#4C72B0", "XGBoost": "#2E7D32"}

    def draw_cm(ax, cm, title):
        ax.imshow(cm, cmap="Blues")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["예측 0\n(관심없음)", "예측 1\n(관심있음)"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["실제 0\n(관심없음)", "실제 1\n(관심있음)"])
        names = [["TN", "FP"], ["FN", "TP"]]
        tot = cm.sum()
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{names[i][j]}\n{cm[i,j]:,}\n({cm[i,j]/tot*100:.1f}%)",
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)

    for ax, (name, c) in zip(axes[0, :2], curves.items()):
        draw_cm(ax, c["cm"], f"{name} · 테스트셋 (임계값 {c['thr']:.2f})")

    # 지표 비교 (test, 두 모델)
    ax = axes[0, 2]
    keys = ["정확도(Accuracy)", "정밀도(Precision)", "재현율(Recall)", "F1", "ROC-AUC"]
    x = np.arange(len(keys)); w = 0.38
    for i, name in enumerate(models):
        mt = res[(res["모델"] == name) & (res["구분"] == "test")].iloc[0]
        ax.bar(x + (i - 0.5) * w, [mt[k] for k in keys], w, label=name, color=colors[name])
        for j, k in enumerate(keys):
            ax.text(x[j] + (i - 0.5) * w, mt[k] + 0.02, f"{mt[k]:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20); ax.set_ylim(0, 1.05)
    ax.set_title("테스트셋 성능 비교", fontsize=12, fontweight="bold"); ax.legend()

    # ROC (test, 두 모델)
    ax = axes[1, 0]
    for name, c in curves.items():
        fpr, tpr, _ = c["roc"]
        auc = res[(res["모델"] == name) & (res["구분"] == "test")]["ROC-AUC"].iloc[0]
        ax.plot(fpr, tpr, color=colors[name], lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="gray")
    ax.set_xlabel("거짓 양성 비율 (FPR)"); ax.set_ylabel("참 양성 비율 = 재현율")
    ax.set_title("ROC 곡선 (테스트셋)", fontsize=12, fontweight="bold"); ax.legend(loc="lower right")

    # PR (test, 두 모델)
    ax = axes[1, 1]
    for name, c in curves.items():
        prec, rec, _ = c["pr"]
        ax.plot(rec, prec, color=colors[name], lw=2, label=f"{name} (AP={c['ap']:.3f})")
    ax.axhline(y_test.mean(), ls="--", c="gray", label=f"기준선 (양성률 {y_test.mean():.3f})")
    ax.set_xlabel("재현율 (Recall)"); ax.set_ylabel("정밀도 (Precision)")
    ax.set_title("정밀도-재현율 곡선 (테스트셋)", fontsize=12, fontweight="bold"); ax.legend(loc="upper right")

    # valid vs test (과적합 점검)
    ax = axes[1, 2]
    x2 = np.arange(len(keys)); w2 = 0.2
    for i, (name, split) in enumerate([("랜덤 포레스트", "valid"), ("랜덤 포레스트", "test"),
                                        ("XGBoost", "valid"), ("XGBoost", "test")]):
        r = res[(res["모델"] == name) & (res["구분"] == split)].iloc[0]
        hatch = "" if split == "test" else "//"
        ax.bar(x2 + (i - 1.5) * w2, [r[k] for k in keys], w2,
               label=f"{name}·{split}", color=colors[name], alpha=0.6 if split == "valid" else 1.0,
               hatch=hatch)
    ax.set_xticks(x2); ax.set_xticklabels(keys, rotation=20); ax.set_ylim(0, 1.05)
    ax.set_title("검증 vs 테스트 (과적합 점검)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle("최종 테스트셋 검증 : 랜덤 포레스트 vs XGBoost", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "test_evaluation.png", dpi=120)
    print("\n저장: outputs/figures/test_evaluation.png, outputs/tables/test_metrics.csv")


if __name__ == "__main__":
    main()
