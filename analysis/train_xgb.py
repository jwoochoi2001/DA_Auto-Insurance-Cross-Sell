# -*- coding: utf-8 -*-
"""
XGBoost 학습 및 평가

모델
- XGBClassifier(n_estimators=1000, max_depth=4, learning_rate=0.05,
  random_state=42), eval_metric="auc"
- 조기 종료: 검증 ROC-AUC 가 30 라운드 연속 개선되지 않으면 중단
  (early_stopping_rounds=30, eval_set=valid)

전처리
- 문자형 범주(Gender, Region_Code, Vehicle_Damage, Policy_Sales_Channel)는
  train 에서만 학습한 원-핫 인코더로 변환하고 valid/test 에 동일 적용.
- Vehicle_Age_ord(0/1/2), Previously_Insured(0/1) 등 수치형은 그대로 사용.

평가는 valid 로 한다(최종 판단 기준 확정 전이므로 test 는 봉인).

출력
- outputs/figures/xgb_evaluation.png
- outputs/tables/xgb_metrics.csv
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
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


def make_pipeline(**xgb_kwargs) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL)],
        remainder="passthrough",
    )
    params = dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                  random_state=RANDOM_STATE, eval_metric="auc", n_jobs=-1)
    params.update(xgb_kwargs)
    return Pipeline([("pre", pre), ("xgb", XGBClassifier(**params))])


def cross_validate(X_train, y_train):
    """5-겹 층화 교차검증 ROC-AUC (조기종료 없이 400그루 고정)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base = cross_val_score(make_pipeline(), X_train, y_train,
                           scoring="roc_auc", cv=skf, n_jobs=-1)
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    weighted = cross_val_score(make_pipeline(scale_pos_weight=spw), X_train, y_train,
                               scoring="roc_auc", cv=skf, n_jobs=-1)
    print("5-겹 CV ROC-AUC")
    print(f"  기본(scale_pos_weight=1)        : {base.mean():.4f} ± {base.std():.4f}")
    print(f"  불균형 보정(scale_pos_weight={spw:.1f}) : {weighted.mean():.4f} ± {weighted.std():.4f}")
    print("  -> AUC 차이가 미미하므로 기본값 사용, 불균형은 임계값 조정으로 대응\n")
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "설정": ["scale_pos_weight=1", f"scale_pos_weight={spw:.1f}"],
        "CV_ROC_AUC_평균": [round(base.mean(), 4), round(weighted.mean(), 4)],
        "CV_ROC_AUC_표준편차": [round(base.std(), 4), round(weighted.std(), 4)],
    }).to_csv(TBL_DIR / "xgb_cv.csv", index=False, encoding="utf-8")
    return base, weighted


def main():
    X_train, y_train = load("train")
    X_valid, y_valid = load("valid")

    cv_base, cv_weighted = cross_validate(X_train, y_train)

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL)],
        remainder="passthrough",
    )
    Xtr = pre.fit_transform(X_train)
    Xva = pre.transform(X_valid)
    print(f"원-핫 인코딩 후 입력 피처 수: {Xtr.shape[1]}")

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        eval_metric="auc",
        early_stopping_rounds=30,
        n_jobs=-1,
    )
    model.fit(Xtr, y_train, eval_set=[(Xva, y_valid)], verbose=50)

    best_iter = model.best_iteration
    best_score = model.best_score
    print(f"\n조기 종료: best_iteration={best_iter} (총 트리 {best_iter + 1}그루), "
          f"검증 ROC-AUC={best_score:.4f}")

    proba = model.predict_proba(Xva)[:, 1]
    base = y_valid.mean()
    auc = roc_auc_score(y_valid, proba)

    def summarize(thr):
        p = (proba >= thr).astype(int)
        return {
            "임계값": thr,
            "정확도(Accuracy)": accuracy_score(y_valid, p),
            "정밀도(Precision)": precision_score(y_valid, p, zero_division=0),
            "재현율(Recall)": recall_score(y_valid, p, zero_division=0),
            "F1": f1_score(y_valid, p, zero_division=0),
            "ROC-AUC": auc,
        }

    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f1s = [f1_score(y_valid, (proba >= t).astype(int), zero_division=0) for t in grid]
    precs = [precision_score(y_valid, (proba >= t).astype(int), zero_division=0) for t in grid]
    recs = [recall_score(y_valid, (proba >= t).astype(int), zero_division=0) for t in grid]
    thr_best = float(grid[int(np.argmax(f1s))])

    m_default, m_best = summarize(0.5), summarize(thr_best)
    cm_default = confusion_matrix(y_valid, (proba >= 0.5).astype(int))
    cm_best = confusion_matrix(y_valid, (proba >= thr_best).astype(int))

    print("\n=== valid 평가 ===")
    for tag, m in [("임계값 0.5", m_default), (f"임계값 {thr_best:.2f}(F1 최대)", m_best)]:
        print(f"[{tag}] " + " / ".join(f"{k} {v:.4f}" for k, v in m.items() if k != "임계값"))
    print("혼동행렬 @0.5:\n", cm_default)
    print(f"혼동행렬 @{thr_best:.2f}:\n", cm_best)

    TBL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([m_default, m_best]).to_csv(TBL_DIR / "xgb_metrics.csv", index=False, encoding="utf-8")

    # ---- 시각화 ----
    fpr, tpr, _ = roc_curve(y_valid, proba)
    prec_c, rec_c, _ = precision_recall_curve(y_valid, proba)
    ap = average_precision_score(y_valid, proba)
    evals = model.evals_result()["validation_0"]["auc"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    def draw_cm(ax, cm, title):
        ax.imshow(cm, cmap="Greens")
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

    draw_cm(axes[0, 0], cm_default, "혼동행렬 · 임계값 0.5")
    draw_cm(axes[0, 1], cm_best, f"혼동행렬 · 임계값 {thr_best:.2f} (F1 최대)")

    ax = axes[0, 2]
    keys = ["정확도(Accuracy)", "정밀도(Precision)", "재현율(Recall)", "F1", "ROC-AUC"]
    x = np.arange(len(keys)); w = 0.38
    ax.bar(x - w/2, [m_default[k] for k in keys], w, label="임계값 0.5", color="#B0B0B0")
    ax.bar(x + w/2, [m_best[k] for k in keys], w, label=f"임계값 {thr_best:.2f}", color="#2E7D32")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20); ax.set_ylim(0, 1.05)
    ax.set_title("성능 지표 비교", fontsize=12, fontweight="bold"); ax.legend()
    for i, k in enumerate(keys):
        ax.text(i - w/2, m_default[k] + 0.02, f"{m_default[k]:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, m_best[k] + 0.02, f"{m_best[k]:.2f}", ha="center", fontsize=9)

    ax = axes[1, 0]
    ax.plot(fpr, tpr, color="#2E7D32", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="gray", label="무작위 (0.5)")
    ax.set_xlabel("거짓 양성 비율 (FPR)"); ax.set_ylabel("참 양성 비율 = 재현율 (TPR)")
    ax.set_title("ROC 곡선", fontsize=12, fontweight="bold"); ax.legend(loc="lower right")

    ax = axes[1, 1]
    ax.plot(rec_c, prec_c, color="#C44E52", lw=2, label=f"PR (AP = {ap:.3f})")
    ax.axhline(base, ls="--", c="gray", label=f"기준선 (양성률 {base:.3f})")
    ax.set_xlabel("재현율 (Recall)"); ax.set_ylabel("정밀도 (Precision)")
    ax.set_title("정밀도-재현율 곡선", fontsize=12, fontweight="bold"); ax.legend(loc="upper right")

    ax = axes[1, 2]
    ax.plot(range(1, len(evals) + 1), evals, color="#4C72B0", lw=1.5)
    ax.axvline(best_iter + 1, ls="--", c="gray", label=f"best = {best_iter + 1}그루")
    ax.set_xlabel("트리 개수 (부스팅 라운드)"); ax.set_ylabel("검증 ROC-AUC")
    ax.set_title("학습 곡선 (조기 종료)", fontsize=12, fontweight="bold"); ax.legend(loc="lower right")

    fig.suptitle(
        f"XGBoost 성능 (검증셋)  ·  5겹 교차검증 ROC-AUC {cv_base.mean():.3f} ± {cv_base.std():.3f}",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "xgb_evaluation.png", dpi=120)
    print("\n저장: outputs/figures/xgb_evaluation.png, outputs/tables/xgb_metrics.csv")


if __name__ == "__main__":
    main()
