# -*- coding: utf-8 -*-
"""
랜덤 포레스트 학습 및 평가

전처리 점검
- 문자형 범주(Gender, Region_Code, Vehicle_Damage, Policy_Sales_Channel)는
  sklearn 이 직접 쓸 수 없으므로 원-핫 인코딩한다.
- 인코딩 기준(범주 목록)은 train 에서만 학습하고 valid/test 에 동일 적용한다.
  (OneHotEncoder(handle_unknown="ignore"))
- Vehicle_Age_ord(0/1/2), Previously_Insured(0/1)은 이미 수치형이라 그대로 사용.

모델
- RandomForestClassifier(n_estimators=150, max_depth=12,
  min_samples_split=20, random_state=42, n_jobs=-1)
- 최종 판단 기준을 정하기 전이므로 평가는 valid 로 한다. test 는 봉인.

출력
- outputs/figures/rf_evaluation.png  (혼동행렬/지표막대/ROC/PR 곡선)
- outputs/tables/rf_metrics.csv
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
)

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


def load(name: str):
    df = pd.read_csv(PROC / f"{name}.csv")
    y = df[TARGET].to_numpy()
    X = df.drop(columns=DROP + [TARGET])
    return X, y


def main() -> None:
    X_train, y_train = load("train")
    X_valid, y_valid = load("valid")

    # ---- 전처리 점검 출력 ------------------------------------------------
    print("=== 범주형 변수 전처리 점검 ===")
    obj_cols = X_train.select_dtypes(include="object").columns.tolist()
    num_cols = X_train.select_dtypes(exclude="object").columns.tolist()
    print(f"문자형(원-핫 필요): {obj_cols}")
    print(f"수치형(그대로 사용): {num_cols}")
    for c in NOMINAL:
        tr_set, va_set = set(X_train[c]), set(X_valid[c])
        print(f"  - {c}: train {len(tr_set)}개 범주 / valid 신규 범주 {sorted(va_set - tr_set) or '없음'}")

    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL)],
        remainder="passthrough",
    )
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipe = Pipeline([("pre", pre), ("rf", model)])
    pipe.fit(X_train, y_train)

    n_feat = pipe.named_steps["pre"].transform(X_train.head(1)).shape[1]
    print(f"\n원-핫 인코딩 후 총 입력 피처 수: {n_feat}")

    # ---- 평가 (valid) --------------------------------------------------
    proba = pipe.predict_proba(X_valid)[:, 1]
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

    # 임계값 스윕 -> F1 최대 지점을 운영 임계값 후보로 (valid 기준으로 결정, test 미사용)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f1s = [f1_score(y_valid, (proba >= t).astype(int), zero_division=0) for t in grid]
    precs = [precision_score(y_valid, (proba >= t).astype(int), zero_division=0) for t in grid]
    recs = [recall_score(y_valid, (proba >= t).astype(int), zero_division=0) for t in grid]
    thr_best = float(grid[int(np.argmax(f1s))])

    m_default = summarize(0.5)
    m_best = summarize(thr_best)
    cm_default = confusion_matrix(y_valid, (proba >= 0.5).astype(int))
    cm_best = confusion_matrix(y_valid, (proba >= thr_best).astype(int))

    print("\n=== valid 평가 ===")
    for tag, m in [("임계값 0.5", m_default), (f"임계값 {thr_best:.2f}(F1 최대)", m_best)]:
        print(f"[{tag}] " + " / ".join(
            f"{k} {v:.4f}" for k, v in m.items() if k != "임계값"))
    print("혼동행렬 @0.5 [[TN,FP],[FN,TP]]:\n", cm_default)
    print(f"혼동행렬 @{thr_best:.2f} [[TN,FP],[FN,TP]]:\n", cm_best)

    TBL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([m_default, m_best]).to_csv(
        TBL_DIR / "rf_metrics.csv", index=False, encoding="utf-8"
    )

    # ---- 시각화 ------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_valid, proba)
    prec_c, rec_c, _ = precision_recall_curve(y_valid, proba)
    ap = average_precision_score(y_valid, proba)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

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

    draw_cm(axes[0, 0], cm_default, "혼동행렬 · 임계값 0.5")
    draw_cm(axes[0, 1], cm_best, f"혼동행렬 · 임계값 {thr_best:.2f} (F1 최대)")

    # 지표 막대 (두 임계값 비교)
    ax = axes[0, 2]
    keys = ["정확도(Accuracy)", "정밀도(Precision)", "재현율(Recall)", "F1", "ROC-AUC"]
    x = np.arange(len(keys)); w = 0.38
    ax.bar(x - w/2, [m_default[k] for k in keys], w, label="임계값 0.5", color="#B0B0B0")
    ax.bar(x + w/2, [m_best[k] for k in keys], w, label=f"임계값 {thr_best:.2f}", color="#4C72B0")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20); ax.set_ylim(0, 1.05)
    ax.set_title("성능 지표 비교", fontsize=12, fontweight="bold"); ax.legend()
    for i, k in enumerate(keys):
        ax.text(i - w/2, m_default[k] + 0.02, f"{m_default[k]:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, m_best[k] + 0.02, f"{m_best[k]:.2f}", ha="center", fontsize=9)

    # ROC
    ax = axes[1, 0]
    ax.plot(fpr, tpr, color="#55A868", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="gray", label="무작위 (0.5)")
    ax.set_xlabel("거짓 양성 비율 (FPR)"); ax.set_ylabel("참 양성 비율 = 재현율 (TPR)")
    ax.set_title("ROC 곡선", fontsize=12, fontweight="bold"); ax.legend(loc="lower right")

    # PR
    ax = axes[1, 1]
    ax.plot(rec_c, prec_c, color="#C44E52", lw=2, label=f"PR (AP = {ap:.3f})")
    ax.axhline(base, ls="--", c="gray", label=f"기준선 (양성률 {base:.3f})")
    ax.set_xlabel("재현율 (Recall)"); ax.set_ylabel("정밀도 (Precision)")
    ax.set_title("정밀도-재현율 곡선", fontsize=12, fontweight="bold"); ax.legend(loc="upper right")

    # 임계값 스윕
    ax = axes[1, 2]
    ax.plot(grid, precs, label="정밀도", color="#DD8452")
    ax.plot(grid, recs, label="재현율", color="#C44E52")
    ax.plot(grid, f1s, label="F1", color="#4C72B0", lw=2)
    ax.axvline(thr_best, ls="--", c="gray", label=f"F1 최대 = {thr_best:.2f}")
    ax.set_xlabel("임계값 (threshold)"); ax.set_ylabel("점수")
    ax.set_title("임계값에 따른 지표 변화", fontsize=12, fontweight="bold"); ax.legend()

    fig.suptitle("랜덤 포레스트 성능 (검증셋)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "rf_evaluation.png", dpi=120)
    print("\n저장: outputs/figures/rf_evaluation.png, outputs/tables/rf_metrics.csv")


if __name__ == "__main__":
    main()
