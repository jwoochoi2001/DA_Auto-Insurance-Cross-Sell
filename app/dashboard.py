# -*- coding: utf-8 -*-
"""
자동차보험 교차판매 - 영업 운영 대시보드 (Streamlit)

실행:
    streamlit run app/dashboard.py

구성
- 상단: 주요 지표(KPI)
- 탭1 운영 현황     : 등급 분포 / 모델 성능 / 접촉 진행 현황
- 탭2 변수별 관심도 : 변수별 가입 관심율(Response=1 비율) 분석
- 탭3 영업 대상 리스트: 필터 / 접촉상태·메모 편집 / 저장
- 탭4 신규 고객 추가 : 입력 -> XGBoost 관심 확률 + 5단계 등급 자동 산출 후 리스트에 추가

전제: analysis/build_dashboard_assets.py 를 먼저 실행해 자산 생성.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import joblib
import altair as alt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "analysis"))
from features import (  # noqa: E402
    build_model_frame, probability_to_tier, MODEL_FEATURES,
    VEHICLE_AGE_MAP, VEHICLE_AGE_INV, PREV_INSURED_LABEL, TIER_LABELS,
)

MODEL_DIR = ROOT / "outputs" / "model"
DASH_DIR = ROOT / "outputs" / "dashboard"
CUST_PATH = ROOT / "data" / "processed" / "sales_customers.csv"

TIER_ORDER = TIER_LABELS[::-1]  # 매우 높음 -> 매우 낮음
TIER_COLOR = {
    "매우 높음": "#1a9850", "높음": "#91cf60", "보통": "#fee08b",
    "낮음": "#fc8d59", "매우 낮음": "#d73027",
}
PRIORITY_TIERS = ["매우 높음", "높음"]
STATUS_OPTIONS = ["미접촉", "통화시도", "접촉완료", "가입", "거절", "부재중"]

st.set_page_config(page_title="자동차보험 영업 대시보드", page_icon="📞", layout="wide")


# ---------------------------------------------------------------- 로더
@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "xgb_pipeline.joblib")


@st.cache_data
def load_meta():
    return json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))


@st.cache_data
def load_variable_analysis():
    return pd.read_csv(DASH_DIR / "variable_analysis.csv")


@st.cache_data
def load_feature_importance():
    return pd.read_csv(DASH_DIR / "feature_importance.csv")


@st.cache_data
def load_customers(mtime: float) -> pd.DataFrame:
    return pd.read_csv(CUST_PATH)


def read_customers() -> pd.DataFrame:
    df = load_customers(CUST_PATH.stat().st_mtime).copy()
    for col in ("메모", "접촉상태", "차량연식", "기존_자동차보험", "등급",
                "Gender", "Vehicle_Damage", "Region_Code", "Policy_Sales_Channel"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


def save_customers(df: pd.DataFrame):
    df.to_csv(CUST_PATH, index=False, encoding="utf-8")
    load_customers.clear()


def score(raw: dict, model, meta) -> tuple[float, str]:
    frame = build_model_frame(pd.DataFrame([raw]), meta["top_channels"])
    prob = float(model.predict_proba(frame[MODEL_FEATURES])[:, 1][0])
    tier = probability_to_tier(prob, meta["tier_cutpoints_prob"])
    return prob, tier


# ---------------------------------------------------------------- 준비
if not (MODEL_DIR / "xgb_pipeline.joblib").exists():
    st.error("자산이 없습니다. 먼저 `python analysis/build_dashboard_assets.py` 를 실행하세요.")
    st.stop()

model = load_model()
meta = load_meta()
cust = read_customers()
thr = meta["operating_threshold"]
cut = meta["tier_cutpoints_prob"]
tm = meta["test_metrics"]

# ---------------------------------------------------------------- 헤더 + KPI
cv = meta.get("cv_roc_auc_train", {})
thr_profit = meta.get("threshold_profit_max", thr)

st.title("📞 자동차보험 교차판매 영업 대시보드")
st.caption(
    f"모델: XGBoost (트리 {meta['params']['trees_used']}그루, depth {meta['params']['max_depth']}, "
    f"lr {meta['params']['learning_rate']}) · 5겹 CV ROC-AUC "
    f"{cv.get('평균', float('nan')):.3f}±{cv.get('표준편차', float('nan')):.3f} · "
    f"운영 임계값 {thr:.2f}(F1) / 이익최대 {thr_profit:.2f} · 등급 컷포인트(확률) {cut}"
)

n_total = len(cust)
n_priority = int(cust["등급"].isin(PRIORITY_TIERS).sum())
mean_p = cust["관심확률(%)"].mean()
n_done = int(cust["접촉상태"].isin(["접촉완료", "가입", "거절"]).sum())
n_join = int((cust["접촉상태"] == "가입").sum())
exp_interested = int(round((cust["관심확률(%)"] / 100).sum()))

k = st.columns(6)
k[0].metric("영업 대상 총원", f"{n_total:,}명")
k[1].metric("우선 대상 (높음+매우높음)", f"{n_priority:,}명", f"{n_priority/n_total*100:.0f}%")
k[2].metric("평균 관심 확률", f"{mean_p:.1f}%")
k[3].metric("리스트 기대 관심고객 수", f"{exp_interested:,}명",
           help="Σ(개별 관심확률). 이 리스트에 전부 전화 시 관심 고객 기대치")
k[4].metric("접촉 완료", f"{n_done:,}명", f"{(n_done/n_total*100):.0f}%")
k[5].metric("가입 성사", f"{n_join:,}건")

k2 = st.columns(6)
k2[0].metric("모델 ROC-AUC (테스트)", f"{tm['ROC_AUC']:.3f}")
k2[1].metric("예상 통화 성공률", f"{tm['정밀도']*100:.0f}%",
            help=f"임계값 {thr:.2f}에서의 정밀도. 무작위 전화는 {meta['base_rate']*100:.0f}%")
k2[2].metric("관심고객 도달률", f"{tm['재현율']*100:.0f}%",
            help="임계값 이상으로 잡아내는 실제 관심고객 비율(재현율)")
k2[3].metric("리프트", f"{tm['정밀도']/meta['base_rate']:.1f}배",
            help="무작위 대비 통화 성공률 배수")
k2[4].metric("F1 (테스트)", f"{tm['F1']:.3f}")
k2[5].metric("기준 관심율", f"{meta['base_rate']*100:.1f}%")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 운영 현황", "📈 변수별 관심도 분석", "📋 영업 대상 리스트", "➕ 신규 고객 추가"]
)

# ================================================================ 탭1
with tab1:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("영업 대상 등급 분포")
        dist = (cust["등급"].value_counts()
                .reindex(TIER_ORDER).fillna(0).astype(int).reset_index())
        dist.columns = ["등급", "고객수"]
        chart = (alt.Chart(dist).mark_bar().encode(
            x=alt.X("고객수:Q"),
            y=alt.Y("등급:N", sort=TIER_ORDER),
            color=alt.Color("등급:N",
                            scale=alt.Scale(domain=list(TIER_COLOR), range=list(TIER_COLOR.values())),
                            legend=None),
            tooltip=["등급", "고객수"],
        ).properties(height=220))
        st.altair_chart(chart, width="stretch")

        st.subheader("등급별 의미 (검증셋 기준)")
        ts = pd.DataFrame(meta["tier_stats_valid"]).T.reset_index()
        ts.columns = ["등급", "고객비율(%)", "평균 관심확률(%)", "실제 관심율(%)"]
        ts["등급"] = pd.Categorical(ts["등급"], categories=TIER_ORDER, ordered=True)
        st.dataframe(ts.sort_values("등급"), hide_index=True, width="stretch")
        st.caption(
            "예측 확률(평균)과 실제 관심율이 거의 일치하므로 모델 확률이 잘 보정되어 있습니다. "
            "‘매우 높음/높음’ 고객부터 전화하면 통화 성공률이 크게 오릅니다."
        )

    with c2:
        st.subheader("접촉 진행 현황")
        sc = cust["접촉상태"].value_counts().reindex(STATUS_OPTIONS).fillna(0).astype(int).reset_index()
        sc.columns = ["접촉상태", "건수"]
        st.altair_chart(
            alt.Chart(sc).mark_bar(color="#4C72B0").encode(
                x="건수:Q", y=alt.Y("접촉상태:N", sort=STATUS_OPTIONS), tooltip=["접촉상태", "건수"]
            ).properties(height=220),
            width="stretch",
        )

        st.subheader("모델 성능 (최종 테스트셋)")
        mdf = pd.DataFrame({
            "지표": ["ROC-AUC", "정확도", "정밀도(통화성공률)", "재현율(도달률)", "F1"],
            "값": [tm["ROC_AUC"], tm["정확도"], tm["정밀도"], tm["재현율"], tm["F1"]],
        })
        st.altair_chart(
            alt.Chart(mdf).mark_bar(color="#2E7D32").encode(
                x=alt.X("값:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("지표:N", sort=None), tooltip=["지표", "값"],
            ).properties(height=200),
            width="stretch",
        )
        st.caption(
            f"임계값 {thr:.2f}에서 전화 대상의 약 {tm['정밀도']*100:.0f}%가 실제 관심고객이며, "
            f"전체 관심고객의 {tm['재현율']*100:.0f}%에 도달합니다. "
            f"(5겹 교차검증 ROC-AUC {cv.get('평균', float('nan')):.3f} ± {cv.get('표준편차', float('nan')):.3f})"
        )
        pa = meta.get("profit_assumptions", {})
        if pa:
            st.info(
                f"**임계값 선택**: 현재 운영값은 F1 최대({thr:.2f}). "
                f"통화비용 {pa['통화비용_원']:,}원 / 관심리드 가치 {pa['관심리드_가치_원']:,}원 가정 시 "
                f"기대이익 최대 임계값은 {thr_profit:.2f} "
                f"(검증셋 기대이익 {pa['valid_기대이익_원']:,}원). "
                "실제 단가로 두 값을 바꿔 운영 기준을 정하세요."
            )

# ================================================================ 탭2
with tab2:
    st.subheader("변수 중요도 (모델이 무엇을 보고 판단하는가)")
    fi = load_feature_importance()
    VAR_KO = {
        "Previously_Insured": "기존 자동차보험", "Vehicle_Damage": "차량 손상 이력",
        "Age": "나이", "Policy_Sales_Channel": "판매채널",
        "Vehicle_Age_ord": "차량 연식", "Region_Code": "지역코드",
        "Annual_Premium": "연간보험료", "Gender": "성별", "Vintage": "관계기간",
    }
    fi_disp = fi.assign(변수명=fi["변수"].map(VAR_KO))
    ci1, ci2 = st.columns(2)
    with ci1:
        st.caption("순열 중요도: 그 변수를 무작위로 섞으면 검증 ROC-AUC가 얼마나 떨어지는가 (클수록 중요)")
        st.altair_chart(
            alt.Chart(fi_disp).mark_bar(color="#4C72B0").encode(
                x=alt.X("순열중요도:Q", title="ROC-AUC 감소폭"),
                y=alt.Y("변수명:N", sort="-x"),
                tooltip=["변수명", "순열중요도", "순열중요도_표준편차"],
            ).properties(height=300),
            width="stretch",
        )
    with ci2:
        st.caption("XGBoost gain 중요도: 트리 분기에서 그 변수가 기여한 정보량 비중(%)")
        st.altair_chart(
            alt.Chart(fi_disp).mark_bar(color="#2E7D32").encode(
                x=alt.X("gain중요도(%):Q"),
                y=alt.Y("변수명:N", sort="-x"),
                tooltip=["변수명", "gain중요도(%)"],
            ).properties(height=300),
            width="stretch",
        )
    st.caption(
        "**기존 자동차보험 보유 여부**와 **차량 손상 이력** 두 변수가 예측력의 대부분을 차지합니다. "
        "판매채널·지역·나이가 그 다음이며, 성별·연간보험료·관계기간은 기여가 거의 없습니다."
    )
    st.divider()

    st.subheader("변수별 가입 관심율 (Response=1 비율, 학습셋 기준)")
    va = load_variable_analysis()
    base = meta["base_rate"] * 100

    left, right = st.columns([1, 3])
    with left:
        var = st.selectbox("변수 선택", va["변수"].unique())
        st.metric("전체 평균 관심율", f"{base:.1f}%")
        st.caption("막대가 평균선(빨강)보다 길면 그 구간 고객의 가입 관심이 평균보다 높다는 뜻입니다.")
    with right:
        sub = va[va["변수"] == var].copy()
        order = sub["구간"].tolist()
        bars = alt.Chart(sub).mark_bar().encode(
            x=alt.X("구간:N", sort=order),
            y=alt.Y("관심율(%):Q"),
            color=alt.condition(alt.datum["관심율(%)"] >= base,
                                alt.value("#1a9850"), alt.value("#fc8d59")),
            tooltip=["구간", "고객수", "관심율(%)", "전체평균대비"],
        )
        rule = alt.Chart(pd.DataFrame({"y": [base]})).mark_rule(
            color="red", strokeDash=[4, 4]).encode(y="y:Q")
        st.altair_chart((bars + rule).properties(height=380), width="stretch")

    st.dataframe(
        va[va["변수"] == var][["구간", "고객수", "관심율(%)", "전체평균대비"]],
        hide_index=True, width="stretch",
    )
    with st.expander("해석 도움말"):
        st.markdown(
            "- **차량 손상 이력 = Yes**, **기존 자동차보험 = 미가입**, **차량 연식 > 2년**, "
            "**나이 31-50세** 구간에서 관심율이 크게 높습니다.\n"
            "- **기존 자동차보험 = 기가입** 고객은 관심율이 거의 0이라 영업 대상에서 제외해도 무방합니다.\n"
            "- 관계기간(Vintage)·연간보험료는 관심율 차이가 작아 예측 기여도가 낮습니다."
        )

# ================================================================ 탭3
with tab3:
    st.subheader("영업 대상 리스트")
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1])
    tier_sel = f1.multiselect("등급", TIER_ORDER, default=PRIORITY_TIERS)
    stat_sel = f2.multiselect("접촉상태", STATUS_OPTIONS, default=["미접촉"])
    pmin = f3.slider("최소 관심확률(%)", 0, 100, 0, 5)
    only_priority = f4.checkbox("우선 대상만", value=False)

    view = cust.copy()
    if tier_sel:
        view = view[view["등급"].isin(tier_sel)]
    if stat_sel:
        view = view[view["접촉상태"].isin(stat_sel)]
    view = view[view["관심확률(%)"] >= pmin]
    if only_priority:
        view = view[view["등급"].isin(PRIORITY_TIERS)]
    view = view.sort_values("관심확률(%)", ascending=False)

    st.caption(f"필터 결과 {len(view):,}명 / 전체 {len(cust):,}명. 접촉상태·메모는 표에서 바로 수정 후 아래 버튼으로 저장하세요.")

    show_cols = ["id", "등급", "관심확률(%)", "접촉상태", "메모", "Gender", "Age",
                 "기존_자동차보험", "차량연식", "Vehicle_Damage", "Annual_Premium",
                 "Region_Code", "Policy_Sales_Channel", "Vintage"]
    edited = st.data_editor(
        view[show_cols],
        hide_index=True, width="stretch", height=440,
        column_config={
            "관심확률(%)": st.column_config.NumberColumn(disabled=True, format="%.1f"),
            "등급": st.column_config.TextColumn(disabled=True),
            "접촉상태": st.column_config.SelectboxColumn(options=STATUS_OPTIONS, required=True),
            "메모": st.column_config.TextColumn(width="medium"),
            **{c: st.column_config.Column(disabled=True) for c in
               ["id", "Gender", "Age", "기존_자동차보험", "차량연식", "Vehicle_Damage",
                "Annual_Premium", "Region_Code", "Policy_Sales_Channel", "Vintage"]},
        },
        key="editor",
    )

    cbtn1, cbtn2 = st.columns([1, 4])
    if cbtn1.button("💾 변경사항 저장", type="primary"):
        upd = cust.set_index("id")
        e = edited.set_index("id")
        upd.loc[e.index, "접촉상태"] = e["접촉상태"]
        upd.loc[e.index, "메모"] = e["메모"].fillna("")
        save_customers(upd.reset_index())
        st.success(f"{len(e)}건 저장 완료")
        st.rerun()

    csv = view[show_cols].to_csv(index=False).encode("utf-8-sig")
    cbtn2.download_button("⬇️ 현재 목록 CSV 내려받기", csv, "sales_call_list.csv", "text/csv")

# ================================================================ 탭4
with tab4:
    st.subheader("신규 고객 추가: 관심 확률·등급 자동 산출")
    opt = meta["input_options"]
    top_ch = meta["top_channels"]

    with st.form("new_customer"):
        c = st.columns(3)
        gender = c[0].selectbox("성별 Gender", opt["Gender"])
        age = c[1].number_input("나이 Age", opt["Age_range"][0], opt["Age_range"][1], 35)
        region = c[2].selectbox("지역코드 Region_Code", list(range(opt["Region_Code_range"][0],
                                                                opt["Region_Code_range"][1] + 1)),
                                index=28)

        c = st.columns(3)
        prev = c[0].selectbox("기존 자동차보험 Previously_Insured", opt["Previously_Insured"],
                              format_func=lambda v: f"{v} ({PREV_INSURED_LABEL[v]})")
        vage = c[1].selectbox("차량 연식 Vehicle_Age", opt["Vehicle_Age"])
        vdam = c[2].selectbox("차량 손상 이력 Vehicle_Damage", opt["Vehicle_Damage"])

        c = st.columns(3)
        premium = c[0].number_input("연간보험료 Annual_Premium",
                                    float(opt["Annual_Premium_range"][0]),
                                    float(opt["Annual_Premium_range"][1]), 30000.0, step=500.0)
        channel = c[1].selectbox("판매채널 Policy_Sales_Channel",
                                 top_ch + ["CHOTHER"],
                                 format_func=lambda s: s.replace("CH", "채널 ").replace("채널 OTHER", "기타"))
        vintage = c[2].number_input("관계기간(일) Vintage", opt["Vintage_range"][0],
                                    opt["Vintage_range"][1], 150)

        submitted = st.form_submit_button("관심 확률 예측 후 리스트에 추가", type="primary")

    if submitted:
        raw = {
            "Gender": gender, "Age": int(age), "Region_Code": int(region),
            "Previously_Insured": int(prev), "Vehicle_Age": vage,
            "Vehicle_Damage": vdam, "Annual_Premium": float(premium),
            "Policy_Sales_Channel": channel, "Vintage": int(vintage),
        }
        prob, tier = score(raw, model, meta)

        m = st.columns(3)
        m[0].metric("관심 확률", f"{prob*100:.1f}%")
        m[1].metric("5단계 등급", tier)
        act = "우선 전화 대상" if tier in PRIORITY_TIERS else (
            "여력 시 전화" if tier == "보통" else "전화 비권장")
        m[2].metric("권고 조치", act)
        st.progress(min(prob / max(cut[-1] * 1.3, 0.4), 1.0))
        if prob >= thr:
            st.success(f"운영 임계값 {thr:.2f} 이상이므로 영업 전화를 권장합니다")
        else:
            st.info(f"운영 임계값 {thr:.2f} 미만이므로 우선순위가 낮습니다")

        new_id = int(cust["id"].max()) + 1
        row = {
            "id": new_id, "Gender": gender, "Age": int(age),
            "Region_Code": f"R{int(region)}", "Previously_Insured": int(prev),
            "Vehicle_Age_ord": VEHICLE_AGE_MAP[vage], "Vehicle_Damage": vdam,
            "Annual_Premium": float(premium),
            "Policy_Sales_Channel": channel if channel in top_ch else "CHOTHER",
            "Vintage": int(vintage), "등급": tier,
            "관심확률(%)": round(prob * 100, 1), "접촉상태": "미접촉",
            "차량연식": vage, "기존_자동차보험": PREV_INSURED_LABEL[int(prev)], "메모": "신규 등록",
        }
        save_customers(pd.concat([cust, pd.DataFrame([row])], ignore_index=True))
        st.success(f"고객 #{new_id} 리스트에 추가 완료 (등급: {tier}). "
                   "‘영업 대상 리스트’ 탭에서 확인하세요. (상단 KPI는 다음 화면 갱신 시 반영)")
