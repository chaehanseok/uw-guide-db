import os
import re
import sqlite3
import requests
import tempfile
from email.utils import parsedate_to_datetime
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st


# =========================
# 설정
# =========================
DB_URL = "https://raw.githubusercontent.com/chaehanseok/uw-guide-db/main/uw_knowledge.db"
MAX_RECOMMENDATIONS = 30  # 추천 표 최대 표시 수


# =========================
# 유틸: GitHub Last-Modified로 기준일(asof) 추정
# =========================
@st.cache_data(ttl=3600)
def get_db_asof_from_github(db_url: str) -> str:
    try:
        r = requests.head(db_url, timeout=10)
        lm = r.headers.get("Last-Modified")
        if not lm:
            return "기준일 미상"
        dt = parsedate_to_datetime(lm)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return "기준일 미상"


# =========================
# DB 다운로드 -> temp path
# (cache_data는 sqlite conn을 인자로 받으면 Unhashable 에러 나므로 path 기반으로 처리)
# =========================
@st.cache_data(ttl=3600)
def download_db_to_temp(db_url: str) -> str:
    r = requests.get(db_url, timeout=30)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.write(r.content)
    tmp.flush()
    tmp.close()
    return tmp.name


def _query_df(db_path: str, sql: str, params: tuple = ()) -> pd.DataFrame:
    c = sqlite3.connect(db_path)
    try:
        return pd.read_sql(sql, c, params=params)
    finally:
        c.close()


@st.cache_data(ttl=3600)
def load_all_diseases(db_url: str) -> list[str]:
    db_path = download_db_to_temp(db_url)
    df = _query_df(db_path, "SELECT DISTINCT disease FROM uw_rows ORDER BY disease")
    return df["disease"].dropna().astype(str).tolist()


@st.cache_data(ttl=3600)
def load_criteria_for_disease(db_url: str, disease: str) -> list[str]:
    db_path = download_db_to_temp(db_url)
    df = _query_df(
        db_path,
        """
        SELECT DISTINCT criteria
        FROM uw_rows
        WHERE disease = ?
          AND criteria IS NOT NULL
          AND TRIM(criteria) <> ''
        ORDER BY criteria
        """,
        params=(disease,),
    )
    return df["criteria"].dropna().astype(str).tolist()


@st.cache_data(ttl=3600)
def load_benefit_decisions(db_url: str, disease: str, criteria: str) -> pd.DataFrame:
    db_path = download_db_to_temp(db_url)
    df = _query_df(
        db_path,
        """
        SELECT benefit, decision
        FROM uw_rows
        WHERE disease = ? AND criteria = ?
        ORDER BY benefit
        """,
        params=(disease, criteria),
    )
    return df


# =========================
# 검색/추천
# =========================
_norm_keep_kor_eng_num = re.compile(r"[^0-9a-zA-Z가-힣]+")

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = _norm_keep_kor_eng_num.sub("", s)
    return s


def similarity(query_raw: str, disease_raw: str) -> float:
    qn = normalize_text(query_raw)
    dn = normalize_text(disease_raw)
    if not qn or not dn:
        return 0.0

    base = SequenceMatcher(None, qn, dn).ratio()

    if qn == dn:
        base = max(base, 0.999)
    elif qn in dn:
        base = min(1.0, base + 0.18)
    elif dn in qn:
        base = min(1.0, base + 0.10)

    return base


def recommend_diseases(query: str, diseases: list[str]) -> pd.DataFrame:
    q = (query or "").strip()
    if not q:
        return pd.DataFrame(columns=["질병명", "일치율(%)"])

    scored = [(d, similarity(q, d)) for d in diseases]
    scored.sort(key=lambda x: x[1], reverse=True)

    df = pd.DataFrame(
        {
            "질병명": [d for d, _ in scored],
            "일치율(%)": [s * 100 for _, s in scored],
        }
    )
    df["일치율(%)"] = df["일치율(%)"].round(1)
    return df


# =========================
# UI
# =========================
st.set_page_config(page_title="질병 심사 가이드", layout="wide")

asof_yyyymmdd = get_db_asof_from_github(DB_URL)

st.title("질병 심사 가이드\n(Underwriting Guide)")
st.warning(
    "본 인수기준은 내부 교육용입니다. "
    f"({asof_yyyymmdd}, LoveAge Plan 질병심사메뉴얼 등록기준).\n"
    "변동 사항이 있을 수 있으며 실제 인수기준은 반드시 확인후 고객에게 안내 바랍니다."
)

diseases = load_all_diseases(DB_URL)
if not diseases:
    st.error("DB에서 질병 목록을 불러오지 못했습니다.")
    st.stop()

# 세션 상태 초기화
if "disease_selectbox" not in st.session_state:
    st.session_state["disease_selectbox"] = diseases[0]
if "criteria_selectbox" not in st.session_state:
    st.session_state["criteria_selectbox"] = None
if "rec_selected_disease" not in st.session_state:
    st.session_state["rec_selected_disease"] = None


# -------------------------
# 1) 질병명 검색/추천 (체크박스 클릭 시 즉시 적용)
# -------------------------
st.subheader("질병명 검색/추천")

query = st.text_input("질병명을 입력하세요 (예: 척추염, 당뇨, 객혈 등)", value="")

min_match = st.slider(
    "최소 일치율(%)",
    min_value=0,
    max_value=100,
    value=70,
    step=1,
    help="이 값 이상인 추천만 표시합니다.",
)

if query.strip():
    rec_df = recommend_diseases(query, diseases)
    rec_df = rec_df[rec_df["일치율(%)"] >= float(min_match)].copy()
    rec_df = rec_df.head(MAX_RECOMMENDATIONS)

    if rec_df.empty:
        st.info("조건에 맞는 추천 결과가 없습니다. 최소 일치율을 낮추거나 다른 키워드로 시도해 보세요.")
    else:
        show_df = rec_df[["질병명", "일치율(%)"]].copy()
        show_df.insert(0, "선택", False)

        # 기존 선택 반영
        if st.session_state["rec_selected_disease"]:
            show_df.loc[show_df["질병명"] == st.session_state["rec_selected_disease"], "선택"] = True

        edited = st.data_editor(
            show_df,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "선택": st.column_config.CheckboxColumn("선택", help="체크하면 아래 질병 선택이 즉시 변경됩니다."),
                "질병명": st.column_config.TextColumn(width="large"),
                "일치율(%)": st.column_config.NumberColumn(format="%.1f"),
            },
            # ✅ 체크박스만 편집 가능하게: 컬럼별 disabled 정확히 지정
            disabled={"질병명": True, "일치율(%)": True},
            key="rec_table",
        )

        # ✅ 단일 선택 강제 + 즉시 반영
        chosen = edited[edited["선택"] == True]
        if len(chosen) > 0:
            new_choice = chosen.iloc[0]["질병명"]

            if new_choice != st.session_state["rec_selected_disease"]:
                st.session_state["rec_selected_disease"] = new_choice
                st.session_state["disease_selectbox"] = new_choice
                st.session_state["criteria_selectbox"] = None
                st.rerun()

st.divider()


# -------------------------
# 2) 질병/심사기준 선택 + 결과
# -------------------------
st.subheader("질병 선택/조회")

disease = st.selectbox("질병 선택", diseases, key="disease_selectbox")

criteria_list = load_criteria_for_disease(DB_URL, disease)
if not criteria_list:
    st.info("선택된 질병에 대한 심사기준이 없습니다.")
    st.stop()

# criteria 세션 보정 (질병이 바뀌면 첫 criteria로 자동 세팅)
if (st.session_state["criteria_selectbox"] is None) or (st.session_state["criteria_selectbox"] not in criteria_list):
    st.session_state["criteria_selectbox"] = criteria_list[0]

crit = st.selectbox("심사기준 선택", criteria_list, key="criteria_selectbox")

df = load_benefit_decisions(DB_URL, disease, crit)

st.subheader("급부별 인수 결과")
st.dataframe(df, use_container_width=True)
