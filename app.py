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
# DB 다운로드(임시파일) + SQL 유틸
# - cache_data가 sqlite conn을 hash 못해서,
#   매번 temp db로 연결해서 읽고 닫는 방식(안정적)
# =========================
def _download_db_to_temp(db_url: str) -> str:
    r = requests.get(db_url, timeout=30)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.write(r.content)
    tmp.flush()
    tmp.close()
    return tmp.name


@st.cache_data(ttl=3600)
def load_all_diseases(db_url: str) -> list[str]:
    tmp_path = _download_db_to_temp(db_url)
    c = sqlite3.connect(tmp_path)
    try:
        df = pd.read_sql("SELECT DISTINCT disease FROM uw_rows ORDER BY disease", c)
        return df["disease"].dropna().astype(str).tolist()
    finally:
        c.close()
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@st.cache_data(ttl=3600)
def load_criteria_for_disease(db_url: str, disease: str) -> list[str]:
    tmp_path = _download_db_to_temp(db_url)
    c = sqlite3.connect(tmp_path)
    try:
        df = pd.read_sql(
            """
            SELECT DISTINCT criteria
            FROM uw_rows
            WHERE disease = ?
              AND criteria IS NOT NULL
              AND TRIM(criteria) <> ''
            ORDER BY criteria
            """,
            c,
            params=(disease,),
        )
        return df["criteria"].dropna().astype(str).tolist()
    finally:
        c.close()
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@st.cache_data(ttl=3600)
def load_benefit_decisions(db_url: str, disease: str, criteria: str) -> pd.DataFrame:
    tmp_path = _download_db_to_temp(db_url)
    c = sqlite3.connect(tmp_path)
    try:
        df = pd.read_sql(
            """
            SELECT benefit, decision
            FROM uw_rows
            WHERE disease = ? AND criteria = ?
            ORDER BY benefit
            """,
            c,
            params=(disease, criteria),
        )
        return df
    finally:
        c.close()
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# =========================
# 검색/추천 (정규화 + 부분포함 보너스)
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


def recommend_diseases(query: str, diseases: list[str], top_k: int = 10) -> pd.DataFrame:
    q = (query or "").strip()
    if not q:
        return pd.DataFrame(columns=["질병명", "일치율(%)"])

    scored = [(d, similarity(q, d)) for d in diseases]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    df = pd.DataFrame(
        {
            "질병명": [d for d, _ in top],
            "일치율(%)": [(s * 100) for _, s in top],
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

# 세션키: selectbox 제어용
if "disease_selectbox" not in st.session_state:
    st.session_state["disease_selectbox"] = diseases[0]
if "criteria_selectbox" not in st.session_state:
    st.session_state["criteria_selectbox"] = None
if "rec_selected_disease" not in st.session_state:
    st.session_state["rec_selected_disease"] = None


# -------------------------
# (1) 질병명 검색/추천
# -------------------------
st.subheader("질병명 검색/추천")

query = st.text_input("질병명을 입력하세요 (예: 척추염, 당뇨, 객혈 등)", value="")
top_k = st.slider("추천 개수", min_value=3, max_value=20, value=10, step=1)

if query.strip():
    rec_df = recommend_diseases(query, diseases, top_k=top_k)

    if rec_df.empty or rec_df["일치율(%)"].max() <= 0:
        st.info("추천 결과가 없습니다. 다른 키워드로 시도해 보세요.")
    else:
        show_df = rec_df[["질병명", "일치율(%)"]].copy()
        show_df.insert(0, "선택", False)

        # 기존 선택 복원
        if st.session_state["rec_selected_disease"]:
            show_df.loc[
                show_df["질병명"] == st.session_state["rec_selected_disease"], "선택"
            ] = True

        edited = st.data_editor(
            show_df,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "선택": st.column_config.CheckboxColumn(
                    "선택",
                    help="체크하면 아래 질병 선택이 즉시 변경됩니다.",
                ),
                "질병명": st.column_config.TextColumn(width="large"),
                "일치율(%)": st.column_config.NumberColumn(format="%.1f"),
            },
            disabled=["질병명", "일치율(%)"],
            key="rec_table",
        )

        chosen = edited[edited["선택"] == True]
        if len(chosen) > 0:
            new_choice = chosen.iloc[0]["질병명"]

            # 여러 개 체크했더라도 첫 번째만 인정
            if new_choice != st.session_state["rec_selected_disease"]:
                st.session_state["rec_selected_disease"] = new_choice
                st.session_state["disease_selectbox"] = new_choice
                st.session_state["criteria_selectbox"] = None
                st.rerun()

st.divider()


# -------------------------
# (2) 질병/심사기준 선택 + 결과 조회
# -------------------------
st.subheader("질병 선택/조회")

disease = st.selectbox("질병 선택", diseases, key="disease_selectbox")

criteria_list = []
if disease:
    criteria_list = load_criteria_for_disease(DB_URL, disease)

if not criteria_list:
    st.info("선택된 질병에 대한 심사기준이 없습니다.")
    st.stop()

# criteria 세션 보정 (질병 바뀔 때 None이거나 목록에 없으면 첫 값으로)
if (
    st.session_state["criteria_selectbox"] is None
    or st.session_state["criteria_selectbox"] not in criteria_list
):
    st.session_state["criteria_selectbox"] = criteria_list[0]

crit = st.selectbox("심사기준 선택", criteria_list, key="criteria_selectbox")

df = load_benefit_decisions(DB_URL, disease, crit)

st.subheader("급부별 인수 결과")
st.dataframe(df, use_container_width=True)
