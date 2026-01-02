import re
import sqlite3
import requests
import tempfile
from email.utils import parsedate_to_datetime

import pandas as pd
import streamlit as st


DB_URL = "https://raw.githubusercontent.com/chaehanseok/uw-guide-db/main/uw_knowledge.db"


@st.cache_data(ttl=3600)
def get_db_asof_from_github(db_url: str) -> str:
    try:
        r = requests.head(db_url, timeout=10)
        lm = r.headers.get("Last-Modified")
        if not lm:
            return "2025-12-31"
        dt = parsedate_to_datetime(lm)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return "2025-12-31"


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
def _get_uw_rows_columns(db_url: str) -> set[str]:
    """uw_rows 테이블 컬럼 목록을 set으로 반환"""
    db_path = download_db_to_temp(db_url)
    c = sqlite3.connect(db_path)
    try:
        cols_df = pd.read_sql("PRAGMA table_info(uw_rows)", c)
        return set(cols_df["name"].astype(str).tolist())
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


@st.cache_data(ttl=3600)
def load_common_info_for_disease(db_url: str, disease: str) -> dict:
    """
    질병별 공통항목(필요서류/진단/TIP)을 criteria와 무관하게 조회.
    - DB 컬럼명이 한글/영문 어느 쪽이든 동작하도록 후보를 둔다.
    """
    cols = _get_uw_rows_columns(db_url)

    # 표시 라벨 -> (DB 컬럼 후보들)
    col_candidates = {
        "필요서류": ["필요서류", "need_doc", "need_docs", "required_docs", "documents"],
        "진단": ["진단", "diagnosis", "dx"],
        "TIP": ["TIP", "tip", "tips", "memo", "note", "비고", "유의사항"],
    }

    # 실제 존재하는 컬럼 매핑
    col_map = {}
    for label, cands in col_candidates.items():
        for c in cands:
            if c in cols:
                col_map[label] = c
                break

    if not col_map:
        return {}

    select_exprs = []
    for label, col in col_map.items():
        # 컬럼명은 항상 안전하게 quoting
        select_exprs.append(f'TRIM(COALESCE("{col}", \'\')) AS "{label}"')

    sql = f"""
    SELECT DISTINCT
        {", ".join(select_exprs)}
    FROM uw_rows
    WHERE disease = ?
    """

    db_path = download_db_to_temp(db_url)
    df = _query_df(db_path, sql, params=(disease,))

    info = {}
    for label in ["필요서류", "진단", "TIP"]:
        if label not in df.columns:
            continue
        vals = (
            df[label]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda x: x != ""]
            .unique()
            .tolist()
        )
        if vals:
            info[label] = vals

    return info

# =========================
# UI
# =========================
st.set_page_config(page_title="질병 심사 가이드", layout="wide")

asof_yyyymmdd = get_db_asof_from_github(DB_URL)

st.title("질병 심사 가이드\n(Underwriting Guide)")

msg = f"""
이 자료는 미래에셋생명 LoveAge Plan 질병심사메뉴얼을
({asof_yyyymmdd}) 기준으로 수집한 자료입니다.<br>
본 자료는 미래에셋금융서비스 구성원 대상 내부 교육자료이며,
질병별 인수기준은 수시로 변경될 수 있습니다.<br>
고객 안내 및 청약 전에는 반드시 미래에셋생명에
최신 인수기준을 직접 확인하시기 바랍니다.
"""

with st.warning(""):
    st.markdown(msg, unsafe_allow_html=True)

# -------------------------
# DB 로드
# -------------------------
diseases = load_all_diseases(DB_URL)
if not diseases:
    st.error("DB에서 질병 목록을 불러오지 못했습니다.")
    st.stop()

# 세션 상태
if "disease_selectbox" not in st.session_state:
    st.session_state["disease_selectbox"] = ""   # 초기값 공란
if "criteria_selectbox" not in st.session_state:
    st.session_state["criteria_selectbox"] = None

st.divider()

# -------------------------
# 질병/심사기준 선택
# -------------------------
st.subheader("질병 선택/조회")

disease_options = [""] + diseases

# ✅ options에 disease_options를 넣어야 공란이 적용됨
disease = st.selectbox(
    "질병 선택",
    disease_options,
    key="disease_selectbox",
)

if not disease:
    st.info("질병명을 입력하거나 선택해 주세요.")
    st.stop()

# -------------------------
# ✅ 질병 공통 정보(필요서류/진단/TIP) 표시 (criteria와 무관)
# -------------------------
common_info = load_common_info_for_disease(DB_URL, disease)

# (디버그가 필요할 때만 잠깐)
# st.write("uw_rows columns:", sorted(list(_get_uw_rows_columns(DB_URL))))
# st.write("common_info:", common_info)

if common_info:
    st.subheader("질병 공통 안내")

    for label in ["필요서류", "진단", "TIP"]:
        vals = common_info.get(label)
        if not vals:
            continue

        with st.expander(label, expanded=True):
            if len(vals) == 1:
                st.write(vals[0])
            else:
                for v in vals:
                    st.write(f"- {v}")

    st.divider()

criteria_list = load_criteria_for_disease(DB_URL, disease)
if not criteria_list:
    st.info("선택된 질병에 대한 심사기준이 없습니다.")
    st.stop()

if (st.session_state["criteria_selectbox"] is None) or (st.session_state["criteria_selectbox"] not in criteria_list):
    st.session_state["criteria_selectbox"] = criteria_list[0]

crit = st.selectbox("심사기준 선택", criteria_list, key="criteria_selectbox")

df = load_benefit_decisions(DB_URL, disease, crit).copy()
df["decision"] = df["decision"].fillna("").astype(str).str.strip()
df["benefit"] = df["benefit"].fillna("").astype(str).str.strip()

# -------------------------
# 인수결과값 요약 + 필터링
# -------------------------
st.subheader("급부별 인수 결과")

counts = df["decision"].replace("", "(빈값)").value_counts().reset_index()
counts.columns = ["인수결과값", "건수"]

c1, c2 = st.columns([1, 2])
with c1:
    st.metric("총 급부 수", int(len(df)))
with c2:
    st.dataframe(counts, hide_index=True, use_container_width=True)

all_decisions = counts["인수결과값"].tolist()
selected = st.multiselect(
    "인수결과값 필터 (선택한 값만 아래 표에 표시)",
    options=all_decisions,
    default=all_decisions,
)

if not selected:
    st.warning("필터가 모두 해제되어 표시할 데이터가 없습니다.")
    st.dataframe(df.iloc[0:0], use_container_width=True)
else:
    df_view = df.copy()
    df_view["decision_show"] = df_view["decision"].replace("", "(빈값)")
    df_view = df_view[df_view["decision_show"].isin(selected)].drop(columns=["decision_show"])
    st.dataframe(df_view, use_container_width=True)


