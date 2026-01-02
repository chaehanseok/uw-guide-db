import os
import sqlite3
import requests
import tempfile
from email.utils import parsedate_to_datetime

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
        r = requests.head(db_url, timeout=10, allow_redirects=True)
        lm = r.headers.get("Last-Modified")
        if not lm:
            return "기준일 미상"
        dt = parsedate_to_datetime(lm)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return "기준일 미상"


# =========================
# DB 로드(리소스 캐시)
# =========================
@st.cache_resource
def load_db(db_url: str) -> sqlite3.Connection:
    r = requests.get(db_url, timeout=30)
    r.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.write(r.content)
    tmp.flush()
    tmp.close()

    return sqlite3.connect(tmp.name, check_same_thread=False)


@st.cache_data(ttl=3600)
def load_all_diseases(db_url: str) -> list[str]:
    # cache_data 함수는 sqlite conn을 직접 인자로 받지 않는 편이 안전함
    r = requests.get(db_url, timeout=30)
    r.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.write(r.content)
    tmp.flush()
    tmp.close()

    c = sqlite3.connect(tmp.name)
    try:
        df = pd.read_sql("SELECT DISTINCT disease FROM uw_rows ORDER BY disease", c)
        return df["disease"].dropna().astype(str).tolist()
    finally:
        c.close()
        try:
            os.remove(tmp.name)
        except Exception:
            pass


@st.cache_data(ttl=3600)
def load_criteria_for_disease(db_url: str, disease: str) -> list[str]:
    r = requests.get(db_url, timeout=30)
    r.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.write(r.content)
    tmp.flush()
    tmp.close()

    c = sqlite3.connect(tmp.name)
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
            os.remove(tmp.name)
        except Exception:
            pass


@st.cache_data(ttl=3600)
def load_benefit_decisions(db_url: str, disease: str, criteria: str) -> pd.DataFrame:
    r = requests.get(db_url, timeout=30)
    r.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.write(r.content)
    tmp.flush()
    tmp.close()

    c = sqlite3.connect(tmp.name)
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
            os.remove(tmp.name)
        except Exception:
            pass


# =========================
# UI
# =========================
st.set_page_config(page_title="질병 심사 가이드", layout="wide")

asof_yyyymmdd = get_db_asof_from_github(DB_URL)

st.title("질병 심사 가이드\n(Underwriting Guide)")

st.warning(
    f"이 자료는 미래에셋생명 LoveAge Plan의 질병심사메뉴얼 자료를 ({asof_yyyymmdd})에 수집하였습니다.\n\n"
    "이 자료는 미래에셋금융서비스 구성원들의 내부 교육자료로 질병별 인수기준은 수시 변동될 수 있음으로 "
    "고객안내 및 청약전 미래에셋생명에 직접 확인하시기 바랍니다."
)

# 연결 유지(향후 확장 대비)
conn = load_db(DB_URL)

diseases_all = load_all_diseases(DB_URL)
if not diseases_all:
    st.error("DB에서 질병 목록을 불러오지 못했습니다.")
    st.stop()

st.subheader("질병 선택/조회")

# 1) 텍스트 입력(placeholder)
query = st.text_input(
    label="질병명",
    value="",
    placeholder="질병명을 입력하세요",
)

# 2) 입력값으로 후보 필터링 (부분 포함, 대소문자 무시)
q = query.strip()
if q:
    diseases_filtered = [d for d in diseases_all if q.lower() in d.lower()]
else:
    diseases_filtered = []

# 3) selectbox는 최초 로딩 시 공란처럼 보이게: placeholder + index=None
# Streamlit 버전에 따라 index=None이 안 되면, 아래 fallback(옵션에 "" 추가)로 처리하세요.
disease = st.selectbox(
    "질병 선택",
    options=diseases_filtered,
    index=None,
    placeholder="질병명을 입력하면 후보가 표시됩니다.",
)

if not disease:
    st.info("질병명을 입력한 뒤, 목록에서 질병을 선택하세요.")
    st.stop()

criteria_list = load_criteria_for_disease(DB_URL, disease)
if not criteria_list:
    st.warning("해당 질병의 심사기준이 없습니다.")
    st.stop()

crit = st.selectbox("심사기준 선택", options=criteria_list)

df = load_benefit_decisions(DB_URL, disease, crit)

st.subheader("급부별 인수 결과")
st.dataframe(df, use_container_width=True)
