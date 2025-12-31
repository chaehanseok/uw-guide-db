import os
import sqlite3
import requests
import tempfile
from email.utils import parsedate_to_datetime

import pandas as pd
import streamlit as st
from difflib import SequenceMatcher


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
# DB 로드(리소스 캐시)
# =========================
@st.cache_resource
def load_db(db_url: str):
    r = requests.get(db_url, timeout=30)
    r.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.write(r.content)
    tmp.flush()
    tmp.close()

    conn = sqlite3.connect(tmp.name, check_same_thread=False)
    return conn


# =========================
# 데이터 로드(cache_data는 conn을 인자로 받지 않음)
# =========================
@st.cache_data(ttl=3600)
def load_all_diseases(db_url: str) -> list[str]:
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
            "SELECT DISTINCT criteria FROM uw_rows WHERE disease = ? ORDER BY criteria",
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
# 검색/추천
# =========================
def similarity(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0
    base = SequenceMatcher(None, a, b).ratio()  # 0~1
    if a in b:
        base = min(1.0, base + 0.15)
    return base


def recommend_diseases(query: str, diseases: list[str], top_k: int = 10) -> pd.DataFrame:
    q = (query or "").strip()
    if not q:
        return pd.DataFrame(columns=["disease", "score"])

    scored = [(d, similarity(q, d)) for d in diseases]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    df = pd.DataFrame(top, columns=["disease", "score"])
    df["score"] = (df["score"] * 100).round(1)
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

# (선택) DB 연결(현재 화면에선 직접 query는 cache_data 쪽으로 하고 있지만,
# 추후 확장 대비해서 리소스 캐시로 유지)
_ = load_db(DB_URL)

# 질병 목록
diseases = load_all_diseases(DB_URL)
if not diseases:
    st.error("DB에서 질병 목록을 불러오지 못했습니다.")
    st.stop()

# ---------------------------------
# 세션 상태: 위젯 key를 기준으로만 관리
# ---------------------------------
if "disease_selectbox" not in st.session_state:
    st.session_state["disease_selectbox"] = diseases[0]

# criteria_selectbox는 disease 바뀔 때마다 재설정될 수 있으므로,
# 아래에서 disease 확정 후 세팅합니다.

# ---------------------------------
# 추천(검색) 영역 + 추천 슬라이드(top_k)
# ---------------------------------
st.subheader("질병명 검색/추천")

colA, colB = st.columns([3, 1])
with colA:
    query = st.text_input("자연어로 질병명을 입력하세요 (예: 척추염, 당뇨, 객혈 등)", value="")
with colB:
    top_k = st.slider("추천 개수", min_value=3, max_value=20, value=10, step=1)  # ✅ 추천 슬라이드 복원

rec_df = recommend_diseases(query, diseases, top_k=top_k)

if query.strip():
    if rec_df.empty or rec_df["score"].max() <= 0:
        st.info("추천 결과가 없습니다. 다른 키워드로 시도해 보세요.")
    else:
        st.caption("아래 추천 질병명을 클릭하면, 아래 ‘질병 리스트’ 선택이 즉시 해당 질병으로 변경됩니다.")

        # 추천 리스트를 버튼으로 제공 (가장 확실하게 클릭 이벤트 처리)
        for i, row in rec_df.iterrows():
            d = row["disease"]
            s = row["score"]
            c1, c2 = st.columns([8, 2])
            with c1:
                if st.button(d, key=f"rec_btn_{i}"):
                    # ✅ 핵심: selectbox key에 직접 값을 넣고 rerun
                    st.session_state["disease_selectbox"] = d
                    # 질병이 바뀌었으니 criteria도 초기화
                    st.session_state.pop("criteria_selectbox", None)
                    st.rerun()
            with c2:
                st.write(f"{s}%")

st.divider()

# ---------------------------------
# 질병/심사기준 선택
# ---------------------------------
st.subheader("질병 선택/조회")

disease = st.selectbox(
    "질병 리스트",
    diseases,
    key="disease_selectbox",  # ✅ key 기반으로 값이 유지/변경됨
)

# disease가 결정된 뒤 criteria 목록 로드
criteria_list = load_criteria_for_disease(DB_URL, disease)
if not criteria_list:
    st.warning("해당 질병의 심사기준이 없습니다.")
    st.stop()

# criteria 초기 세팅(질병 변경 시 pop 했기 때문에 여기서 새로 잡힘)
if "criteria_selectbox" not in st.session_state:
    st.session_state["criteria_selectbox"] = criteria_list[0]
else:
    # 기존 값이 새 목록에 없으면 첫 값으로 보정
    if st.session_state["criteria_selectbox"] not in criteria_list:
        st.session_state["criteria_selectbox"] = criteria_list[0]

crit = st.selectbox(
    "심사기준 선택",
    criteria_list,
    key="criteria_selectbox",
)

df = load_benefit_decisions(DB_URL, disease, crit)

st.subheader("급부별 인수 결과")
st.dataframe(df, use_container_width=True)
