import os
import sqlite3
import requests
import tempfile
from datetime import datetime
from email.utils import parsedate_to_datetime

import pandas as pd
import streamlit as st


# =========================
# 설정
# =========================
DB_URL = "https://raw.githubusercontent.com/chaehanseok/uw-guide-db/main/uw_knowledge.db"


# =========================
# 유틸: GitHub Last-Modified로 asof 추정
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
# DB 로드
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
# 데이터 로드 (cache_data는 conn을 인자로 받지 않게!)
# =========================
@st.cache_data(ttl=3600)
def load_all_diseases(db_url: str) -> list[str]:
    # cache_data에서 conn을 파라미터로 받으면 UnhashableParamError가 나므로
    # 여기서는 임시로 새로 연결해서 읽고 닫습니다(가볍고 안정적).
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
# 검색/추천 로직 (외부 라이브러리 없이 구현)
# - SequenceMatcher는 한글에서도 기본 유사도에 도움
# - 추가로 "부분 포함"을 가산점으로 줌
# =========================
from difflib import SequenceMatcher

def similarity(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0
    base = SequenceMatcher(None, a, b).ratio()  # 0~1
    # 부분 포함 가산점(짧은 키워드 검색에 체감 좋음)
    if a in b:
        base = min(1.0, base + 0.15)
    return base


def recommend_diseases(query: str, diseases: list[str], top_k: int = 10) -> pd.DataFrame:
    q = (query or "").strip()
    if not q:
        return pd.DataFrame(columns=["disease", "score"])

    scored = []
    for d in diseases:
        s = similarity(q, d)
        scored.append((d, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]
    df = pd.DataFrame(top, columns=["disease", "score"])
    df["score"] = (df["score"] * 100).round(1)  # %
    return df


# =========================
# UI
# =========================
st.set_page_config(page_title="질병 심사 가이드", layout="wide")

# 기준일/경고 문구
asof_yyyymmdd = get_db_asof_from_github(DB_URL)

st.title("질병 심사 가이드 (Underwriting Guide)")

st.warning(
    "본 인수기준은 내부 교육용입니다. "
    f"({asof_yyyymmdd}, LoveAge Plan 질병심사메뉴얼 등록기준).\n"
    "변동 사항이 있을 수 있으며 실제 인수기준은 반드시 확인후 고객에게 안내 바랍니다."
)

# 질병 목록 로드
diseases = load_all_diseases(DB_URL)

# 세션 상태: 선택 질병
if "selected_disease" not in st.session_state:
    st.session_state.selected_disease = diseases[0] if diseases else ""

# -------------------------
# 자연어 검색 & 추천
# -------------------------
st.subheader("질병명 검색")

query = st.text_input("자연어로 질병명을 입력하세요 (예: 척추염, 당뇨, 객혈 등)", value="")

rec_df = recommend_diseases(query, diseases, top_k=10)

if query.strip():
    if rec_df.empty or rec_df["score"].max() <= 0:
        st.info("추천 결과가 없습니다. 다른 키워드로 시도해 보세요.")
    else:
        st.caption("추천 결과를 클릭하면 아래 질병 선택이 자동으로 변경됩니다.")

        # 추천결과 클릭 UI: 버튼 리스트
        # (Streamlit 기본 dataframe 클릭 이벤트가 제한적이므로 버튼이 가장 확실)
        for _, row in rec_df.iterrows():
            d = row["disease"]
            s = row["score"]
            col1, col2 = st.columns([6, 1])
            with col1:
                if st.button(f"{d}", key=f"rec_{d}"):
                    st.session_state.selected_disease = d
                    st.session_state.selected_criteria = None  # 기준도 초기화
                    st.rerun()
            with col2:
                st.write(f"{s}%")

# -------------------------
# 질병 선택 (추천 클릭 시 자동 변경)
# -------------------------
st.subheader("질병 선택/조회")

# selectbox의 index는 질병 목록에서 찾아야 함
try:
    disease_index = diseases.index(st.session_state.selected_disease)
except ValueError:
    disease_index = 0

disease = st.selectbox(
    "질병 리스트",
    diseases,
    index=disease_index,
    key="disease_selectbox",
)

# 사용자가 selectbox로 바꿨으면 세션도 갱신
if disease != st.session_state.selected_disease:
    st.session_state.selected_disease = disease
    st.session_state.selected_criteria = None

# 기준(심사기준) 로드
criteria_list = load_criteria_for_disease(DB_URL, st.session_state.selected_disease)

# 세션 상태: 선택 기준
if "selected_criteria" not in st.session_state or st.session_state.selected_criteria is None:
    st.session_state.selected_criteria = criteria_list[0] if criteria_list else ""

# 기준 selectbox 인덱스
try:
    crit_index = criteria_list.index(st.session_state.selected_criteria)
except ValueError:
    crit_index = 0

crit = st.selectbox(
    "심사기준 선택",
    criteria_list,
    index=crit_index,
    key="criteria_selectbox",
)

if crit != st.session_state.selected_criteria:
    st.session_state.selected_criteria = crit

# 결과 조회
if st.session_state.selected_disease and st.session_state.selected_criteria:
    df = load_benefit_decisions(DB_URL, st.session_state.selected_disease, st.session_state.selected_criteria)

    st.subheader("급부별 인수 결과")
    st.dataframe(df, use_container_width=True)
else:
    st.info("질병/심사기준이 비어 있습니다.")
