import sqlite3
import requests
import tempfile
import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz

DB_URL = "https://raw.githubusercontent.com/chaehanseok/uw-guide-db/main/uw_knowledge.db"

@st.cache_resource
def load_db():
    r = requests.get(DB_URL, timeout=30)
    r.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.write(r.content)
    tmp.flush()
    tmp.close()

    conn = sqlite3.connect(tmp.name, check_same_thread=False)
    return conn

conn = load_db()

@st.cache_data
def load_diseases(conn):
    df = pd.read_sql("SELECT DISTINCT disease FROM uw_rows ORDER BY disease", conn)
    diseases = df["disease"].dropna().astype(str).str.strip().tolist()
    diseases = [d for d in diseases if d]
    return diseases

diseases = load_diseases(conn)

st.title("질병 심사 가이드 (Underwriting Guide)")

# -----------------------
# 1) 질병명 자연어 검색 + 추천
# -----------------------
st.subheader("질병명 검색")

query = st.text_input("질병명을 입력하세요 (예: 강직척추염, 객혈, 당뇨 등)", value="")

# 추천 옵션
TOP_N = 15
MIN_SCORE = st.slider("추천 최소 점수", 0, 100, 65, 1)

recommended = []
if query.strip():
    # RapidFuzz: (match, score, index) 튜플 반환
    results = process.extract(
        query.strip(),
        diseases,
        scorer=fuzz.WRatio,  # 한글에도 비교적 안정적, 오타/띄어쓰기 차이에도 강함
        limit=TOP_N
    )
    recommended = [(m, s) for (m, s, _) in results if s >= MIN_SCORE]

    if recommended:
        st.caption(f"추천 결과(상위 {len(recommended)}개, 최소 점수 {MIN_SCORE})")
        rec_df = pd.DataFrame(recommended, columns=["질병명", "일치율(점수)"])
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
    else:
        st.warning("추천 결과가 없습니다. 다른 키워드로 검색하거나 더 정확히 입력해 보세요.")

# 사용자가 추천 결과 중 선택하거나, 전체 리스트에서 선택
if recommended:
    default_disease = recommended[0][0]
else:
    default_disease = diseases[0] if diseases else ""

disease = st.selectbox(
    "질병 선택 (추천 결과가 있으면 첫 후보가 기본 선택됩니다)",
    diseases,
    index=diseases.index(default_disease) if default_disease in diseases else 0
)

# -----------------------
# 2) 심사기준/급부 결과 출력 (기존 로직)
# -----------------------
criteria = pd.read_sql(
    "SELECT DISTINCT criteria FROM uw_rows WHERE disease = ? ORDER BY criteria",
    conn,
    params=(disease,)
)["criteria"].tolist()

crit = st.selectbox("심사기준 선택", criteria)

df = pd.read_sql(
    """
    SELECT benefit, decision
    FROM uw_rows
    WHERE disease = ? AND criteria = ?
    ORDER BY benefit
    """,
    conn,
    params=(disease, crit)
)

st.subheader("급부별 인수 결과")
st.dataframe(df, use_container_width=True)

