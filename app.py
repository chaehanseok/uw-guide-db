import sqlite3
import requests
import tempfile
import streamlit as st
import pandas as pd

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

st.title("질병 심사 가이드 (Underwriting Guide)")

diseases = pd.read_sql(
    "SELECT DISTINCT disease FROM uw_rows ORDER BY disease",
    conn
)["disease"].tolist()

disease = st.selectbox("질병 선택", diseases)

criteria = pd.read_sql(
    "SELECT DISTINCT criteria FROM uw_rows WHERE disease = ?",
    conn,
    params=(disease,)
)["criteria"].tolist()

crit = st.selectbox("심사기준 선택", criteria)

df = pd.read_sql(
    """
    SELECT benefit, decision
    FROM uw_rows
    WHERE disease = ? AND criteria = ?
    """,
    conn,
    params=(disease, crit)
)

st.subheader("급부별 인수 결과")
st.dataframe(df, use_container_width=True)
