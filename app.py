import os
import re
import sqlite3
import requests
import tempfile
from email.utils import parsedate_to_datetime
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st


DB_URL = "https://raw.githubusercontent.com/chaehanseok/uw-guide-db/main/uw_knowledge.db"
MAX_RECOMMENDATIONS = 30

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


_norm_keep_kor_eng_num = re.compile(r"[^0-9a-zA-Z가-힣]+")

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = _norm_keep_kor_eng_num.sub("", s)
    return s


def similarity(query_raw: str, disease_raw: str) -> float:
    q = (query_raw or "").strip()
    d = (disease_raw or "").strip()
    if not q or not d:
        return 0.0

    qn = normalize_text(q)
    dn = normalize_text(d)
    if not qn or not dn:
        return 0.0

    # 1) 짧은 검색어(1~2글자)는 "포함 검색"을 최우선
    if len(qn) <= 2:
        if qn in dn:
            return 0.95  # 포함이면 거의 최상단
        # 포함이 아니면 기존 유사도
        return SequenceMatcher(None, qn, dn).ratio()

    # 2) 일반 길이 검색어는 기존 로직 + 보너스
    base = SequenceMatcher(None, qn, dn).ratio()

    if qn == dn:
        base = max(base, 0.999)
    elif qn in dn:
        base = min(1.0, base + 0.30)  # 포함 보너스 좀 더 강하게
    elif dn in qn:
        base = min(1.0, base + 0.10)

    return base

def recommend_diseases(query: str, diseases: list[str], top_k: int = 50) -> pd.DataFrame:
    q = (query or "").strip()
    if not q:
        return pd.DataFrame(columns=["질병명", "일치율(%)"])

    qn = normalize_text(q)

    # ✅ 포함되는 것 먼저 싹 모으기 (특히 짧은 검색어에서 체감 개선)
    contains = []
    others = []
    for d in diseases:
        dn = normalize_text(d)
        if qn and qn in dn:
            contains.append(d)
        else:
            others.append(d)

    # 포함 그룹은 전부 보여주고, 부족하면 others에서 점수순으로 채움
    scored_others = [(d, similarity(q, d)) for d in others]
    scored_others.sort(key=lambda x: x[1], reverse=True)

    merged = [(d, 0.95) for d in contains] + scored_others
    merged = merged[:top_k]

    df = pd.DataFrame({
        "질병명": [d for d, _ in merged],
        "일치율(%)": [round(s * 100, 1) for _, s in merged],
    })
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

# 세션 상태
if "disease_selectbox" not in st.session_state:
    st.session_state["disease_selectbox"] = diseases[0]
if "criteria_selectbox" not in st.session_state:
    st.session_state["criteria_selectbox"] = None
if "rec_selected_disease" not in st.session_state:
    st.session_state["rec_selected_disease"] = None
if "last_auto_applied" not in st.session_state:
    st.session_state["last_auto_applied"] = None


# -------------------------
# 검색/추천
# -------------------------
st.subheader("질병명 검색/추천")

query = st.text_input("질병명을 입력하세요 (예: 척추염, 당뇨, 객혈 등)", value="")
min_match = st.slider("최소 일치율(%)", 0, 100, 70, 1)

if query.strip():
    rec_df = recommend_diseases(query, diseases)
    rec_df = rec_df[rec_df["일치율(%)"] >= float(min_match)].head(MAX_RECOMMENDATIONS)

    if rec_df.empty:
        st.info("조건에 맞는 추천 결과가 없습니다. 최소 일치율을 낮추거나 다른 키워드로 시도해 보세요.")
    else:
        # 추천 결과가 1개면 자동 적용
        if len(rec_df) == 1:
            only_disease = rec_df.iloc[0]["질병명"]
            signature = f"{query}|{min_match}|{only_disease}"
            if st.session_state["last_auto_applied"] != signature:
                st.session_state["last_auto_applied"] = signature
                st.session_state["rec_selected_disease"] = only_disease
                st.session_state["disease_selectbox"] = only_disease
                st.session_state["criteria_selectbox"] = None
                st.rerun()

        show_df = rec_df[["질병명", "일치율(%)"]].copy()
        show_df.insert(0, "선택", False)

        if st.session_state["rec_selected_disease"]:
            show_df.loc[show_df["질병명"] == st.session_state["rec_selected_disease"], "선택"] = True

        edited = st.data_editor(
            show_df,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            disabled={"질병명": True, "일치율(%)": True},
            column_config={
                "선택": st.column_config.CheckboxColumn("선택"),
                "질병명": st.column_config.TextColumn(width="large"),
                "일치율(%)": st.column_config.NumberColumn(format="%.1f"),
            },
            key="rec_table",
        )

        checked = edited[edited["선택"] == True]["질병명"].tolist()
        if len(checked) >= 1:
            prev = st.session_state["rec_selected_disease"]
            if prev and (prev in checked) and (len(checked) > 1):
                new_choice = next((x for x in checked if x != prev), checked[0])
            else:
                new_choice = checked[0]

            if new_choice != st.session_state["rec_selected_disease"]:
                st.session_state["rec_selected_disease"] = new_choice
                st.session_state["disease_selectbox"] = new_choice
                st.session_state["criteria_selectbox"] = None
                st.rerun()
            else:
                if len(checked) > 1:
                    st.rerun()

st.divider()


# -------------------------
# 질병/심사기준 선택
# -------------------------
st.subheader("질병 선택/조회")

disease = st.selectbox("질병 선택", diseases, key="disease_selectbox")

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
# ✅ 인수결과값 요약 + 필터링 (확실히 보이도록)
# -------------------------
st.subheader("급부별 인수 결과")

# 요약 테이블
counts = df["decision"].replace("", "(빈값)").value_counts().reset_index()
counts.columns = ["인수결과값", "건수"]

# metric 형태 + 표 형태 둘 다 제공 (눈에 확 들어오게)
c1, c2 = st.columns([1, 2])
with c1:
    st.metric("총 급부 수", int(len(df)))
with c2:
    st.dataframe(counts, hide_index=True, use_container_width=True)

# 필터: multiselect (필터 존재가 명확)
all_decisions = counts["인수결과값"].tolist()

# 기본값: 전체 선택
default_selected = all_decisions

selected = st.multiselect(
    "인수결과값 필터 (선택한 값만 아래 표에 표시)",
    options=all_decisions,
    default=default_selected,
)

if not selected:
    st.warning("필터가 모두 해제되어 표시할 데이터가 없습니다.")
    st.dataframe(df.iloc[0:0], use_container_width=True)
else:
    df_view = df.copy()
    df_view["decision_show"] = df_view["decision"].replace("", "(빈값)")
    df_view = df_view[df_view["decision_show"].isin(selected)].drop(columns=["decision_show"])

    st.dataframe(df_view, use_container_width=True)
