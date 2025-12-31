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


def _safe_key(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣_]+", "_", str(s))


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
if "last_auto_applied" not in st.session_state:
    st.session_state["last_auto_applied"] = None


# -------------------------
# 1) 질병명 검색/추천
#    - 단일 체크 강제
#    - 추천 결과가 1개면 자동 적용
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
        # ✅ (2) 추천 결과가 1개면 자동 적용 (무한 rerun 방지)
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

        # 기존 선택 반영(렌더링 단계에서 1개만 True)
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
                "선택": st.column_config.CheckboxColumn("선택", help="체크하면 아래 질병 선택이 즉시 변경됩니다."),
                "질병명": st.column_config.TextColumn(width="large"),
                "일치율(%)": st.column_config.NumberColumn(format="%.1f"),
            },
            disabled={"질병명": True, "일치율(%)": True},  # 체크만 편집 가능
            key="rec_table",
        )

        # ✅ (1) 완전 단일 선택 UX
        checked = edited[edited["선택"] == True]["질병명"].tolist()
        if len(checked) >= 1:
            prev = st.session_state["rec_selected_disease"]
            # 여러 개가 체크되었으면, "이전 선택(prev)"을 제외한 새 선택을 우선
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
                # 같은 것을 다시 체크해도, 단일 True 상태로 강제 렌더링되도록 rerun
                if len(checked) > 1:
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

# 질병 변경 시 criteria 자동 보정
if (st.session_state["criteria_selectbox"] is None) or (st.session_state["criteria_selectbox"] not in criteria_list):
    st.session_state["criteria_selectbox"] = criteria_list[0]

crit = st.selectbox("심사기준 선택", criteria_list, key="criteria_selectbox")

df = load_benefit_decisions(DB_URL, disease, crit).copy()

# -------------------------
# 3) 급부별 인수 결과값(결정값) 요약 + 필터 체크
#    - 요약: 결정값별 건수
#    - 체크한 결정값만 표에 표시
# -------------------------
st.subheader("급부별 인수 결과")

# decision 정리
df["decision"] = df["decision"].fillna("").astype(str)

dec_counts = df["decision"].value_counts(dropna=False).sort_values(ascending=False)
decisions = dec_counts.index.tolist()

# 요약 문구(표 위)
summary_parts = []
for d in decisions:
    label = d if d.strip() else "(빈값)"
    summary_parts.append(f"{label}: {int(dec_counts[d])}")
st.caption(" / ".join(summary_parts))

# 필터 체크 UI
st.markdown("**인수결과값 필터** (체크한 값만 아래 표에 표시)")

# 초기값: 모두 True
if "decision_filter_init" not in st.session_state:
    st.session_state["decision_filter_init"] = True
    for d in decisions:
        st.session_state[f"decf_{_safe_key(d)}"] = True

# 결정값 체크박스들을 가로로 배치
cols = st.columns(min(6, max(1, len(decisions))))
for i, d in enumerate(decisions):
    key = f"decf_{_safe_key(d)}"
    label = d if d.strip() else "(빈값)"
    with cols[i % len(cols)]:
        st.checkbox(f"{label} ({int(dec_counts[d])})", key=key)

selected_decisions = []
for d in decisions:
    key = f"decf_{_safe_key(d)}"
    if st.session_state.get(key, False):
        selected_decisions.append(d)

# 선택이 하나도 없으면 빈 표
if selected_decisions:
    df_view = df[df["decision"].isin(selected_decisions)].copy()
else:
    df_view = df.iloc[0:0].copy()

st.dataframe(df_view, use_container_width=True)
