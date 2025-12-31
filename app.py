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
            "SELECT DISTINCT criteria FROM uw_rows WHERE disease = ? ORDER BY criteria",
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
# 검색/추천 (2) 정규화 기반 개선
# =========================
_norm_keep_kor_eng_num = re.compile(r"[^0-9a-zA-Z가-힣]+")

def normalize_text(s: str) -> str:
    """
    - 공백/특수문자 제거
    - 소문자화
    - 괄호/하이픈/슬래시 등 변형에 강함
    """
    s = (s or "").strip().lower()
    s = _norm_keep_kor_eng_num.sub("", s)
    return s


def similarity(query_raw: str, disease_raw: str) -> float:
    """
    점수 설계(0~1):
    - 정규화된 문자열로 SequenceMatcher
    - 부분 포함 시 보너스(정확 일치/부분 일치 강화)
    """
    qn = normalize_text(query_raw)
    dn = normalize_text(disease_raw)
    if not qn or not dn:
        return 0.0

    base = SequenceMatcher(None, qn, dn).ratio()

    # 포함 보너스 (정규화 기준)
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
        return pd.DataFrame(columns=["질병명", "일치율(%)", "정규화"])

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
    df["정규화"] = df["질병명"].apply(normalize_text)
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

# (연결 유지: 향후 확장 대비)
_ = load_db(DB_URL)

diseases = load_all_diseases(DB_URL)
if not diseases:
    st.error("DB에서 질병 목록을 불러오지 못했습니다.")
    st.stop()

# 세션키: selectbox는 이 키로만 제어
if "disease_selectbox" not in st.session_state:
    st.session_state["disease_selectbox"] = diseases[0]
if "criteria_selectbox" not in st.session_state:
    st.session_state["criteria_selectbox"] = None

# -------------------------
# 1) 추천 UI: 표 + 라디오 + 적용 버튼
# -------------------------

st.subheader("질병명 검색/추천")

colA, colB, colC = st.columns([3, 1, 1])
with colA:
    query = st.text_input("자연어로 질병명을 입력하세요 (예: 척추염, 당뇨, 객혈 등)", value="")
with colB:
    top_k = st.slider("추천 개수", min_value=3, max_value=20, value=10, step=1)
with colC:
    show_norm = st.checkbox("정규화 표시", value=False)

rec_df = recommend_diseases(query, diseases, top_k=top_k) if query.strip() else pd.DataFrame()

if query.strip():
    if rec_df.empty or rec_df["일치율(%)"].max() <= 0:
        st.info("추천 결과가 없습니다. 다른 키워드로 시도해 보세요.")
    else:
        show_df = rec_df.copy()
        if not show_norm:
            show_df = show_df.drop(columns=["정규화"])

        # ✅ 표처럼 보이는 단일 선택용 에디터
        # - 왼쪽에 선택 라디오가 생김
        # - 사용자가 체크/선택하면 즉시 아래에서 적용
        show_df = show_df.reset_index(drop=True)

        edited = st.data_editor(
            show_df,
            hide_index=True,
            use_container_width=True,
            disabled=list(show_df.columns),  # 데이터 편집 불가(표로만 사용)
            num_rows="fixed",
            column_config={
                "질병명": st.column_config.TextColumn(width="large"),
                "일치율(%)": st.column_config.NumberColumn(format="%.1f"),
            },
            key="rec_table_editor",
        )

        # ✅ 단일 선택 구현:
        # Streamlit 기본 data_editor에는 "행 클릭 이벤트"가 없어,
        # 가장 안정적인 방식은 "선택 라디오 컬럼"을 하나 추가하는 것.
        # 따라서 아래처럼 선택용 컬럼을 붙여서 다시 렌더링한다.

        pick_df = show_df.copy()
        pick_df.insert(0, "선택", False)

        picked = st.data_editor(
            pick_df,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "선택": st.column_config.CheckboxColumn("선택", help="선택하면 아래 질병 리스트가 즉시 변경됩니다."),
                "질병명": st.column_config.TextColumn(width="large"),
                "일치율(%)": st.column_config.NumberColumn(format="%.1f"),
            },
            disabled=[c for c in pick_df.columns if c != "선택"],  # 선택만 가능
            key="rec_pick_editor",
        )

        chosen_rows = picked[picked["선택"] == True]
        if len(chosen_rows) > 0:
            chosen_disease = chosen_rows.iloc[0]["질병명"]

            # 여러 개 체크했어도 첫 번째만 사용 + 나머지는 자동 해제 유도
            if st.session_state.get("disease_selectbox") != chosen_disease:
                st.session_state["disease_selectbox"] = chosen_disease
                st.session_state["criteria_selectbox"] = None
                st.rerun()

st.divider()


# -------------------------
# 질병/심사기준 선택
# -------------------------
st.subheader("질병 선택/조회")

disease = st.selectbox(
    "질병 리스트",
    diseases,
    key="disease_selectbox",
)

criteria_list = load_criteria_for_disease(DB_URL, disease)
if not criteria_list:
    st.warning("해당 질병의 심사기준이 없습니다.")
    st.stop()

# criteria 세션 보정
if (st.session_state["criteria_selectbox"] is None) or (st.session_state["criteria_selectbox"] not in criteria_list):
    st.session_state["criteria_selectbox"] = criteria_list[0]

crit = st.selectbox(
    "심사기준 선택",
    criteria_list,
    key="criteria_selectbox",
)

df = load_benefit_decisions(DB_URL, disease, crit)

st.subheader("급부별 인수 결과")
st.dataframe(df, use_container_width=True)
