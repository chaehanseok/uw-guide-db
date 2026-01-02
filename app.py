# app.py
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# =========================
# Config
# =========================
APP_TITLE = "질병명칭 자동완성 기반 급부 조회/필터"
DEFAULT_TIMEOUT = 20
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

# 자동완성 UX 튜닝
MIN_QUERY_LEN = 2          # 2글자부터 후보 조회
DEBOUNCE_MS = 300          # 입력 후 300ms 정지 시 조회 (Streamlit은 이벤트 기반이라 간접 구현)
MAX_SUGGESTIONS = 30       # 콤보 후보 최대


# =========================
# Utilities
# =========================
def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _parse_csv_keywords(raw: str) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[,\n]", raw)
    return [p.strip() for p in parts if p.strip()]


def _contains_any(text: str, keywords: List[str]) -> bool:
    text = (text or "").lower()
    kws = [k.strip().lower() for k in keywords if k.strip()]
    if not kws:
        return True
    return any(k in text for k in kws)


def _contains_none(text: str, keywords: List[str]) -> bool:
    text = (text or "").lower()
    kws = [k.strip().lower() for k in keywords if k.strip()]
    if not kws:
        return True
    return all(k not in text for k in kws)


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return int(v)
        s = str(v).strip()
        if s == "":
            return None
        m = re.search(r"-?\d+", s.replace(",", ""))
        return int(m.group(0)) if m else None
    except Exception:
        return None


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if s == "":
            return None
        s = s.replace(",", "")
        m = re.search(r"-?\d+(\.\d+)?", s)
        return float(m.group(0)) if m else None
    except Exception:
        return None


# =========================
# API Client
# =========================
class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def suggest_diseases_contains(self, query: str, limit: int = MAX_SUGGESTIONS) -> List[Dict[str, Any]]:
        """
        '질병명칭 자동완성(contains)' 후보 조회.
        - 검색 버튼 없음: 입력값 기반으로 여기서 후보를 반환한다.
        - 서버가 contains로 내려주면 그대로 사용.
        - 서버가 광범위하게 내려주면 클라에서 2차 contains 필터를 한다.

        기대 응답(예시)
        - 리스트 또는 {"items":[...]} 형태
        - 각 아이템은 최소 id, name 포함
        """
        q = _clean_text(query)
        if len(q) < MIN_QUERY_LEN:
            return []

        # ### TODO: 실제 자동완성 엔드포인트로 변경
        # 예) GET /api/diseases/suggest?q=...&limit=...
        url = f"{self.base_url}/api/diseases/suggest"
        params = {"q": q, "limit": limit}

        r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
            items = data["items"]
        elif isinstance(data, list):
            items = data
        else:
            items = []

        # 서버가 contains를 보장하지 않는 경우를 대비한 2차 필터
        q_low = q.lower()
        out = []
        for it in items:
            name = str(it.get("name") or it.get("disease_name") or it.get("diseaseName") or "").strip()
            if q_low in name.lower():
                out.append(it)

        return out[:limit]

    def get_benefits_by_disease(self, disease_id: str) -> List[Dict[str, Any]]:
        # ### TODO: 실제 엔드포인트로 변경
        url = f"{self.base_url}/api/diseases/{disease_id}/benefits"
        r = requests.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
            return data["items"]
        if isinstance(data, list):
            return data
        return []


# =========================
# Filtering
# =========================
@dataclass
class BenefitFilters:
    benefit_include: List[str]
    benefit_exclude: List[str]
    reason_include: List[str]
    reason_exclude: List[str]
    exclusion_include: List[str]
    exclusion_exclude: List[str]
    renew_types: List[str]
    product_types: List[str]
    pay_types: List[str]
    min_pay_count: Optional[int]
    max_pay_count: Optional[int]
    min_pay_amount: Optional[float]
    max_pay_amount: Optional[float]


def unique_values(benefits: List[Dict[str, Any]], keys: List[str]) -> List[str]:
    vals = set()
    for b in benefits:
        for k in keys:
            v = b.get(k)
            if v is not None and str(v).strip():
                vals.add(str(v).strip())
    return sorted(vals)


def apply_benefit_filters(benefits: List[Dict[str, Any]], f: BenefitFilters) -> List[Dict[str, Any]]:
    def get_field(b: Dict[str, Any], *keys: str) -> str:
        for k in keys:
            v = b.get(k)
            if v is not None:
                return str(v)
        return ""

    def get_num(b: Dict[str, Any], *keys: str) -> Optional[float]:
        for k in keys:
            if k in b and b.get(k) is not None:
                return _safe_float(b.get(k))
        return None

    filtered: List[Dict[str, Any]] = []

    for b in benefits:
        benefit_name = get_field(b, "benefit_name", "benefitNm", "급부명", "담보명")
        pay_reason = get_field(b, "pay_reason", "payReason", "지급사유", "지급조건")
        exclusion = get_field(b, "exclusion", "exclusionText", "면책", "면책사항")
        renew_type = get_field(b, "renew_type", "renewType", "갱신구분")
        product_type = get_field(b, "product_type", "productType", "계약구분", "주특약구분")
        pay_type = get_field(b, "pay_type", "payType", "급부유형", "지급유형")

        pay_limit_count = get_num(b, "pay_limit_count", "payLimitCount", "지급횟수", "한도횟수")
        pay_limit_amount = get_num(b, "pay_limit_amount", "payLimitAmount", "지급한도", "한도금액")

        if not _contains_any(benefit_name, f.benefit_include):
            continue
        if not _contains_none(benefit_name, f.benefit_exclude):
            continue

        if not _contains_any(pay_reason, f.reason_include):
            continue
        if not _contains_none(pay_reason, f.reason_exclude):
            continue

        if not _contains_any(exclusion, f.exclusion_include):
            continue
        if not _contains_none(exclusion, f.exclusion_exclude):
            continue

        if f.renew_types and renew_type not in f.renew_types:
            continue
        if f.product_types and product_type not in f.product_types:
            continue
        if f.pay_types and pay_type not in f.pay_types:
            continue

        if f.min_pay_count is not None:
            c = _safe_int(pay_limit_count)
            if c is None or c < f.min_pay_count:
                continue
        if f.max_pay_count is not None:
            c = _safe_int(pay_limit_count)
            if c is None or c > f.max_pay_count:
                continue

        if f.min_pay_amount is not None:
            a = _safe_float(pay_limit_amount)
            if a is None or a < f.min_pay_amount:
                continue
        if f.max_pay_amount is not None:
            a = _safe_float(pay_limit_amount)
            if a is None or a > f.max_pay_amount:
                continue

        filtered.append(b)

    return filtered


def render_filters(benefits: List[Dict[str, Any]]) -> BenefitFilters:
    st.subheader("급부 필터")

    col1, col2 = st.columns(2)
    with col1:
        benefit_include_raw = st.text_area("급부명 포함 키워드 (쉼표/줄바꿈)", height=80, key="benefit_include")
        benefit_exclude_raw = st.text_area("급부명 제외 키워드 (쉼표/줄바꿈)", height=80, key="benefit_exclude")

        reason_include_raw = st.text_area("지급사유/조건 포함 키워드 (쉼표/줄바꿈)", height=80, key="reason_include")
        reason_exclude_raw = st.text_area("지급사유/조건 제외 키워드 (쉼표/줄바꿈)", height=80, key="reason_exclude")

        excl_include_raw = st.text_area("면책/감액/대기 관련 포함 키워드 (쉼표/줄바꿈)", height=80, key="excl_include")
        excl_exclude_raw = st.text_area("면책/감액/대기 관련 제외 키워드 (쉼표/줄바꿈)", height=80, key="excl_exclude")

    with col2:
        renew_options = unique_values(benefits, ["renew_type", "renewType", "갱신구분"])
        product_options = unique_values(benefits, ["product_type", "productType", "주특약구분", "계약구분"])
        paytype_options = unique_values(benefits, ["pay_type", "payType", "급부유형", "지급유형"])

        renew_types = st.multiselect("갱신구분", options=renew_options, default=[], key="renew_types")
        product_types = st.multiselect("주/특약 구분", options=product_options, default=[], key="product_types")
        pay_types = st.multiselect("급부유형(진단/입원/수술/통원 등)", options=paytype_options, default=[], key="pay_types")

        st.markdown("#### 숫자 조건")
        c1, c2 = st.columns(2)
        with c1:
            min_pay_count = st.number_input("최소 지급횟수 (0이면 미적용)", min_value=0, value=0, step=1, key="min_pay_count")
            min_pay_count = None if min_pay_count == 0 else int(min_pay_count)
        with c2:
            max_pay_count = st.number_input("최대 지급횟수 (0이면 미적용)", min_value=0, value=0, step=1, key="max_pay_count")
            max_pay_count = None if max_pay_count == 0 else int(max_pay_count)

        a1, a2 = st.columns(2)
        with a1:
            min_pay_amount = st.number_input("최소 한도금액 (0이면 미적용)", min_value=0.0, value=0.0, step=10000.0, key="min_pay_amount")
            min_pay_amount = None if min_pay_amount == 0 else float(min_pay_amount)
        with a2:
            max_pay_amount = st.number_input("최대 한도금액 (0이면 미적용)", min_value=0.0, value=0.0, step=10000.0, key="max_pay_amount")
            max_pay_amount = None if max_pay_amount == 0 else float(max_pay_amount)

    return BenefitFilters(
        benefit_include=_parse_csv_keywords(benefit_include_raw),
        benefit_exclude=_parse_csv_keywords(benefit_exclude_raw),
        reason_include=_parse_csv_keywords(reason_include_raw),
        reason_exclude=_parse_csv_keywords(reason_exclude_raw),
        exclusion_include=_parse_csv_keywords(excl_include_raw),
        exclusion_exclude=_parse_csv_keywords(excl_exclude_raw),
        renew_types=renew_types,
        product_types=product_types,
        pay_types=pay_types,
        min_pay_count=min_pay_count,
        max_pay_count=max_pay_count,
        min_pay_amount=min_pay_amount,
        max_pay_amount=max_pay_amount,
    )


def render_benefits(disease: Dict[str, Any], benefits: List[Dict[str, Any]], filtered: List[Dict[str, Any]]) -> None:
    st.subheader("조회 결과")

    left, right = st.columns([2, 1])
    with left:
        st.markdown("**선택된 질병**")
        st.json(disease)
    with right:
        st.metric("총 급부", len(benefits))
        st.metric("필터 적용 후", len(filtered))

    st.markdown("---")
    if not filtered:
        st.info("필터 적용 후 표시할 급부가 없습니다.")
        return

    rows = []
    for b in filtered:
        rows.append({
            "급부명": b.get("benefit_name") or b.get("benefitNm") or b.get("급부명") or b.get("담보명"),
            "지급사유/조건": b.get("pay_reason") or b.get("payReason") or b.get("지급사유") or b.get("지급조건"),
            "면책/감액/대기": b.get("exclusion") or b.get("exclusionText") or b.get("면책") or b.get("면책사항"),
            "갱신구분": b.get("renew_type") or b.get("renewType") or b.get("갱신구분"),
            "주/특약": b.get("product_type") or b.get("productType") or b.get("주특약구분") or b.get("계약구분"),
            "급부유형": b.get("pay_type") or b.get("payType") or b.get("급부유형") or b.get("지급유형"),
            "지급횟수": b.get("pay_limit_count") or b.get("payLimitCount") or b.get("지급횟수") or b.get("한도횟수"),
            "한도금액": b.get("pay_limit_amount") or b.get("payLimitAmount") or b.get("지급한도") or b.get("한도금액"),
        })
    st.dataframe(rows, use_container_width=True)


# =========================
# Main
# =========================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "검색 버튼 없이 입력 즉시 자동완성 콤보에서 포함(contains) 후보를 보여주고, "
        "선택 시 급부를 조회한 뒤 급부 필터링을 적용합니다. "
        "(질병 유사도/일치율 조정 기능은 제거)"
    )

    client = ApiClient(API_BASE_URL)

    # 상태
    if "last_input_ts" not in st.session_state:
        st.session_state.last_input_ts = 0.0
    if "suggestions" not in st.session_state:
        st.session_state.suggestions = []
    if "selected_disease_id" not in st.session_state:
        st.session_state.selected_disease_id = ""
    if "selected_disease" not in st.session_state:
        st.session_state.selected_disease = None
    if "benefits" not in st.session_state:
        st.session_state.benefits = []

    st.markdown("### 질병명칭 (자동완성)")
    disease_query = st.text_input(
        "질병명칭을 입력하면 포함하는 후보가 즉시 표시됩니다.",
        value=st.session_state.get("disease_query", ""),
        key="disease_query",
        placeholder="예: 심근, 당뇨, 뇌출혈 ..."
    )

    # 입력 변화 감지 및 디바운스 유사 처리
    now = time.time()
    prev_query = st.session_state.get("_prev_query", "")
    if disease_query != prev_query:
        st.session_state["_prev_query"] = disease_query
        st.session_state.last_input_ts = now

    # 디바운스 경과 시에만 조회
    elapsed_ms = (now - st.session_state.last_input_ts) * 1000.0
    should_fetch = len(_clean_text(disease_query)) >= MIN_QUERY_LEN and elapsed_ms >= DEBOUNCE_MS

    # 후보 조회
    if should_fetch:
        try:
            with st.spinner("후보 조회 중..."):
                st.session_state.suggestions = client.suggest_diseases_contains(disease_query)
        except Exception as e:
            st.session_state.suggestions = []
            st.error(f"자동완성 조회 오류: {e}")

    suggestions = st.session_state.suggestions or []

    # 콤보(Selectbox) 구성
    # label은 사용자가 읽는 문자열, value는 item 보관
    def _label(it: Dict[str, Any]) -> str:
        name = it.get("name") or it.get("disease_name") or it.get("diseaseName") or ""
        did = it.get("id") or it.get("disease_id") or it.get("diseaseId") or ""
        # 코드가 있으면 같이 보여줌
        return f"{name} ({did})" if did else f"{name}"

    option_labels = ["(선택)"] + [_label(it) for it in suggestions]
    selected_label = st.selectbox("질병 선택", options=option_labels, index=0, key="disease_selectbox")

    # 선택 처리: 선택 시 급부 로딩
    if selected_label and selected_label != "(선택)":
        # label로 다시 찾기 (동일 label 중복 가능성이 있으면 id 기반으로 바꾸는 것을 권장)
        idx = option_labels.index(selected_label) - 1
        chosen = suggestions[idx] if 0 <= idx < len(suggestions) else None

        if chosen:
            disease_id = chosen.get("id") or chosen.get("disease_id") or chosen.get("diseaseId")
            # 이미 같은 질병이면 재조회 방지
            if disease_id and str(disease_id) != st.session_state.selected_disease_id:
                st.session_state.selected_disease_id = str(disease_id)
                st.session_state.selected_disease = chosen
                try:
                    with st.spinner("급부 조회 중..."):
                        st.session_state.benefits = client.get_benefits_by_disease(str(disease_id))
                except Exception as e:
                    st.session_state.benefits = []
                    st.error(f"급부 조회 오류: {e}")

    # 급부/필터 영역
    selected_disease = st.session_state.selected_disease
    benefits = st.session_state.benefits or []

    if selected_disease:
        filters = render_filters(benefits)
        filtered = apply_benefit_filters(benefits, filters)
        render_benefits(selected_disease, benefits, filtered)
    else:
        st.info("질병명칭을 입력하고, 콤보에서 질병을 선택하세요.")


if __name__ == "__main__":
    main()
