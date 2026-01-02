# app.py
import os
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st


# =========================
# Config
# =========================
APP_TITLE = "질병명칭 기반 급부 조회/필터"
DEFAULT_TIMEOUT = 20

# 환경변수로 API 서버 지정 (없으면 로컬 가정)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


# =========================
# Utilities
# =========================
def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


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


def _parse_csv_keywords(raw: str) -> List[str]:
    if not raw:
        return []
    # 쉼표/줄바꿈 기준 모두 지원
    parts = re.split(r"[,\n]", raw)
    return [p.strip() for p in parts if p.strip()]


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return int(v)
        s = str(v).strip()
        if s == "":
            return None
        # "10회", "10 회", "10회 지급" 같은 케이스 대응
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
# API Client (Adjust endpoints/fields here)
# =========================
class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def search_disease_by_name(self, disease_name: str) -> Dict[str, Any]:
        """
        질병명칭 검색(이번 버전 단일 기능) 호출.
        - '질명명 추출/검색'이나 '유사도 매칭'은 여기서 하지 않음.
        - 입력값 그대로 전달.

        기대 응답(예시):
        {
          "id": "D12345",
          "name": "급성심근경색증",
          "normalized": "급성심근경색증",
          "meta": {...}
        }

        실제 스펙에 맞춰 endpoint/path/field만 맞추면 됨.
        """
        disease_name = _clean_text(disease_name)
        if not disease_name:
            raise ValueError("질병명칭이 비어 있습니다.")

        # ### TODO: 실제 엔드포인트에 맞추세요.
        # 예) GET /api/diseases/search?name=...
        url = f"{self.base_url}/api/diseases/search"
        params = {"name": disease_name}

        r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        # ### TODO: API가 리스트를 준다면 대표값 선택 로직 필요
        # 예: {"items":[{...},{...}]} 형태일 경우
        if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
            items = data["items"]
            if not items:
                raise ValueError("질병명칭 검색 결과가 없습니다.")
            # 대표값 1개를 반환한다고 하셨으니, 우선 첫 번째를 사용
            return items[0]

        if not data:
            raise ValueError("질병명칭 검색 결과가 없습니다.")
        return data

    def get_benefits_by_disease(self, disease_id: str) -> List[Dict[str, Any]]:
        """
        질병 기준 급부(담보/급부) 리스트 조회

        기대 응답(예시): benefits list
        [
          {
            "benefit_id": "...",
            "benefit_name": "...",
            "coverage_name": "...",
            "pay_reason": "...",
            "exclusion": "...",
            "waiting_period": "...",
            "pay_limit_count": 10,
            "pay_limit_amount": 1000000,
            "renew_type": "갱신형",
            "product_type": "특약",
            ...
          },
          ...
        ]
        """
        disease_id = _clean_text(disease_id)
        if not disease_id:
            return []

        # ### TODO: 실제 엔드포인트에 맞추세요.
        url = f"{self.base_url}/api/diseases/{disease_id}/benefits"
        r = requests.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        # 리스트/딕셔너리 모두 대응
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

    renew_types: List[str]          # ["갱신형", "비갱신형"] 등
    product_types: List[str]        # ["주계약", "특약"] 등
    pay_types: List[str]            # ["진단", "입원", "수술", "통원"] 등 (프로젝트에 맞게)

    min_pay_count: Optional[int]
    max_pay_count: Optional[int]

    min_pay_amount: Optional[float]
    max_pay_amount: Optional[float]


def apply_benefit_filters(benefits: List[Dict[str, Any]], f: BenefitFilters) -> List[Dict[str, Any]]:
    """
    급부별 필터링을 복구/유지하는 핵심 함수.
    프로젝트의 benefit 필드명에 맞춰 아래 'field mapping'만 맞추면 됨.
    """

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

        # 텍스트 포함/제외: 급부명
        if not _contains_any(benefit_name, f.benefit_include):
            continue
        if not _contains_none(benefit_name, f.benefit_exclude):
            continue

        # 텍스트 포함/제외: 지급사유/지급조건
        if not _contains_any(pay_reason, f.reason_include):
            continue
        if not _contains_none(pay_reason, f.reason_exclude):
            continue

        # 텍스트 포함/제외: 면책/감액/대기기간 텍스트
        # (프로젝트에서 면책/감액/대기기간을 한 필드로 다루면 여기서 처리)
        if not _contains_any(exclusion, f.exclusion_include):
            continue
        if not _contains_none(exclusion, f.exclusion_exclude):
            continue

        # 선택형 필터: 갱신구분
        if f.renew_types:
            if renew_type not in f.renew_types:
                continue

        # 선택형 필터: 주/특약 등
        if f.product_types:
            if product_type not in f.product_types:
                continue

        # 선택형 필터: 급부유형(진단/입원/수술/통원 등)
        if f.pay_types:
            if pay_type not in f.pay_types:
                continue

        # 숫자 조건: 지급횟수
        if f.min_pay_count is not None:
            c = _safe_int(pay_limit_count)
            if c is None or c < f.min_pay_count:
                continue
        if f.max_pay_count is not None:
            c = _safe_int(pay_limit_count)
            if c is None or c > f.max_pay_count:
                continue

        # 숫자 조건: 한도금액
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


def unique_values(benefits: List[Dict[str, Any]], keys: List[str]) -> List[str]:
    vals = set()
    for b in benefits:
        for k in keys:
            v = b.get(k)
            if v is not None and str(v).strip() != "":
                vals.add(str(v).strip())
    return sorted(vals)


# =========================
# UI
# =========================
def render_filters(benefits: List[Dict[str, Any]]) -> BenefitFilters:
    st.subheader("급부 필터")

    col1, col2 = st.columns(2)

    with col1:
        benefit_include_raw = st.text_area("급부명 포함 키워드 (쉼표/줄바꿈)", height=80, key="benefit_include")
        benefit_exclude_raw = st.text_area("급부명 제외 키워드 (쉼표/줄바꿈)", height=80, key="benefit_exclude")

        reason_include_raw = st.text_area("지급사유/조건 포함 키워드 (쉼표/줄바꿈)", height=80, key="reason_include")
        reason_exclude_raw = st.text_area("지급사유/조건 제외 키워드 (쉼표/줄바꿈)", height=80, key="reason_exclude")

        exclusion_include_raw = st.text_area("면책/감액/대기 관련 포함 키워드 (쉼표/줄바꿈)", height=80, key="excl_include")
        exclusion_exclude_raw = st.text_area("면책/감액/대기 관련 제외 키워드 (쉼표/줄바꿈)", height=80, key="excl_exclude")

    with col2:
        # 선택값은 데이터에서 자동 수집 (필드명 프로젝트에 맞추어 unique_values 키 조정 가능)
        renew_options = unique_values(benefits, ["renew_type", "renewType", "갱신구분"])
        product_options = unique_values(benefits, ["product_type", "productType", "주특약구분", "계약구분"])
        paytype_options = unique_values(benefits, ["pay_type", "payType", "급부유형", "지급유형"])

        renew_types = st.multiselect("갱신구분", options=renew_options, default=[], key="renew_types")
        product_types = st.multiselect("주/특약 구분", options=product_options, default=[], key="product_types")
        pay_types = st.multiselect("급부유형(진단/입원/수술/통원 등)", options=paytype_options, default=[], key="pay_types")

        st.markdown("#### 숫자 조건")
        c1, c2 = st.columns(2)
        with c1:
            min_pay_count = st.number_input("최소 지급횟수", min_value=0, value=0, step=1, key="min_pay_count")
            min_pay_count = None if min_pay_count == 0 else int(min_pay_count)
        with c2:
            max_pay_count = st.number_input("최대 지급횟수 (0이면 미적용)", min_value=0, value=0, step=1, key="max_pay_count")
            max_pay_count = None if max_pay_count == 0 else int(max_pay_count)

        a1, a2 = st.columns(2)
        with a1:
            min_pay_amount = st.number_input("최소 한도금액", min_value=0.0, value=0.0, step=10000.0, key="min_pay_amount")
            min_pay_amount = None if min_pay_amount == 0 else float(min_pay_amount)
        with a2:
            max_pay_amount = st.number_input("최대 한도금액 (0이면 미적용)", min_value=0.0, value=0.0, step=10000.0, key="max_pay_amount")
            max_pay_amount = None if max_pay_amount == 0 else float(max_pay_amount)

    return BenefitFilters(
        benefit_include=_parse_csv_keywords(benefit_include_raw),
        benefit_exclude=_parse_csv_keywords(benefit_exclude_raw),
        reason_include=_parse_csv_keywords(reason_include_raw),
        reason_exclude=_parse_csv_keywords(reason_exclude_raw),
        exclusion_include=_parse_csv_keywords(exclusion_include_raw),
        exclusion_exclude=_parse_csv_keywords(exclusion_exclude_raw),
        renew_types=renew_types,
        product_types=product_types,
        pay_types=pay_types,
        min_pay_count=min_pay_count,
        max_pay_count=max_pay_count,
        min_pay_amount=min_pay_amount,
        max_pay_amount=max_pay_amount,
    )


def render_results(disease: Dict[str, Any], benefits: List[Dict[str, Any]], filtered: List[Dict[str, Any]]) -> None:
    st.subheader("검색 결과")

    left, right = st.columns([2, 1])
    with left:
        st.markdown("**질병(대표값)**")
        st.json(disease)
    with right:
        st.metric("총 급부", len(benefits))
        st.metric("필터 적용 후", len(filtered))

    st.markdown("---")

    if not filtered:
        st.info("필터 적용 후 표시할 급부가 없습니다.")
        return

    # 표 표시: 주요 필드만 우선 노출 (필드명은 프로젝트에 맞게 조정 가능)
    display_rows = []
    for b in filtered:
        display_rows.append({
            "급부명": b.get("benefit_name") or b.get("benefitNm") or b.get("급부명") or b.get("담보명"),
            "지급사유/조건": b.get("pay_reason") or b.get("payReason") or b.get("지급사유") or b.get("지급조건"),
            "면책/감액/대기": b.get("exclusion") or b.get("exclusionText") or b.get("면책") or b.get("면책사항"),
            "갱신구분": b.get("renew_type") or b.get("renewType") or b.get("갱신구분"),
            "주/특약": b.get("product_type") or b.get("productType") or b.get("주특약구분") or b.get("계약구분"),
            "급부유형": b.get("pay_type") or b.get("payType") or b.get("급부유형") or b.get("지급유형"),
            "지급횟수": b.get("pay_limit_count") or b.get("payLimitCount") or b.get("지급횟수") or b.get("한도횟수"),
            "한도금액": b.get("pay_limit_amount") or b.get("payLimitAmount") or b.get("지급한도") or b.get("한도금액"),
        })

    st.dataframe(display_rows, use_container_width=True)

    with st.expander("원본(JSON) 보기", expanded=False):
        st.json(filtered)


# =========================
# Main
# =========================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    st.caption(
        "이번 버전: 질병명칭 검색이 질명명 처리까지 포함. "
        "질명명 검색/질병 유사도/일치율 조정 기능은 제거하고, 급부별 필터링 기능은 복구."
    )

    client = ApiClient(API_BASE_URL)

    st.markdown("### 질병명칭 입력")
    disease_name = st.text_input("질병명칭", value=st.session_state.get("disease_name", ""), key="disease_name")

    colA, colB = st.columns([1, 3])
    with colA:
        do_search = st.button("검색", type="primary", use_container_width=True)

    # 상태 보관
    if "disease" not in st.session_state:
        st.session_state["disease"] = None
    if "benefits" not in st.session_state:
        st.session_state["benefits"] = []

    if do_search:
        try:
            if not _clean_text(disease_name):
                st.warning("질병명칭을 입력하세요.")
            else:
                with st.spinner("질병명칭 검색 중..."):
                    disease = client.search_disease_by_name(disease_name)

                disease_id = disease.get("id") or disease.get("disease_id") or disease.get("diseaseId")
                if not disease_id:
                    st.error("질병 검색 결과에서 질병 ID를 찾지 못했습니다. (API 응답 필드 확인 필요)")
                else:
                    with st.spinner("급부 목록 조회 중..."):
                        benefits = client.get_benefits_by_disease(str(disease_id))

                    st.session_state["disease"] = disease
                    st.session_state["benefits"] = benefits

        except requests.HTTPError as e:
            st.error(f"API 오류: {e}")
        except Exception as e:
            st.error(str(e))

    disease = st.session_state.get("disease")
    benefits = st.session_state.get("benefits") or []

    if disease and benefits is not None:
        # 급부 필터 UI는 검색 후 표시
        filters = render_filters(benefits)
        filtered = apply_benefit_filters(benefits, filters)
        render_results(disease, benefits, filtered)
    elif disease and not benefits:
        st.info("급부 데이터가 없습니다(0건).")
    else:
        st.info("질병명칭을 입력하고 검색을 실행하세요.")


if __name__ == "__main__":
    main()
