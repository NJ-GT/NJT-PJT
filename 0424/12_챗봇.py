# -*- coding: utf-8 -*-
"""
서울 숙박시설 화재안전 RAG 챗봇
  - 위험 시설 데이터 검색 (retrieval)
  - 소방법 규정 자동 매칭 (augmented)
  - 비상대피 안내 다국어 지원 (KO / EN / ZH / JA)
  - Claude(Anthropic) / GPT(OpenAI) 선택
"""
import streamlit as st
import pandas as pd
import numpy as np
import glob
import re

st.set_page_config(page_title="화재안전 챗봇", page_icon="🔥", layout="wide")

# ── 전역 CSS ──
st.markdown("""
<style>
[data-testid="stChatMessage"] { border-radius: 10px; padding: 4px 8px; }
.rag-badge {
    background:#EFF6FF; border:1px solid #BFDBFE; border-radius:6px;
    padding:6px 10px; font-size:0.82rem; color:#1D4ED8; margin-bottom:6px;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════
# 사이드바 설정
# ════════════════════════════════════════
with st.sidebar:
    st.markdown("### 설정")

    lang = st.selectbox("응답 언어", ["한국어", "English", "中文", "日本語"])
    lang_code = {"한국어": "ko", "English": "en", "中文": "zh", "日本語": "ja"}[lang]

    st.markdown("---")
    provider = st.selectbox("AI 공급자", ["Claude (Anthropic)", "GPT (OpenAI)"])
    is_claude = provider.startswith("Claude")

    api_key = None
    try:
        api_key = st.secrets["anthropic" if is_claude else "openai"]["api_key"]
    except Exception:
        pass
    if not api_key:
        api_key = st.text_input(
            f"{'Anthropic' if is_claude else 'OpenAI'} API Key",
            type="password",
            placeholder="sk-ant-..." if is_claude else "sk-...",
        )
    if api_key:
        st.success("API 키 연결됨")

    st.markdown("---")
    model_name = st.selectbox(
        "모델",
        ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"] if is_claude
        else ["gpt-4o-mini", "gpt-4o"],
    )

    st.markdown("---")
    st.caption(
        "**secrets.toml 자동 로드**\n"
        "```\n[anthropic]\napi_key = \"sk-ant-...\"\n\n"
        "[openai]\napi_key = \"sk-...\"\n```"
    )

# ════════════════════════════════════════
# 데이터 로드 (캐시)
# ════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_facility_data():
    """시설 데이터를 컬럼 위치 기반으로 로드하여 clean DataFrame 반환"""
    # ── Main (RISK_VARS 포함) ──
    f_main = glob.glob("data/*0423*.csv")[0]
    main   = pd.read_csv(f_main, encoding="utf-8-sig")
    main.columns = [
        "구", "동", "업소명", "업종", "주변건물수",
        "건물나이", "단속위험도", "도로폭위험도", "집중도",
        "위도", "경도", "구조노후도",
    ]

    # ── Core (AHP, 이동시간, 화재수 포함) ──
    core = pd.read_csv("data/data_with_fire_targets.csv", encoding="utf-8-sig")
    # 알려진 컬럼 위치로 선택
    c = core.columns
    core_sel = core[[c[4], c[5], c[11], c[17], c[22], c[34], c[45]]].copy()
    core_sel.columns = [
        "위도", "경도", "이동시간초", "소방위험도_점수",
        "위험점수_AHP", "공식도로폭m", "반경100m_화재수",
    ]

    df = pd.merge(main, core_sel, on=["위도", "경도"], how="left")
    df["위험점수_AHP"] = pd.to_numeric(df["위험점수_AHP"], errors="coerce")
    df["이동시간초"]   = pd.to_numeric(df["이동시간초"],   errors="coerce")
    df["반경100m_화재수"] = pd.to_numeric(df["반경100m_화재수"], errors="coerce").fillna(0)
    df["공식도로폭m"]  = pd.to_numeric(df["공식도로폭m"],  errors="coerce")
    return df.dropna(subset=["위험점수_AHP"]).reset_index(drop=True)


FACILITY_DF = load_facility_data()

# ════════════════════════════════════════
# 소방법 규정 지식베이스
# ════════════════════════════════════════
FIRE_LAW_KB = """
[소방시설 설치 기준 — 숙박시설 (소방시설법 시행령 별표 4)]

■ 소화기
  - 모든 숙박시설: 연면적 33m²마다 능력단위 1 이상

■ 옥내소화전
  - 연면적 1,500m² 이상
  - 지하층·무창층·4층 이상 바닥면적 300m² 이상

■ 스프링클러 (가장 중요)
  - 숙박시설 연면적 1,000m² 이상
  - 6층 이상 건물 (수용 10명 이상)
  - 지하층·무창층 수용 10명 이상
  ※ 외국인관광도시민박업(연면적 230m² 미만 주택)은 간이스프링클러 가능

■ 자동화재탐지설비
  - 모든 숙박시설 (면적 무관)
  - 연면적 400m² 미만: 단독경보형감지기로 대체 가능

■ 비상방송설비
  - 연면적 3,500m² 이상 또는 11층 이상

■ 시각경보기
  - 수용인원 100명 이상

■ 피난구조 설비
  - 피난기구(완강기·구조대): 3층 이상 피난층 제외
  - 유도등: 모든 숙박시설
  - 비상조명등: 모든 숙박시설
  - 공기호흡기: 수용인원 30명 이상 (야간 영업 숙박업)

■ 소화용수
  - 연면적 3,000m² 이상: 상수도 소화용수설비

[층수별 요약]
1~2층  : 소화기 + 단독경보형감지기 + 유도등
3~5층  : + 완강기(피난기구) + 자동화재탐지설비
6~10층 : + 스프링클러 + 옥내소화전
11층↑  : + 비상방송설비 (전층 스프링클러 의무)

[업종별 특이사항]
- 외국인관광도시민박업: 주택 전용, 연면적 230m² 미만 → 간이스프링클러 적용
- 관광숙박업(호텔): 연면적 무관 스프링클러 의무 (수용 10인 이상)
- 일반숙박업: 위 기준 그대로 적용
"""

EVACUATION_GUIDE = {
    "ko": """
[화재 비상대피 매뉴얼 — 숙박시설]
1. 화재경보 시 즉시 외투·귀중품 두고 대피
2. 문 손잡이 확인: 뜨거우면 창문으로 구조 요청, 차가우면 조심히 열고 대피
3. 엘리베이터 절대 사용 금지 → 피난계단(비상구 표시 따라) 이용
4. 연기 발생 시: 수건/옷 적셔 코·입 막고 낮은 자세(기어서)로 이동
5. 옥외 비상집결지(지정 장소) 집합 후 인원 확인
6. 소방서 신고: 119
""",
    "en": """
[Emergency Fire Evacuation — Lodging Facilities]
1. When alarm sounds: leave immediately, do NOT take valuables
2. Check door before opening: if hot → stay inside, signal from window; if cool → open carefully
3. NEVER use elevators → use emergency stairs (follow green EXIT signs)
4. If smoke: cover nose/mouth with wet cloth, stay low (crawl)
5. Gather at designated outdoor assembly point; await headcount
6. Call fire department: 119
""",
    "zh": """
[火灾紧急疏散手册 — 住宿设施]
1. 警报响起时：立即撤离，不要携带贵重物品
2. 开门前检查门把手：发烫→留在室内从窗户求救；不烫→小心开门撤离
3. 严禁使用电梯 → 走紧急楼梯（沿绿色"EXIT"指示牌）
4. 有烟雾时：用湿毛巾捂住口鼻，保持低姿势匍匐前进
5. 在室外指定集合点集合，清点人数
6. 拨打消防电话：119
""",
    "ja": """
[火災緊急避難マニュアル — 宿泊施設]
1. 警報が鳴ったら：すぐに避難、貴重品は置いていく
2. ドアを開ける前に確認：熱い→室内待機・窓から救助要請、冷たい→慎重に開けて避難
3. エレベーター厳禁 → 非常階段（緑色の「EXIT」表示に従う）
4. 煙がある場合：濡れタオルで鼻・口を覆い、低い姿勢（はって）移動
5. 屋外の指定集合場所に集合し、人数確認
6. 消防署通報：119
""",
}

# ════════════════════════════════════════
# RAG: 시설 검색 + 규정 매칭
# ════════════════════════════════════════
GU_NAMES = ["강남구","강서구","마포구","서초구","성동구","송파구","영등포구","용산구","종로구","중구"]
UPJONG_MAP = {
    "민박": "외국인관광도시민박업",
    "관광숙박": "관광숙박업",
    "숙박": "숙박업",
    "호텔": "관광숙박업",
    "외국인": "외국인관광도시민박업",
}

def retrieve_facilities(query: str, n: int = 5) -> str:
    """쿼리에서 구/업종 키워드 추출 → 위험도 높은 시설 반환"""
    df = FACILITY_DF.copy()

    # 구 필터
    for gu in GU_NAMES:
        if gu in query or gu.replace("구", "") in query:
            df = df[df["구"] == gu]
            break

    # 업종 필터
    for kw, upjong in UPJONG_MAP.items():
        if kw in query:
            df = df[df["업종"] == upjong]
            break

    # 업소명 키워드
    name_kw = re.findall(r"[가-힣]{2,}", query)
    for kw in name_kw:
        if kw not in GU_NAMES and kw not in ["위험도", "소방", "화재", "대피", "설비", "스프링클러"]:
            match = df[df["업소명"].str.contains(kw, na=False)]
            if len(match):
                df = match
                break

    top = df.nlargest(n, "위험점수_AHP")[
        ["구", "업소명", "업종", "위험점수_AHP", "이동시간초", "공식도로폭m", "반경100m_화재수"]
    ]

    if top.empty:
        return "조건에 맞는 시설이 없습니다."

    lines = ["[검색된 고위험 시설]"]
    for _, r in top.iterrows():
        lines.append(
            f"• {r['구']} {r['업소명']} ({r['업종']})\n"
            f"  AHP위험점수={r['위험점수_AHP']:.1f}, 소방도달={r['이동시간초']:.0f}초, "
            f"도로폭={r['공식도로폭m']:.1f}m, 반경화재수={int(r['반경100m_화재수'])}건"
        )
    return "\n".join(lines)


def retrieve_fire_regulations(query: str) -> str:
    """연면적·층수·업종 키워드가 있으면 소방법 규정 섹션 반환"""
    keywords = ["스프링클러", "소화전", "소화기", "탐지", "감지기", "비상방송",
                "완강기", "피난", "연면적", "층수", "소방설비", "소방시설", "설비"]
    if any(k in query for k in keywords) or re.search(r"\d+m²|\d+층|\d+평", query):
        return FIRE_LAW_KB
    return ""


def retrieve_evacuation(query: str, lang_code: str) -> str:
    """대피 관련 질문이면 해당 언어 매뉴얼 반환"""
    keywords = ["대피", "피난", "evacuati", "避难", "避難", "逃", "escape",
                "경보", "알람", "연기", "불"]
    if any(k in query.lower() for k in keywords):
        return EVACUATION_GUIDE.get(lang_code, EVACUATION_GUIDE["ko"])
    return ""


def build_retrieval_context(query: str, lang_code: str) -> str:
    """쿼리에서 관련 컨텍스트 자동 수집 (RAG Retrieval)"""
    parts = []

    fac = retrieve_facilities(query)
    if fac and "없습니다" not in fac:
        parts.append(fac)

    reg = retrieve_fire_regulations(query)
    if reg:
        parts.append(reg)

    evac = retrieve_evacuation(query, lang_code)
    if evac:
        parts.append(evac)

    # 기본 통계 요약 (항상 포함)
    parts.append(
        f"[프로젝트 기본 통계]\n"
        f"총 {len(FACILITY_DF):,}개 시설, 서울 10개구, "
        f"평균 AHP위험점수={FACILITY_DF['위험점수_AHP'].mean():.1f}, "
        f"최고위험={FACILITY_DF['위험점수_AHP'].max():.1f}, "
        f"평균 소방도달={FACILITY_DF['이동시간초'].mean():.0f}초"
    )

    return "\n\n".join(parts)


# ════════════════════════════════════════
# 시스템 프롬프트 빌더
# ════════════════════════════════════════
LANG_INST = {
    "ko": "모든 답변을 한국어로 작성하세요.",
    "en": "Answer entirely in English.",
    "zh": "请用中文回答所有问题。",
    "ja": "すべての回答を日本語で書いてください。",
}

BASE_SYSTEM = """당신은 서울 숙박시설 화재안전 전문 AI 어시스턴트입니다.
다음 역할을 수행합니다:
1. 고위험 숙박시설 정보 조회 및 설명
2. 연면적·층수 기반 소방설비 설치 기준 안내 (한국 소방시설법)
3. 비상대피 절차 안내 (다국어 지원)
4. 분석 방법론(GWR·공간회귀·클러스터) 질문 답변

답변 원칙:
- 핵심을 먼저, 근거는 간결하게
- 숫자는 항상 단위 포함
- 개별 시설 주소/상세 정보는 보안상 제공 불가
- 모르는 것은 모른다고 솔직하게
{lang_inst}
"""


def make_system_prompt(lang_code: str, context: str) -> str:
    system = BASE_SYSTEM.format(lang_inst=LANG_INST[lang_code])
    system += f"\n--- 검색된 관련 데이터 ---\n{context}\n--- 데이터 끝 ---"
    return system


# ════════════════════════════════════════
# LLM 호출
# ════════════════════════════════════════
def call_llm(system_prompt: str, messages: list, model: str, api_key: str, is_claude: bool) -> str:
    if is_claude:
        import anthropic as _ant
        client = _ant.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model, max_tokens=1200, system=system_prompt, messages=messages
        )
        return resp.content[0].text
    else:
        from openai import OpenAI as _OAI
        client = _OAI(api_key=api_key)
        oai_msgs = [{"role": "system", "content": system_prompt}] + messages
        resp = client.chat.completions.create(model=model, max_tokens=1200, messages=oai_msgs)
        return resp.choices[0].message.content


# ════════════════════════════════════════
# UI
# ════════════════════════════════════════
st.title("🔥 화재안전 RAG 챗봇")
flag = {"한국어": "🇰🇷", "English": "🇺🇸", "中文": "🇨🇳", "日本語": "🇯🇵"}[lang]
st.caption(f"응답 언어: {flag} {lang}  |  위험 시설 검색 · 소방설비 안내 · 비상대피 가이드")

if not api_key:
    st.info("👈 사이드바에서 API Key를 입력하거나 secrets.toml에 추가하세요.")
    st.stop()

# ── 빠른 질문 버튼 ──
st.markdown("**빠른 질문**")
quick_qs_map = {
    "한국어": [
        "마포구에서 가장 위험한 숙박시설 5곳을 알려줘",
        "연면적 800m² 5층 숙박업에 필요한 소방설비는?",
        "화재 시 비상대피 방법을 알려줘",
        "스프링클러는 언제 의무야?",
        "GWR 분석 결과 어떤 구가 위험해?",
    ],
    "English": [
        "Show me the top 5 highest-risk lodging facilities",
        "What fire systems are required for 800m² 5-floor hotel?",
        "Explain the fire evacuation procedure",
        "When is a sprinkler system mandatory?",
        "Which district is most dangerous based on GWR?",
    ],
    "中文": [
        "请显示风险最高的5家住宿设施",
        "800平米5层酒店需要什么消防设备？",
        "请说明火灾紧急疏散程序",
        "什么情况下必须安装自动喷水灭火系统？",
        "根据GWR分析，哪个区最危险？",
    ],
    "日本語": [
        "危険度の高い宿泊施設トップ5を教えて",
        "800m²・5階建て宿泊施設に必要な消防設備は？",
        "火災時の緊急避難方法を教えてください",
        "スプリンクラーはいつ義務になりますか？",
        "GWR分析でどの区が最も危険ですか？",
    ],
}
quick_qs = quick_qs_map.get(lang, quick_qs_map["한국어"])

btn_cols = st.columns(len(quick_qs))
for col, q in zip(btn_cols, quick_qs):
    if col.button(q[:20] + "…" if len(q) > 20 else q, use_container_width=True, help=q):
        st.session_state.setdefault("messages", [])
        st.session_state.messages.append({"role": "user", "content": q})

st.markdown("---")

# ── 대화 초기화 ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 히스토리 출력 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("context_used"):
            st.markdown(
                f"<div class='rag-badge'>📎 검색 컨텍스트 사용됨 — {msg['context_used']}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])

# ── 사용자 입력 ──
if prompt := st.chat_input({"한국어": "질문을 입력하세요…", "English": "Ask a question…",
                             "中文": "请输入问题…", "日本語": "質問を入力してください…"}[lang]):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# ── 응답 생성 ──
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_q = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        try:
            # RAG: 컨텍스트 검색
            context = build_retrieval_context(last_q, lang_code)
            system  = make_system_prompt(lang_code, context)

            # 히스토리 (최근 16턴)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-16:]
                if m["role"] in ("user", "assistant")
            ]

            # 컨텍스트 배지 생성 (어떤 데이터 사용했는지)
            used_labels = []
            if "검색된 고위험" in context:  used_labels.append("시설 DB")
            if "소방시설법" in context:      used_labels.append("소방법 규정")
            if "비상대피" in context or "Evacuation" in context or "疏散" in context or "避難" in context:
                used_labels.append("대피 매뉴얼")
            ctx_label = " · ".join(used_labels) if used_labels else "기본 통계"

            with st.spinner("검색 중…"):
                answer = call_llm(system, history, model_name, api_key, is_claude)

            st.markdown(
                f"<div class='rag-badge'>📎 참조 데이터: {ctx_label}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "context_used": ctx_label,
            })

        except Exception as e:
            err = str(e)
            if "auth" in err.lower() or "401" in err or "invalid_api_key" in err.lower():
                st.error("API 키가 올바르지 않습니다.")
            else:
                st.error(f"오류: {e}")

# ── 하단 초기화 ──
if st.session_state.get("messages"):
    st.markdown("---")
    if st.button("대화 초기화", type="secondary"):
        st.session_state.messages = []
        st.rerun()
