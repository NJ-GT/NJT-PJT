# -*- coding: utf-8 -*-
"""
외국인 방문자 수 추이 — 서울 10개 구 연도별 평균 라인 차트.

목적:
    같은 폴더의 CSV(외국인 방문자 데이터)에서 분석 대상 10개 구를 추려,
    연도별 평균 외국인 방문자 수를 라인 차트로 시각화한다.

주요 처리:
    - 한글 폰트 자동 등록(맑은고딕/나눔고딕/유니코드 MS 중 하나)
    - 연도×지역 피벗 후 우리 분석 대상 10개 구만 유지
    - 2025년 값 기준 내림차순으로 범례 정렬
    - 마커/색상은 구별 개별 지정, 라인엔 그림자 효과로 미적 강조
"""

import os
from pathlib import Path

# matplotlib 백엔드 강제 (PNG 저장 전용)
import matplotlib

matplotlib.use("Agg")

# 라인 그림자 효과
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
# y축 단위(천 단위) 포매팅 등을 위해 ticker 사용
import matplotlib.ticker as mticker
import pandas as pd
# 한글 폰트 등록을 위한 font_manager
from matplotlib import font_manager, rcParams


# 현재 스크립트 폴더(NJT-PJT/0502/)
HERE = Path(__file__).resolve().parent
# 폴더 내 첫 CSV를 입력으로 자동 선택 (단일 CSV 가정)
CSV_PATH = next(HERE.glob("*.csv"))
# 결과 PNG 저장 경로
OUT_PATH = HERE / "foreign_visitors_10gu_yearly_average_line.png"

# 한글 표시용 폰트 후보 — 환경에 설치된 첫 번째를 사용
font_candidates = ["Malgun Gothic", "NanumGothic", "Arial Unicode MS"]
available_fonts = {font.name for font in font_manager.fontManager.ttflist}
for font in font_candidates:
    if font in available_fonts:
        rcParams["font.family"] = font
        break
# 음수 부호 깨짐 방지
rcParams["axes.unicode_minus"] = False

# 분석 대상 10개 구 (시각화에 포함될 구만 필터)
our_gu = [
    "강남구",
    "강서구",
    "마포구",
    "서초구",
    "성동구",
    "송파구",
    "영등포구",
    "용산구",
    "종로구",
    "중구",
]

# CSV 로드 (인코딩은 자동 — 보통 utf-8)
df = pd.read_csv(CSV_PATH)
# 날짜는 정수 연도 — Int64로 변환 (NaN 허용)
df["날짜"] = pd.to_numeric(df["날짜"], errors="coerce").astype("Int64")
# 외국인 방문자 수는 부동소수 변환
df["외국인 방문자수"] = pd.to_numeric(
    df["외국인 방문자수"], errors="coerce"
)

# 분석 대상 10개 구만 추리고, (연도, 지역) 기준 평균을 구한 뒤 피벗
plot_df = (
    df[df["지역"].isin(our_gu)]
    .groupby(["날짜", "지역"], as_index=False)[
        "외국인 방문자수"
    ]
    .mean()
    .pivot(
        index="날짜",
        columns="지역",
        values="외국인 방문자수",
    )
    # 컬럼 순서를 our_gu 순서로 통일
    .reindex(columns=our_gu)
    .sort_index()
)

# 정수 연도 리스트 (x축 틱)
years = plot_df.index.astype(int).tolist()
# 범례 정렬 — 가장 최근(2025년) 값이 큰 구가 상단에 오도록
legend_order = plot_df.loc[2025].sort_values(ascending=False).index.tolist()
# 구별 색상 매핑 (지정 팔레트)
colors = {
    "강남구": "#ff7f50",
    "강서구": "#9bd35a",
    "마포구": "#f58bdc",
    "서초구": "#ff6b7e",
    "성동구": "#f5a85a",
    "송파구": "#2f80ed",
    "영등포구": "#02c875",
    "용산구": "#ff7fa3",
    "종로구": "#50f050",
    "중구": "#ffc247",
}
# 구별 마커 — 시각적 식별성 ↑
markers = ["*", "s", "h", "X", "P", "^", "o", "8", "<", "D"]

# 차트 시작 — 가로로 길고 좌우 여백을 둠 (범례를 우측 외부로 뺌)
fig, ax = plt.subplots(figsize=(15.5, 6.4), dpi=160)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# 구별 라인 + 마커 그리기
for index, gu in enumerate(our_gu):
    y = plot_df[gu]
    (line,) = ax.plot(
        years,
        y,
        label=gu,
        color=colors[gu],
        linewidth=2.0,
        # 마커는 markers 리스트에서 순환 사용
        marker=markers[index % len(markers)],
        markersize=10.5,
        markeredgewidth=0,
        zorder=3,
    )
    # 라인에 살짝 그림자 효과 — 입체감
    line.set_path_effects(
        [pe.SimpleLineShadow(offset=(1.8, -1.8), alpha=0.22, rho=0.95), pe.Normal()]
    )
    # 마커를 강조하기 위한 별도 scatter (마커 크기 키움)
    ax.scatter(
        years,
        y,
        s=118,
        color=colors[gu],
        marker=markers[index % len(markers)],
        edgecolors="none",
        zorder=4,
    )

# 축 범위/라벨 설정
ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
ax.set_ylim(0, max(plot_df.max()) * 1.15)
ax.set_xticks(years)
ax.tick_params(axis="x", labelrotation=35, labelsize=10)
ax.tick_params(axis="y", labelsize=10)
# y축 틱은 보기 좋은 6칸 안팎으로 자동 분할
ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
# y축 값을 'k' 단위 (천 단위) 라벨로 — 0은 그대로 0
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, pos: "0" if x == 0 else f"{int(x / 1000):,} k")
)

# 그리드/축 스타일 — 부드러운 회색 톤
ax.grid(True, which="major", color="#9a9a9a", alpha=0.32, linewidth=0.85)
for spine in ax.spines.values():
    spine.set_color("#d0d0d0")
    spine.set_linewidth(0.9)

# 제목
ax.set_title(
    "외국인 지역별 방문자 수 추이 - 10개구 연도별 평균",
    fontsize=16,
    pad=18,
    weight="bold",
)
ax.set_xlabel("")
ax.set_ylabel("")

# 범례를 legend_order 순서로 재배치 (2025년 기준 큰 값이 위로)
handles, labels = ax.get_legend_handles_labels()
handle_by_label = dict(zip(labels, handles))
ax.legend(
    [handle_by_label[label] for label in legend_order],
    legend_order,
    title="2025년 기준",
    loc="center left",
    # 범례를 차트 외부 오른쪽에 위치
    bbox_to_anchor=(1.01, 0.5),
    ncol=1,
    frameon=False,
    fontsize=10.5,
    title_fontsize=11,
    handlelength=2.2,
    handletextpad=0.5,
)

# 차트 영역과 범례 영역의 균형을 맞추기 위한 여백 직접 지정
plt.subplots_adjust(left=0.07, right=0.84, top=0.86, bottom=0.13)
# bbox_inches="tight" — 외부 범례까지 잘리지 않도록 자동 자르기
fig.savefig(OUT_PATH, bbox_inches="tight", facecolor="white")

# 결과 경로와 데이터 미리보기 출력
print(os.fspath(OUT_PATH))
print(plot_df.round(2).to_string())
