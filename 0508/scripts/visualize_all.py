# -*- coding: utf-8 -*-
"""
통합숙박시설 종합 EDA 시각화 스크립트 (PNG 12장 + HTML 1장).

목적:
    - 통합숙박시설_최종안0421.csv 로부터 13개 분석 차트를 일괄 생성해 발표/보고용 자료 제작.
    - 위치 산포 / 건물나이 / 연도별 추이 / 골든타임 / PCA-AHP / 군집 레이더 / 상관 / Hexbin /
      구별 위험점수 / 시설군 / 층수 / 히트맵(HTML) / 연면적 분포 등 종합 패널.

입력:
    - data/통합숙박시설_최종안0421.csv (위도/경도/구/건물나이/위험점수 등 통합 변수)

출력:
    - data/viz_all/01~13_*.png + 12_위험점수_히트맵.html (총 13개)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns  # 히트맵 시각화에 사용
import folium  # 인터랙티브 지도
from folium.plugins import HeatMap  # 위험점수 히트맵 레이어
import sys
import os

# 콘솔 한글 출력 깨짐 방지
sys.stdout.reconfigure(encoding="utf-8")

# 시스템 '맑은 고딕' 자동 탐색 후 등록
for font in fm.findSystemFonts():
    if "malgun" in font.lower():
        plt.rcParams["font.family"] = fm.FontProperties(fname=font).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

BASE = "c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data"
OUT = f"{BASE}/viz_all"
os.makedirs(OUT, exist_ok=True)

# 통합 마스터 로드
df = pd.read_csv(f"{BASE}/통합숙박시설_최종안0421.csv", encoding="utf-8-sig")
GU_LIST = sorted(df["구"].unique())
# 자치구별 색상 — tab10 팔레트에서 알파벳 순으로 매핑
GU_COLORS = dict(zip(GU_LIST, plt.cm.tab10.colors[: len(GU_LIST)]))
print(f"로드: {len(df)}행")

# ══════════════════════════════════════════════════════════════
# 1. 위치 산포도 — 구별 색상
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 10))
# 자치구별로 점 색깔/라벨 구분 — 작게(s=8) 알파 0.6 으로 겹침 시 농도 표현
for gu in GU_LIST:
    sub = df[df["구"] == gu]
    ax.scatter(sub["경도"], sub["위도"], c=[GU_COLORS[gu]], s=8, alpha=0.6, label=gu)
ax.set_title("서울 10개 구 숙박시설 위치 분포", fontsize=14, fontweight="bold")
ax.set_xlabel("경도")
ax.set_ylabel("위도")
ax.legend(loc="lower right", fontsize=8, ncol=2, markerscale=2)
ax.set_facecolor("#f0f0f0")  # 옅은 회색 배경 — 점 가독성 향상
plt.tight_layout()
plt.savefig(f"{OUT}/01_위치_산포도.png", dpi=150, bbox_inches="tight")
plt.close()
print("01 완료")

# ══════════════════════════════════════════════════════════════
# 2. 건물나이 바이올린 플롯 — 구별
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))
# 구별 건물나이 배열 리스트 — violinplot 입력
data_by_gu = [df[df["구"] == gu]["건물나이"].dropna().values for gu in GU_LIST]
vp = ax.violinplot(
    data_by_gu, positions=range(len(GU_LIST)), showmedians=True, showextrema=True
)
# 바이올린 본체 색상 — 자치구 색에 맞춰 채움
for i, body in enumerate(vp["bodies"]):
    body.set_facecolor(list(GU_COLORS.values())[i])
    body.set_alpha(0.75)
# 중앙값선만 흰색 굵게 강조
vp["cmedians"].set_color("white")
vp["cmedians"].set_linewidth(2)
ax.set_xticks(range(len(GU_LIST)))
ax.set_xticklabels(GU_LIST, rotation=30)
ax.set_ylabel("건물나이 (년)")
ax.set_title("구별 숙박시설 건물나이 분포", fontsize=14, fontweight="bold")
ax.set_facecolor("#fafafa")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/02_건물나이_바이올린.png", dpi=150, bbox_inches="tight")
plt.close()
print("02 완료")

# ══════════════════════════════════════════════════════════════
# 3. 연도별 신규 숙박시설 — 구별 스택 영역 차트
# ══════════════════════════════════════════════════════════════
# 2000년 이후 승인 시설만 — 연도×구 피벗 (결측은 0)
yearly = (
    df[df["승인연도"] >= 2000].groupby(["승인연도", "구"]).size().unstack(fill_value=0)
)
fig, ax = plt.subplots(figsize=(14, 6))
# 누적 영역 차트로 자치구 기여도 시각화
yearly.plot.area(ax=ax, alpha=0.8, colormap="tab10")
ax.set_title("연도별 신규 숙박시설 수 (2000~)", fontsize=14, fontweight="bold")
ax.set_xlabel("승인연도")
ax.set_ylabel("신규 숙박시설 수")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/03_연도별_신규_스택.png", dpi=150, bbox_inches="tight")
plt.close()
print("03 완료")

# ══════════════════════════════════════════════════════════════
# 4. 소방 골든타임 분석 — 도로폭 보정 예상도착시간
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("소방 골든타임 분석 (5분=300초 기준)", fontsize=14, fontweight="bold")

golden = 300  # 5분 = 300초 — 화재 골든타임
# 자치구별 골든타임 초과율(%) — 내림차순 정렬
gu_over = (
    df.groupby("구")
    .apply(lambda x: (x["도로폭_보정예상도착초"] > golden).mean() * 100)
    .sort_values(ascending=False)
)
# 초과율에 따른 위험 신호등 — 빨강(>50)/주황(>30)/초록
colors = [
    "#e74c3c" if v > 50 else "#e67e22" if v > 30 else "#2ecc71" for v in gu_over.values
]
bars = axes[0].bar(gu_over.index, gu_over.values, color=colors)
# 막대 위에 % 라벨
for bar, v in zip(bars, gu_over.values):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{v:.0f}%",
        ha="center",
        fontsize=9,
    )
# 50% 기준선 — 위험 임계
axes[0].axhline(50, color="red", lw=1.5, linestyle="--", label="50% 기준선")
axes[0].set_xlabel("구")
axes[0].set_ylabel("골든타임 초과율 (%)")
axes[0].set_title("구별 골든타임(5분) 초과 비율")
axes[0].legend()
axes[0].tick_params(axis="x", rotation=30)

# 우측 패널 — 자치구별 도착시간 박스플롯
data_box = [
    df[df["구"] == gu]["도로폭_보정예상도착초"].dropna().values for gu in GU_LIST
]
bp = axes[1].boxplot(data_box, labels=GU_LIST, patch_artist=True, notch=True)
for patch, gu in zip(bp["boxes"], GU_LIST):
    patch.set_facecolor(GU_COLORS[gu])
    patch.set_alpha(0.7)
# 골든타임 300초 수평선 — 박스가 이 위에 있으면 위험
axes[1].axhline(golden, color="red", lw=1.5, linestyle="--", label="골든타임 300초")
axes[1].set_ylabel("예상도착시간 (초)")
axes[1].set_title("구별 예상도착시간 분포")
axes[1].legend()
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(f"{OUT}/04_소방_골든타임.png", dpi=150, bbox_inches="tight")
plt.close()
print("04 완료")

# ══════════════════════════════════════════════════════════════
# 5. AHP vs PCA 산점도 — 군집 색상
# ══════════════════════════════════════════════════════════════
df_s = df.dropna(subset=["위험점수_PCA", "위험점수_AHP"])
fig, ax = plt.subplots(figsize=(10, 8))
clusters = sorted(df_s["군집"].unique())
cmap = plt.cm.Set1
# 군집별로 색상 다르게 하여 PCA-AHP 위험점수 산점도
for cl in clusters:
    sub = df_s[df_s["군집"] == cl]
    ax.scatter(
        sub["위험점수_PCA"],
        sub["위험점수_AHP"],
        c=[cmap(cl / max(clusters))],
        s=15,
        alpha=0.5,
        label=f"군집 {cl}",
    )
ax.set_xlabel("위험점수_PCA", fontsize=11)
ax.set_ylabel("위험점수_AHP", fontsize=11)
ax.set_title("PCA vs AHP 위험점수 — 군집별 분포", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, markerscale=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/05_PCA_AHP_산점도.png", dpi=150, bbox_inches="tight")
plt.close()
print("05 완료")

# ══════════════════════════════════════════════════════════════
# 6. 군집별 레이더 차트 — 6개 변수 각도별 비교
# ══════════════════════════════════════════════════════════════
# 레이더에 사용할 원본 변수명
radar_vars = [
    "건물나이",
    "반경_50m_건물수",
    "집중도(%)",
    "구조_노후_통합점수",
    "도로폭_위험도",
    "소방위험도_점수",
]
# 차트에 표시할 짧은 라벨
radar_labels = [
    "건물나이",
    "주변건물수",
    "집중도",
    "구조노후도",
    "도로폭위험도",
    "소방위험도",
]
N = len(radar_vars)
# 6개 축의 각도 + 마지막에 첫 각도 한 번 더 추가(폐곡선 형성)
angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

# 군집 수만큼 가로로 polar subplot 나열
fig, axes = plt.subplots(
    1, len(clusters), figsize=(5 * len(clusters), 5), subplot_kw=dict(polar=True)
)
fig.suptitle("군집별 특성 레이더 차트", fontsize=14, fontweight="bold")
if len(clusters) == 1:
    axes = [axes]  # subplot 이 1개면 axes 가 단일 객체 → 리스트화

# 변수별 0~1 정규화 (군집 비교를 위함)
norm_df = df.copy()
for v in radar_vars:
    mn, mx = df[v].min(), df[v].max()
    norm_df[v] = (df[v] - mn) / (mx - mn + 1e-9)  # 0 나눗셈 방지

for ax, cl in zip(axes, clusters):
    # 군집의 각 변수 평균 + 마지막에 첫 값을 한 번 더(폐곡선)
    vals = norm_df[norm_df["군집"] == cl][radar_vars].mean().tolist() + [
        norm_df[norm_df["군집"] == cl][radar_vars].mean().tolist()[0]
    ]
    ax.plot(angles, vals, "o-", linewidth=2, color=cmap(cl / max(clusters)))
    ax.fill(angles, vals, alpha=0.25, color=cmap(cl / max(clusters)))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=9)
    ax.set_title(
        f"군집 {cl}\n(N={len(df[df['군집'] == cl])})",
        fontsize=11,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(f"{OUT}/06_군집_레이더.png", dpi=150, bbox_inches="tight")
plt.close()
print("06 완료")

# ══════════════════════════════════════════════════════════════
# 7. 상관 히트맵 — 9개 핵심 변수
# ══════════════════════════════════════════════════════════════
corr_vars = [
    "건물나이",
    "반경_50m_건물수",
    "집중도(%)",
    "구조_노후_통합점수",
    "도로폭_위험도",
    "소방위험도_점수",
    "위험점수_AHP",
    "안전센터_유클리드m",
    "상업비율(%)",
]
corr_labels = [
    "건물나이",
    "주변건물수",
    "집중도",
    "구조노후도",
    "도로폭위험도",
    "소방위험도",
    "AHP위험점수",
    "안전센터거리",
    "상업비율",
]
# 상관 행렬 계산 후 라벨 깔끔하게 교체
corr = df[corr_vars].corr()
corr.index = corr.columns = corr_labels

fig, ax = plt.subplots(figsize=(11, 9))
# 상삼각 마스크 — 중복 표기 방지(대칭 행렬)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    mask=mask,
    ax=ax,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",  # 빨강(+)/파랑(-) 발산형 컬러맵
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    annot_kws={"size": 10},
    cbar_kws={"shrink": 0.8},
)
ax.set_title("주요 변수 상관 히트맵", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/07_상관_히트맵.png", dpi=150, bbox_inches="tight")
plt.close()
print("07 완료")

# ══════════════════════════════════════════════════════════════
# 8. 건물나이 vs 구조노후도 — Hexbin (점밀도 시각화)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
# Hexbin — 산점도 너무 많을 때 6각형 셀 안에 점 개수로 색상 표현
hb = ax.hexbin(
    df["건물나이"], df["구조_노후_통합점수"], gridsize=40, cmap="YlOrRd", mincnt=1
)
plt.colorbar(hb, ax=ax, label="숙소 수")
ax.set_xlabel("건물나이 (년)", fontsize=11)
ax.set_ylabel("구조_노후_통합점수", fontsize=11)
ax.set_title("건물나이 vs 구조노후도 (Hexbin)", fontsize=14, fontweight="bold")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUT}/08_건물나이_구조노후도_hexbin.png", dpi=150, bbox_inches="tight")
plt.close()
print("08 완료")

# ══════════════════════════════════════════════════════════════
# 9. 구별 평균 위험점수_AHP — 수평 barplot
# ══════════════════════════════════════════════════════════════
gu_ahp = df.groupby("구")["위험점수_AHP"].mean().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
# 막대 색상을 RdYlGn_r 컬러맵에 매핑 — 빨강(위험)→초록(안전)
norm = mcolors.Normalize(vmin=gu_ahp.min(), vmax=gu_ahp.max())
colors = [cm.RdYlGn_r(norm(v)) for v in gu_ahp.values]
bars = ax.barh(gu_ahp.index, gu_ahp.values, color=colors)
# 막대 끝 점수 텍스트
for bar, v in zip(bars, gu_ahp.values):
    ax.text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{v:.1f}",
        va="center",
        fontsize=9,
    )
ax.set_xlabel("평균 AHP 위험점수")
ax.set_title("구별 평균 AHP 위험점수\n(높을수록 위험)", fontsize=14, fontweight="bold")
# 컬러바 — 위험점수 매핑 가이드
sm = cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
plt.colorbar(sm, ax=ax, label="위험점수")
plt.tight_layout()
plt.savefig(f"{OUT}/09_구별_AHP위험점수.png", dpi=150, bbox_inches="tight")
plt.close()
print("09 완료")

# ══════════════════════════════════════════════════════════════
# 10. 주요_시설군 구별 누적 barplot
# ══════════════════════════════════════════════════════════════
# 구×시설군 피벗 — 각 칸은 숙소 수
sil_pivot = df.groupby(["구", "주요_시설군"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(13, 6))
# 누적 막대 — 자치구별 시설군 구성비 비교 가능
sil_pivot.plot.bar(ax=ax, stacked=True, colormap="Set2", alpha=0.9)
ax.set_title("구별 숙박시설 주변 시설군 구성", fontsize=14, fontweight="bold")
ax.set_xlabel("구")
ax.set_ylabel("숙소 수")
# 범례를 차트 우측 외부에 배치 — 막대 가독성 보존
ax.legend(title="주요시설군", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(f"{OUT}/10_시설군_구성.png", dpi=150, bbox_inches="tight")
plt.close()
print("10 완료")

# ══════════════════════════════════════════════════════════════
# 11. 층수 분포 — 구별 boxplot (30층 이하)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 6))
# 30층 초과 이상치는 제외 — 시각화 가독성 (대다수 시설은 10층 이하)
df_floor = df[df["총층수"] <= 30]
data_floor = [
    df_floor[df_floor["구"] == gu]["총층수"].dropna().values for gu in GU_LIST
]
bp = ax.boxplot(data_floor, labels=GU_LIST, patch_artist=True, notch=False)
for patch, gu in zip(bp["boxes"], GU_LIST):
    patch.set_facecolor(GU_COLORS[gu])
    patch.set_alpha(0.8)
ax.set_ylabel("총층수")
ax.set_title("구별 숙박시설 층수 분포 (30층 이하)", fontsize=14, fontweight="bold")
ax.tick_params(axis="x", rotation=30)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/11_층수_분포.png", dpi=150, bbox_inches="tight")
plt.close()
print("11 완료")

# ══════════════════════════════════════════════════════════════
# 12. Folium 히트맵 — 위험점수_AHP (인터랙티브 HTML)
# ══════════════════════════════════════════════════════════════
# 다크 매터 타일 — 야간 화재 위험 분위기 강조
m = folium.Map(location=[37.555, 126.985], zoom_start=12, tiles="CartoDB dark_matter")
heat_data = df[["위도", "경도", "위험점수_AHP"]].dropna().values.tolist()
HeatMap(
    heat_data,
    radius=12,
    blur=15,
    max_zoom=14,
    # 위험도 그라디언트 — 파(저) → 라임 → 주황 → 빨강(고)
    gradient={0.2: "blue", 0.5: "lime", 0.8: "orange", 1.0: "red"},
).add_to(m)
folium.LayerControl().add_to(m)
m.save(f"{OUT}/12_위험점수_히트맵.html")
print("12 완료 (HTML)")

# ══════════════════════════════════════════════════════════════
# 13. 연면적 분포 — 로그스케일 히스토그램
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5))
# 0 이하 제거 — log10 정의역
df_area = df[df["연면적(㎡)"] > 0]["연면적(㎡)"]
ax.hist(np.log10(df_area), bins=60, color="#3498db", edgecolor="white", alpha=0.85)
ax.set_xlabel("log₁₀(연면적, ㎡)")
ax.set_ylabel("빈도")
ax.set_title("숙박시설 연면적 분포 (로그스케일)", fontsize=14, fontweight="bold")
# x축 눈금을 10^t 표기로 — 사람 읽기 쉬운 면적값 병기
ticks = [1, 2, 3, 4, 5]
ax.set_xticks(ticks)
ax.set_xticklabels([f"10^{t}\n({10**t:,}㎡)" for t in ticks])
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/13_연면적_분포.png", dpi=150, bbox_inches="tight")
plt.close()
print("13 완료")

print(f"\n모든 시각화 완료 → {OUT}")
