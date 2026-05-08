# -*- coding: utf-8 -*-
"""
EDA: 변수 간 상관계수 히트맵 + Pairplot + VIF(다중공선성) 검증.

대상 변수(원본 -> 라벨):
    소방위험도_점수 -> 소방위험도
    노후도_점수 -> 노후도
    반경_50m_건물수 -> 밀집도
    집중도(%) -> 집중도
    로그_주변대비_상대위험도_고유단속지점_50m -> 불법주정차
    공식도로폭m -> 도로폭(m)

산출:
    data/corr_heatmap.png  — 상관 히트맵
    data/pairplot.png      — pairplot
    data/vif_table.png     — VIF 표 (이미지)
"""
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
# VIF 계산기 (statsmodels)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Windows 콘솔 한글 깨짐 방지
sys.stdout.reconfigure(encoding="utf-8")

# 한글 폰트 + 음수 부호 깨짐 방지
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 입력 데이터
BASE = "c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data"
df = pd.read_csv(f"{BASE}/분석변수_테이블.csv", encoding="utf-8-sig")

# 분석 대상 변수 (원본 컬럼명) 와 표시용 라벨
VARS = [
    "소방위험도_점수",
    "노후도_점수",
    "반경_50m_건물수",
    "집중도(%)",
    "로그_주변대비_상대위험도_고유단속지점_50m",
    "공식도로폭m",
]
LABELS = ["소방위험도", "노후도", "밀집도", "집중도", "불법주정차", "도로폭(m)"]

# 표시용 라벨로 컬럼명 교체
data = df[VARS].copy()
data.columns = LABELS

# ── 1. 상관계수 히트맵 ────────────────────────────────────────────────
corr = data.corr()
fig, ax = plt.subplots(figsize=(9, 7))
# 상삼각만 마스킹 (대칭 행렬 가독성 ↑) — k=1 로 대각선 위 유지
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 10},
)
ax.set_title("변수 간 상관계수 히트맵", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(f"{BASE}/corr_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("[저장] corr_heatmap.png")

# ── 2. Pairplot ───────────────────────────────────────────────────────
# 대각선 KDE, 산점도 작은 점/투명도로 가독성 ↑
g = sns.pairplot(
    data,
    diag_kind="kde",
    plot_kws={"alpha": 0.3, "s": 5},
    diag_kws={"color": "#2980b9"},
)
g.figure.suptitle("Pairplot", y=1.01, fontsize=13)
g.figure.savefig(f"{BASE}/pairplot.png", dpi=120, bbox_inches="tight")
plt.close()
print("[저장] pairplot.png")

# ── 3. VIF (독립변수만) ───────────────────────────────────────────
# 종속변수(소방위험도)는 제외, 나머지 5개로 VIF 계산
X_vars = LABELS[1:]
X = data[X_vars].copy()
# VIF 계산을 위해 절편 컬럼 추가 (statsmodels 관습)
X.insert(0, "const", 1)

# 변수별 VIF 산출 — i+1로 const 컬럼 인덱스 건너뜀
vif_data = pd.DataFrame(
    {
        "변수": X_vars,
        "VIF": [variance_inflation_factor(X.values, i + 1) for i in range(len(X_vars))],
    }
).sort_values("VIF", ascending=False)
# 통상 임계 — 5 이상 주의, 10 이상 다중공선성 의심
vif_data["판정"] = vif_data["VIF"].apply(
    lambda v: "다중공선성 의심(>10)" if v > 10 else ("주의(5~10)" if v > 5 else "양호")
)

print("\n=== VIF 검증 결과 ===")
print(vif_data.to_string(index=False))

# VIF 결과를 표 형태 PNG로도 저장
fig, ax = plt.subplots(figsize=(6, 3))
ax.axis("off")
tbl = ax.table(
    cellText=vif_data[["변수", "VIF", "판정"]]
    .assign(VIF=vif_data["VIF"].round(3))
    .values,
    colLabels=["변수", "VIF", "판정"],
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.4, 1.8)
ax.set_title("VIF 다중공선성 검증", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(f"{BASE}/vif_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("[저장] vif_table.png")
