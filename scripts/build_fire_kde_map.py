# -*- coding: utf-8 -*-
"""
서울 10구 화재출동 KDE 밀도 지도 + 숙박시설 분포 시각화.

목적:
    화재출동 위치(위경도)에 가우시안 커널 밀도(KDE)를 적용하여
    화재가 자주 발생하는 지역을 컨투어로 표현하고,
    그 위에 숙박시설 위치를 겹쳐 시각적으로 비교한다.

산출:
    NJT-PJT/data/fire_kde_map.png
"""
import sys
import numpy as np
import pandas as pd
import matplotlib

# PNG 저장만 하므로 비-GUI Agg
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 알파(투명도) 그라디언트 컬러맵 생성용
import matplotlib.colors as mcolors
# 가우시안 커널 밀도 추정
from scipy.stats import gaussian_kde

# Windows 콘솔 한글 깨짐 방지
sys.stdout.reconfigure(encoding="utf-8")

# 입출력 경로
BASE = "c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT"
FIRE_PATH = f"{BASE}/data/화재출동/화재출동_2021_2024.csv"
ACC_PATH = f"{BASE}/data/서울10구_숙소_소방거리_유클리드.csv"
OUT_PATH = f"{BASE}/data/fire_kde_map.png"

# ── 데이터 로드 ───────────────────────────────────────────────────────
fire = pd.read_csv(FIRE_PATH, encoding="utf-8-sig", low_memory=False)
# 손상된 행 건너뛰기 (on_bad_lines="skip")
acc = pd.read_csv(ACC_PATH, encoding="utf-8-sig", on_bad_lines="skip")

# 분석 대상 13개 구 (KDE를 서울권으로 제한하기 위함)
target_gu = [
    "종로구",
    "중구",
    "용산구",
    "성동구",
    "광진구",
    "마포구",
    "서대문구",
    "은평구",
    "서초구",
    "강남구",
    "송파구",
    "강서구",
    "영등포구",
]
# 서울 위경도 박스 + 대상 구로 화재 데이터 필터
fire = fire[
    (fire["위도"] > 37.4)
    & (fire["위도"] < 37.7)
    & (fire["경도"] > 126.7)
    & (fire["경도"] < 127.3)
    & fire["발생시군구"].isin(target_gu)
].copy()
print(f"화재: {len(fire)}건 | 숙박시설: {len(acc)}개")

# ── KDE 계산 ───────────────────────────────────────────────────────────
# 화재 위치를 (2, N) 형태로 — gaussian_kde 입력 규격
xy = np.vstack([fire["경도"].values, fire["위도"].values])
# bw_method=0.03: 밴드위드 작은 편 -> 핫스팟 강조
kde = gaussian_kde(xy, bw_method=0.03)

# 평가 그리드 (서울 중심부)
lon_min, lon_max = 126.82, 127.18
lat_min, lat_max = 37.44, 37.65

# 300×300 해상도
GRID_N = 300
grid_lon, grid_lat = np.mgrid[
    lon_min : lon_max : GRID_N * 1j, lat_min : lat_max : GRID_N * 1j
]
# 그리드 좌표에서 KDE 값 평가 후 행렬로 reshape
kde_values = kde(np.vstack([grid_lon.ravel(), grid_lat.ravel()])).reshape(
    GRID_N, GRID_N
)
# 0~1 정규화 (시각화 컨투어 안정)
kde_norm = (kde_values - kde_values.min()) / (kde_values.max() - kde_values.min())
print("KDE 완료")

# ── 시각화 ────────────────────────────────────────────────────────────
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(14, 12), dpi=150)

# YlOrRd 팔레트의 하단 영역(낮은 KDE)을 투명하게 만들어 배경에 자연스럽게 깔리도록
cmap = plt.get_cmap("YlOrRd")
cmap_alpha = cmap(np.linspace(0, 1, 256))
# 가장 낮은 30 단계는 완전 투명
cmap_alpha[:30, 3] = 0
# 이후 30 단계는 서서히 보이게
cmap_alpha[30:60, 3] = np.linspace(0, 0.5, 30)
custom_cmap = mcolors.LinearSegmentedColormap.from_list("fire_alpha", cmap_alpha)
# 25단계 컨투어로 그라데이션
cf = ax.contourf(grid_lon, grid_lat, kde_norm, levels=25, cmap=custom_cmap, alpha=0.85)

# 숙박시설 위치를 점으로 표시 (파란색, 작은 점)
ax.scatter(
    acc["경도"],
    acc["위도"],
    s=6,
    c="#2980b9",
    alpha=0.5,
    zorder=3,
    label=f"숙박시설 ({len(acc):,}개)",
)

# 컬러바 — 그림 옆 중앙에 압축해 배치
cbar = fig.colorbar(cf, ax=ax, shrink=0.5, pad=0.02)
cbar.set_label("화재 발생 밀도 (KDE)", fontsize=11)

# 축/라벨/제목
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xlabel("경도", fontsize=11)
ax.set_ylabel("위도", fontsize=11)
ax.set_title(
    f"서울 10구 화재출동 KDE 밀도 + 숙박시설 분포\n(화재출동 {len(fire):,}건, 2017~2024)",
    fontsize=14,
    pad=15,
)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"[저장 완료] {OUT_PATH}")
