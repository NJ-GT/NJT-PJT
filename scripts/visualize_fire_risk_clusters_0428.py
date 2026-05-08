# -*- coding: utf-8 -*-
"""
사용자 정의 AHP 가중치 + KMeans(K=3) 으로 화재위험 군집을 만들어 4패널 PNG 로 시각화한다.

목적:
    - WEIGHTS 사전의 9개 변수를 MinMax 정규화 후 가중합으로 0~100 점수
    - 점수로 K=3 KMeans 군집 → 평균 점수 순으로 저/중/고위험군 라벨 재정렬
    - 4패널: 군집 규모+평균점수 / 변수 프로파일 히트맵 / 위치 산점 / TOP20 시설

입력:
    - NJT-PJT/0424/data/분석변수_최종테이블0428.csv (9개 변수 + 좌표 + 숙소명/구)

출력:
    - 0424/data/최종_화재위험_분석결과_0428.csv  (점수+군집 부착된 결과)
    - 0424/data/군집별_화재위험_시각화_0428.png  (4패널 카드)

처리 흐름:
    1) build_scored_data(): CSV 로딩 → MinMax → 가중합 점수 → KMeans(K=3) → 라벨 재정렬 → CSV 저장
    2) main(): 4패널 그리드 그리기 + 결과 텍스트 출력
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# K-평균 군집 + MinMax 정규화
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# 경로 상수
BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "0424" / "data" / "분석변수_최종테이블0428.csv"
OUT_CSV = BASE / "0424" / "data" / "최종_화재위험_분석결과_0428.csv"
OUT_PNG = BASE / "0424" / "data" / "군집별_화재위험_시각화_0428.png"

# AHP 사용자 정의 가중치 (합 = 1.00)
WEIGHTS = {
    "구조노후도": 0.24,
    "단속위험도": 0.16,
    "도로폭위험도": 0.14,
    "최근접_소화용수_거리등급": 0.12,
    "소방위험도_점수": 0.11,
    "연면적": 0.09,
    "집중도": 0.07,
    "주변건물수": 0.05,
    "총층수": 0.02,
}

# 히트맵 컬럼 라벨용 줄바꿈 매핑
FEATURE_LABELS = {
    "구조노후도": "구조\n노후",
    "단속위험도": "단속\n위험",
    "도로폭위험도": "도로폭\n위험",
    "최근접_소화용수_거리등급": "소화용수\n거리",
    "소방위험도_점수": "소방\n위험",
    "연면적": "연면적",
    "집중도": "집중도",
    "주변건물수": "주변\n건물",
    "총층수": "총층수",
}

# 군집 정렬 라벨 + 색
RISK_LABELS = ["저위험군", "중위험군", "고위험군"]
CLUSTER_COLORS = {
    "저위험군": "#16A34A",
    "중위험군": "#F59E0B",
    "고위험군": "#DC2626",
}


def build_scored_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """원본 CSV → 점수 + 군집 부착된 df + 정규화 df + 변수 리스트 반환.

    부수 효과:
        OUT_CSV (분석결과 csv) 를 저장한다.
    """
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    features = list(WEIGHTS.keys())

    # 필수 컬럼 검증
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # 9개 변수 숫자형 변환 + NaN → 0 + MinMax 정규화
    x = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(scaler.fit_transform(x), columns=features, index=df.index)

    # 가중합 → 0~100 스케일
    weights = pd.Series(WEIGHTS)
    df["최종_화재위험점수"] = (scaled[features] * weights).sum(axis=1) * 100

    # KMeans(K=3) 클러스터링 (정규화된 변수 그대로 사용)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    raw_cluster = kmeans.fit_predict(scaled[features])
    df["cluster"] = raw_cluster

    # 군집을 평균 화재위험점수 오름차순으로 정렬해 저→중→고 라벨 부여
    risk_order = (
        df.groupby("cluster")["최종_화재위험점수"].mean().sort_values().index.tolist()
    )
    risk_map = {cluster: label for cluster, label in zip(risk_order, RISK_LABELS)}
    df["위험군"] = df["cluster"].map(risk_map)

    # 정규화 df + 위험군 라벨 (히트맵용)
    scaled_with_cluster = scaled.copy()
    scaled_with_cluster["위험군"] = df["위험군"]
    # 점수+군집 부착된 결과 CSV 저장
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    return df, scaled_with_cluster, features


def main() -> None:
    """4패널 PNG 생성."""
    df, scaled, features = build_scored_data()

    # 군집 요약 (시설수/평균/중앙)
    summary = (
        df.groupby("위험군", observed=True)
        .agg(
            시설수=("숙소명", "size"),
            평균위험점수=("최종_화재위험점수", "mean"),
            중앙위험점수=("최종_화재위험점수", "median"),
        )
        .reindex(RISK_LABELS)
    )
    # 군집×변수 정규화 평균 (히트맵용)
    profile = (
        scaled.groupby("위험군", observed=True)[features].mean().reindex(RISK_LABELS)
    )
    # TOP 20 시설 (점수 내림차순)
    top20 = df.nlargest(20, "최종_화재위험점수").copy()

    # 한글 폰트
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 캔버스: 2×2 그리드, 좌상=count, 우상=heatmap, 좌하=map, 우하=top20
    fig = plt.figure(figsize=(18, 11), facecolor="#F8FAFC")
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.08, 1.22],
        height_ratios=[0.95, 1.05],
        hspace=0.34,
        wspace=0.22,
    )
    ax_count = fig.add_subplot(gs[0, 0], facecolor="white")
    ax_heat = fig.add_subplot(gs[0, 1], facecolor="white")
    ax_map = fig.add_subplot(gs[1, 0], facecolor="white")
    ax_top = fig.add_subplot(gs[1, 1], facecolor="white")

    # ── 좌상: 군집 규모(막대) + 평균 위험점수(라인, twinx) ─────────────
    colors = [CLUSTER_COLORS[label] for label in RISK_LABELS]
    x_pos = np.arange(len(RISK_LABELS))
    bars = ax_count.bar(
        x_pos,
        summary["시설수"],
        color=colors,
        width=0.58,
        edgecolor="white",
        linewidth=1.2,
    )
    # 평균 점수는 우측 y축
    ax_score = ax_count.twinx()
    ax_score.plot(
        x_pos,
        summary["평균위험점수"],
        color="#0F172A",
        marker="o",
        linewidth=2.4,
        markersize=7,
    )
    # 막대 위 시설수 / 라인 위 평균점수 라벨
    for i, (bar, score) in enumerate(zip(bars, summary["평균위험점수"])):
        ax_count.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 28,
            f"{int(bar.get_height()):,}개",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
        ax_score.text(
            i,
            score + 0.8,
            f"{score:.1f}점",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#0F172A",
            fontweight="bold",
        )
    ax_count.set_xticks(x_pos, RISK_LABELS, fontsize=12, fontweight="bold")
    ax_count.set_ylabel("시설수", color="#475569")
    ax_score.set_ylabel("평균 위험점수", color="#475569")
    ax_count.set_title(
        "군집 규모와 평균 위험점수", loc="left", fontsize=16, fontweight="bold", pad=14
    )
    ax_count.grid(axis="y", color="#E2E8F0")
    ax_count.set_axisbelow(True)
    ax_count.spines[["top", "right", "left"]].set_visible(False)
    ax_score.spines[["top", "left"]].set_visible(False)

    # ── 우상: 군집×변수 정규화 평균 히트맵 ───────────────────────────
    im = ax_heat.imshow(
        profile.to_numpy(), cmap="YlOrRd", vmin=0, vmax=1, aspect="auto"
    )
    ax_heat.set_xticks(
        np.arange(len(features)), [FEATURE_LABELS[col] for col in features], fontsize=10
    )
    ax_heat.set_yticks(
        np.arange(len(RISK_LABELS)), RISK_LABELS, fontsize=12, fontweight="bold"
    )
    ax_heat.set_title(
        "군집별 변수 프로파일", loc="left", fontsize=16, fontweight="bold", pad=14
    )
    # 셀에 숫자 라벨 (밝기에 따라 글자색 자동 변경)
    for y in range(profile.shape[0]):
        for x in range(profile.shape[1]):
            val = profile.iloc[y, x]
            ax_heat.text(
                x,
                y,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if val > 0.52 else "#0F172A",
                fontweight="bold",
            )
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.02)
    cbar.set_label("정규화 평균", color="#475569")

    # ── 좌하: 군집별 위치 산점도 (고위험군 강조) ─────────────────────
    sample = df.copy()
    for label in RISK_LABELS:
        sub = sample[sample["위험군"].eq(label)]
        ax_map.scatter(
            sub["경도"],
            sub["위도"],
            s=14 if label != "고위험군" else 22,
            c=CLUSTER_COLORS[label],
            alpha=0.55 if label != "고위험군" else 0.78,
            label=f"{label} ({len(sub):,})",
            linewidths=0,
        )
    ax_map.set_title(
        "숙소 위치별 군집 분포", loc="left", fontsize=16, fontweight="bold", pad=14
    )
    ax_map.set_xlabel("경도", color="#475569")
    ax_map.set_ylabel("위도", color="#475569")
    ax_map.grid(color="#E2E8F0")
    ax_map.legend(frameon=False, loc="lower left", ncols=1, fontsize=10)
    ax_map.spines[["top", "right"]].set_visible(False)

    # ── 우하: TOP 20 시설 가로 막대 (점수 + 구) ──────────────────────
    top20 = top20.sort_values("최종_화재위험점수", ascending=True)
    top_labels = top20["숙소명"].str.slice(0, 18)
    top_colors = top20["위험군"].map(CLUSTER_COLORS)
    ax_top.barh(
        top_labels,
        top20["최종_화재위험점수"],
        color=top_colors,
        edgecolor="white",
        linewidth=1.0,
    )
    # 막대 끝에 점수 + 구명 라벨
    for y, score, gu in zip(range(len(top20)), top20["최종_화재위험점수"], top20["구"]):
        ax_top.text(
            score + 0.4,
            y,
            f"{score:.1f} | {gu}",
            va="center",
            ha="left",
            fontsize=9,
            color="#0F172A",
            fontweight="bold",
        )
    ax_top.set_title(
        "최종 화재 위험 시설 TOP 20", loc="left", fontsize=16, fontweight="bold", pad=14
    )
    ax_top.set_xlabel("최종 화재위험점수", color="#475569")
    ax_top.grid(axis="x", color="#E2E8F0")
    ax_top.set_axisbelow(True)
    ax_top.spines[["top", "right", "left"]].set_visible(False)
    ax_top.tick_params(axis="y", labelsize=9)

    # 전체 제목/부제/캡션
    fig.suptitle(
        "군집별 화재위험 분석",
        x=0.055,
        y=0.975,
        ha="left",
        fontsize=25,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.055,
        0.94,
        "사용자 정의 AHP 가중치 기반 최종 점수 + KMeans(K=3) 군집화 | 군집명은 평균 위험점수 기준 재정렬",
        ha="left",
        fontsize=12,
        color="#475569",
    )
    fig.text(
        0.055,
        0.03,
        "색상: 초록=저위험군, 주황=중위험군, 빨강=고위험군. 변수 프로파일은 Min-Max 정규화 평균입니다.",
        ha="left",
        fontsize=11,
        color="#64748B",
    )
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.08, right=0.97)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())

    print(f"saved_png={OUT_PNG}")
    print(f"saved_csv={OUT_CSV}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
