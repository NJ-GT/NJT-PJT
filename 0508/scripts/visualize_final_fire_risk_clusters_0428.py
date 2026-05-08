# -*- coding: utf-8 -*-
"""
최종 화재위험 K=3 군집 결과를 한 장 대시보드 PNG 로 시각화하는 스크립트.

목적:
    - AHP 가중치 기반 최종 화재위험점수 + K-Means(K=3) 결과를
      6분면 대시보드(군집 규모/박스플롯/변수 프로파일/구별 고위험 비율/공간분포/TOP10) 로 요약.

입력:
    - 0424/data/최종_화재위험_분석결과_0428.csv
        (cluster_k3, 최종_화재위험점수, 위도/경도/구/숙소명, 9개 위험변수)

출력:
    - 0424/data/최종_화재위험_군집화_대시보드_0428.png

특징:
    - 위험군 명칭은 cluster_k3 의 평균 점수 기준으로 저/중/고위험군 자동 재정렬.
    - 변수 프로파일은 MinMax 정규화 평균값 — 군집간 상대 비교 용이.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

# 헤드리스 환경 PNG 저장
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# 경로 — 스크립트 위치 기준
BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "0424" / "data" / "최종_화재위험_분석결과_0428.csv"
OUT = BASE / "0424" / "data" / "최종_화재위험_군집화_대시보드_0428.png"

# 군집 프로파일에 사용할 9개 위험변수
FEATURES = [
    "구조노후도",
    "단속위험도",
    "도로폭위험도",
    "최근접_소화용수_거리등급",
    "소방위험도_점수",
    "연면적",
    "집중도",
    "주변건물수",
    "총층수",
]

# 히트맵 가독성을 위해 변수명 짧게 줄바꿈한 라벨
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

# 위험군 라벨 + 색상 (저=초록, 중=주황, 고=빨강)
RISK_LABELS = ["저위험군", "중위험군", "고위험군"]
COLORS = {
    "저위험군": "#15803D",
    "중위험군": "#F59E0B",
    "고위험군": "#DC2626",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    CSV 로드 → cluster_k3 평균점수 기준 위험군 재라벨 → MinMax 정규화 데이터 준비.

    반환:
        df     : 원본 + '위험군' 컬럼이 추가된 DataFrame
        scaled : MinMax 정규화된 9변수 + '위험군' DataFrame
    """
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    # 군집 ID 를 평균 점수 오름차순으로 정렬해 → 0번이 가장 안전, 마지막이 가장 위험
    cluster_order = (
        df.groupby("cluster_k3")["최종_화재위험점수"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    # 정렬 순서대로 저/중/고 라벨 매핑
    label_map = {cluster: label for cluster, label in zip(cluster_order, RISK_LABELS)}
    df["위험군"] = df["cluster_k3"].map(label_map)
    # 시각화 정렬을 위해 카테고리 순서 고정
    df["위험군"] = pd.Categorical(df["위험군"], categories=RISK_LABELS, ordered=True)

    # 9변수 — 결측은 0으로 처리 후 MinMax 정규화
    x = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(x), columns=FEATURES, index=df.index
    )
    scaled["위험군"] = df["위험군"]
    return df, scaled


def main() -> None:
    """6분면 대시보드 그리기 및 저장."""
    df, scaled = load_data()
    # 위험군별 시설 수 + 점수 통계
    summary = (
        df.groupby("위험군", observed=True)
        .agg(
            시설수=("숙소명", "size"),
            평균점수=("최종_화재위험점수", "mean"),
            중앙점수=("최종_화재위험점수", "median"),
            최고점수=("최종_화재위험점수", "max"),
        )
        .reindex(RISK_LABELS)
    )
    # 위험군별 변수 평균 (정규화 후) — 히트맵 입력
    profile = (
        scaled.groupby("위험군", observed=True)[FEATURES].mean().reindex(RISK_LABELS)
    )
    # 구×위험군 시설 수 + 구별 전체 시설 수
    gu_cluster = (
        df.groupby(["구", "위험군"], observed=True)
        .size()
        .rename("시설수")
        .reset_index()
    )
    gu_total = df.groupby("구").size().rename("전체")
    # 구별 고위험군 비율 산출
    gu_cluster = gu_cluster.merge(gu_total, on="구")
    gu_cluster["비율"] = gu_cluster["시설수"] / gu_cluster["전체"] * 100
    # 고위험군 비율을 구별로 추려 오름차순 정렬 (수평막대 가독성)
    high_share = (
        gu_cluster[gu_cluster["위험군"].eq("고위험군")]
        .set_index("구")["비율"]
        .sort_values()
    )
    # 위험점수 상위 10개 시설 — 가로 막대용으로 오름차순 정렬
    top10 = df.nlargest(10, "최종_화재위험점수").sort_values("최종_화재위험점수")

    # 한글 폰트 / 마이너스 부호 처리
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # ── Figure / GridSpec ──
    fig = plt.figure(figsize=(18, 10.5), facecolor="#F8FAFC")
    # 2행 3열 — 좌상부터: 군집규모 / 박스 / 히트맵 / 구별비율 / 공간분포 / TOP10
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.08, 1.1],
        height_ratios=[1.0, 1.08],
        hspace=0.35,
        wspace=0.28,
    )
    ax_summary = fig.add_subplot(gs[0, 0], facecolor="white")
    ax_box = fig.add_subplot(gs[0, 1], facecolor="white")
    ax_heat = fig.add_subplot(gs[0, 2], facecolor="white")
    ax_gu = fig.add_subplot(gs[1, 0], facecolor="white")
    ax_map = fig.add_subplot(gs[1, 1], facecolor="white")
    ax_top = fig.add_subplot(gs[1, 2], facecolor="white")

    # ── [좌상] 군집 규모 + 평균 점수 (이중축) ──
    xs = np.arange(len(RISK_LABELS))
    bars = ax_summary.bar(
        xs,
        summary["시설수"],
        color=[COLORS[l] for l in RISK_LABELS],
        width=0.56,
        edgecolor="white",
        linewidth=1.5,
    )
    # 동일 x축에 평균 점수 라인을 우측 y축으로 추가
    ax_summary2 = ax_summary.twinx()
    ax_summary2.plot(
        xs,
        summary["평균점수"],
        color="#0F172A",
        linewidth=2.4,
        marker="o",
        markersize=7,
    )
    # 막대 위/라인 위 텍스트 라벨
    for i, label in enumerate(RISK_LABELS):
        ax_summary.text(
            i,
            summary.loc[label, "시설수"] + 24,
            f"{summary.loc[label, '시설수']:,}개",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
        ax_summary2.text(
            i,
            summary.loc[label, "평균점수"] + 0.8,
            f"{summary.loc[label, '평균점수']:.1f}점",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#0F172A",
        )
    ax_summary.set_xticks(xs, RISK_LABELS, fontsize=12, fontweight="bold")
    ax_summary.set_title(
        "군집 규모와 평균 점수", loc="left", fontsize=15, fontweight="bold", pad=12
    )
    ax_summary.set_ylabel("시설수", color="#475569")
    ax_summary2.set_ylabel("평균 점수", color="#475569")
    ax_summary.grid(axis="y", color="#E2E8F0")
    ax_summary.set_axisbelow(True)
    ax_summary.spines[["top", "right", "left"]].set_visible(False)
    ax_summary2.spines[["top", "left"]].set_visible(False)

    # ── [중상] 위험점수 박스플롯 ──
    box_data = [
        df.loc[df["위험군"].eq(label), "최종_화재위험점수"] for label in RISK_LABELS
    ]
    bp = ax_box.boxplot(
        box_data, patch_artist=True, labels=RISK_LABELS, widths=0.55, showfliers=False
    )
    # 박스에 위험군별 색상 적용
    for patch, label in zip(bp["boxes"], RISK_LABELS):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.82)
        patch.set_edgecolor("white")
    # 수염/모자/중앙선 색 통일
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color("#334155")
            item.set_linewidth(1.3)
    ax_box.set_title(
        "위험점수 분포", loc="left", fontsize=15, fontweight="bold", pad=12
    )
    ax_box.set_ylabel("최종 화재위험점수", color="#475569")
    ax_box.grid(axis="y", color="#E2E8F0")
    ax_box.spines[["top", "right"]].set_visible(False)

    # ── [우상] 변수 프로파일 히트맵 ──
    im = ax_heat.imshow(
        profile.to_numpy(), cmap="YlOrRd", vmin=0, vmax=1, aspect="auto"
    )
    ax_heat.set_xticks(
        np.arange(len(FEATURES)), [FEATURE_LABELS[f] for f in FEATURES], fontsize=9
    )
    ax_heat.set_yticks(
        np.arange(len(RISK_LABELS)), RISK_LABELS, fontsize=11, fontweight="bold"
    )
    ax_heat.set_title(
        "군집별 변수 프로파일", loc="left", fontsize=15, fontweight="bold", pad=12
    )
    # 셀 값 텍스트 — 0.52 초과면 흰 글자, 이하면 검정 글자(가독성)
    for y in range(profile.shape[0]):
        for x in range(profile.shape[1]):
            value = profile.iloc[y, x]
            ax_heat.text(
                x,
                y,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="white" if value > 0.52 else "#111827",
            )
    # 테두리 제거 + 우측에 컬러바
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02)
    cbar.set_label("정규화 평균", color="#475569")

    # ── [좌하] 구별 고위험군 비율 (수평 막대) ──
    # 중앙값 미만은 주황, 이상은 빨강 (위험 강도 시각 강조)
    colors_gu = [
        "#F59E0B" if v < high_share.median() else "#DC2626" for v in high_share
    ]
    ax_gu.barh(
        high_share.index,
        high_share.values,
        color=colors_gu,
        edgecolor="white",
        linewidth=1.0,
    )
    # 막대 끝에 비율 텍스트
    for idx, value in enumerate(high_share.values):
        ax_gu.text(
            value + 0.8,
            idx,
            f"{value:.1f}%",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#0F172A",
        )
    ax_gu.set_title(
        "구별 고위험군 비율", loc="left", fontsize=15, fontweight="bold", pad=12
    )
    ax_gu.set_xlabel("고위험군 비율 (%)", color="#475569")
    ax_gu.grid(axis="x", color="#E2E8F0")
    ax_gu.set_axisbelow(True)
    ax_gu.spines[["top", "right", "left"]].set_visible(False)

    # ── [중하] 공간 산점도 — 위/경도 위에 위험군 색상으로 점 ──
    for label in RISK_LABELS:
        sub = df[df["위험군"].eq(label)]
        # 고위험군은 점을 약간 크고 진하게 — 시각적 강조
        ax_map.scatter(
            sub["경도"],
            sub["위도"],
            s=14 if label != "고위험군" else 22,
            c=COLORS[label],
            alpha=0.42 if label != "고위험군" else 0.75,
            label=f"{label} ({len(sub):,})",
            linewidths=0,
        )
    ax_map.set_title("공간 분포", loc="left", fontsize=15, fontweight="bold", pad=12)
    ax_map.set_xlabel("경도", color="#475569")
    ax_map.set_ylabel("위도", color="#475569")
    ax_map.grid(color="#E2E8F0")
    ax_map.legend(frameon=False, loc="lower left", fontsize=9.5)
    ax_map.spines[["top", "right"]].set_visible(False)

    # ── [우하] TOP 10 위험 시설 (수평 막대) ──
    ax_top.barh(
        top10["숙소명"].str.slice(0, 18),  # 이름 너무 길면 18자로 잘라 표기
        top10["최종_화재위험점수"],
        color=top10["위험군"].map(COLORS),
        edgecolor="white",
        linewidth=1.0,
    )
    # 막대 끝에 점수와 자치구 텍스트
    for idx, (_, row) in enumerate(top10.iterrows()):
        ax_top.text(
            row["최종_화재위험점수"] + 0.4,
            idx,
            f"{row['최종_화재위험점수']:.1f} | {row['구']}",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#0F172A",
        )
    ax_top.set_title(
        "최고 위험 시설 TOP 10", loc="left", fontsize=15, fontweight="bold", pad=12
    )
    ax_top.set_xlabel("최종 화재위험점수", color="#475569")
    ax_top.grid(axis="x", color="#E2E8F0")
    ax_top.spines[["top", "right", "left"]].set_visible(False)
    ax_top.tick_params(axis="y", labelsize=9)

    # 메인 타이틀 + 서브 캡션 + 하단 안내문구
    fig.suptitle(
        "최종 화재위험 군집화 대시보드",
        x=0.055,
        y=0.982,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.055,
        0.946,
        "AHP 가중치 기반 최종 화재위험점수 + KMeans(K=3) | 군집명은 평균 위험점수 기준으로 재정렬",
        ha="left",
        fontsize=12,
        color="#475569",
    )
    fig.text(
        0.055,
        0.032,
        "변수 프로파일은 Min-Max 정규화 평균입니다. 색상: 초록=저위험군, 주황=중위험군, 빨강=고위험군.",
        ha="left",
        fontsize=11,
        color="#64748B",
    )
    # 여백 미세조정 — suptitle/footer 가 잘리지 않도록
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.08, right=0.97)
    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved={OUT}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
