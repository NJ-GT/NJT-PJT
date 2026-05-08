# -*- coding: utf-8 -*-
"""
dashboard.py 의 dong_focus_map 함수와 관련 안내 문구를 새 버전으로 패치하는 일회성 스크립트.

목적:
    - 기존 dong_focus_map(점 분포만) → 법정동 경계 폴리곤 + 점 + 라벨 + 위험군 색 채움 버전으로 교체
    - 같이 깨져 있던 한글 안내 문구(broken) 도 정상 한글 메시지로 치환

전제:
    - 같은 폴더 상위(parents[1])에 dashboard.py 가 위치한다.
    - dashboard.py 안에 'def dong_focus_map' 정의와 '\n\nGWR_VARIABLE_COLORS' 마커가 둘 다 존재한다.

처리 흐름:
    1) dashboard.py 텍스트 로드
    2) dong_focus_map 함수 시작 ~ 'GWR_VARIABLE_COLORS' 직전까지를 NEW_DONG_FOCUS_MAP 으로 치환
    3) 깨진 안내 문구(overview, risk) 치환
    4) 결과 텍스트로 dashboard.py 덮어쓰기
"""
from __future__ import annotations

from pathlib import Path


# 패치 대상 dashboard.py 경로
ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "dashboard.py"


# 새로 삽입할 dong_focus_map 함수 본문
# (NJT-PJT/dashboard.py 안에서 def dong_focus_map ~ 다음 GWR_VARIABLE_COLORS 사이를 통째 치환)
NEW_DONG_FOCUS_MAP = r"""def dong_focus_map(df: pd.DataFrame, gu: str, clusters: list[str]) -> go.Figure:
    scoped = df[(df["구"] == gu) & (df["cluster_label"].isin(clusters))].dropna(subset=["위도", "경도"]).copy()
    if scoped.empty:
        return go.Figure()

    cluster_order = [label for label in ["저위험군", "중위험군", "고위험군"] if label in clusters]
    elegant_cluster_colors = {
        "저위험군": "#58C7A5",
        "중위험군": "#F2B84B",
        "고위험군": "#E96B6C",
    }
    elegant_cluster_fill = {
        "저위험군": "rgba(88,199,165,0.62)",
        "중위험군": "rgba(242,184,75,0.64)",
        "고위험군": "rgba(233,107,108,0.66)",
    }

    dong_summary = (
        scoped.groupby("동", as_index=False)
        .agg(
            숙박시설수=("숙소명", "count"),
            평균위험도=("최종위험점수_new", "mean"),
            최고위험도=("최종위험점수_new", "max"),
            평균소화용수등급=("최근접_소화용수_거리등급", "mean"),
        )
        .sort_values(["숙박시설수", "평균위험도"], ascending=False)
    )
    cluster_counts = (
        scoped.pivot_table(index="동", columns="cluster_label", values="숙소명", aggfunc="count", fill_value=0)
        .reindex(columns=["저위험군", "중위험군", "고위험군"], fill_value=0)
        .reset_index()
    )
    dong_summary = dong_summary.merge(cluster_counts, on="동", how="left")
    for label in ["저위험군", "중위험군", "고위험군"]:
        if label not in dong_summary.columns:
            dong_summary[label] = 0
    selected_cluster_order = [label for label in ["저위험군", "중위험군", "고위험군"] if label in clusters]
    dong_summary["대표위험군"] = dong_summary[selected_cluster_order].idxmax(axis=1)
    dong_summary["대표위험군수"] = dong_summary[selected_cluster_order].max(axis=1).astype(int)
    dong_summary["위험군구성"] = dong_summary.apply(
        lambda row: " / ".join(f"{label.replace('위험군', '')} {int(row[label])}개" for label in selected_cluster_order),
        axis=1,
    )

    summary_by_dong = {str(row["동"]): row for _, row in dong_summary.iterrows()}
    fig = go.Figure()

    boundary_features = []
    for feature in dong_boundary_geo.get("features", []):
        props = feature.get("properties", {})
        feature_gu = infer_gu_name(props)
        dong_name = props.get("법정동명") or props.get("EMD_KOR_NM")
        if feature_gu == gu and dong_name in summary_by_dong:
            boundary_features.append((feature, str(dong_name)))

    label_rows = []
    for feature, dong_name in boundary_features:
        row = summary_by_dong[dong_name]
        xs, ys = polygon_line_coords(feature.get("geometry", {}))
        if not xs:
            continue
        risk_group = row["대표위험군"]
        hover_text = (
            f"<b>{dong_name}</b><br>"
            f"대표 위험군: {risk_group}<br>"
            f"숙박시설: {int(row['숙박시설수']):,}개<br>"
            f"위험군 구성: {row['위험군구성']}<br>"
            f"평균 위험도: {float(row['평균위험도']):.2f}점<br>"
            f"최고 위험도: {float(row['최고위험도']):.2f}점<br>"
            f"평균 소화용수 거리등급: {float(row['평균소화용수등급']):.2f}"
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                fillcolor=elegant_cluster_fill.get(risk_group, "rgba(226,232,240,0.56)"),
                line=dict(color="rgba(61,78,96,0.58)", width=0.75),
                text=[hover_text] * len(xs),
                hovertemplate="%{text}<extra></extra>",
                name=dong_name,
                showlegend=False,
            )
        )
        try:
            centroid = shape(feature.get("geometry", {})).centroid
            label_rows.append({"동": dong_name, "경도": centroid.x, "위도": centroid.y, "숙박시설수": row["숙박시설수"]})
        except Exception:
            pass

    if not boundary_features:
        fig.add_trace(
            go.Scatter(
                x=scoped["경도"],
                y=scoped["위도"],
                mode="markers",
                marker=dict(size=7, color=scoped["cluster_label"].map(elegant_cluster_colors), opacity=0.78, line=dict(color="white", width=0.7)),
                customdata=scoped[["숙소명", "동", "업종", "cluster_label", "최종위험점수_new"]],
                hovertemplate="<b>%{customdata[0]}</b><br>법정동: %{customdata[1]}<br>업종: %{customdata[2]}<br>위험군: %{customdata[3]}<br>위험도: %{customdata[4]:.2f}점<extra></extra>",
                name="숙박시설 위치",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=scoped["경도"],
                y=scoped["위도"],
                mode="markers",
                marker=dict(size=4.2, color="rgba(30,41,59,0.34)", line=dict(color="rgba(255,255,255,0.62)", width=0.4), opacity=0.78),
                customdata=scoped[["숙소명", "동", "업종", "cluster_label", "최종위험점수_new"]],
                hovertemplate="<b>%{customdata[0]}</b><br>법정동: %{customdata[1]}<br>업종: %{customdata[2]}<br>위험군: %{customdata[3]}<br>위험도: %{customdata[4]:.2f}점<extra></extra>",
                name="숙박시설 위치",
            )
        )

    if label_rows:
        label_df = pd.DataFrame(label_rows).sort_values("숙박시설수", ascending=False).head(14)
        fig.add_trace(
            go.Scatter(
                x=label_df["경도"],
                y=label_df["위도"],
                mode="text",
                text=label_df["동"],
                textfont=dict(size=11, color="#223044"),
                textposition="middle center",
                hoverinfo="skip",
                name="법정동명",
                showlegend=False,
            )
        )

    for label in cluster_order:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=13, color=elegant_cluster_colors[label], symbol="square"),
                name=f"{label} 우세 동",
                hoverinfo="skip",
            )
        )

    lon_values = []
    lat_values = []
    for trace in fig.data:
        xs = [v for v in getattr(trace, "x", []) if v is not None]
        ys = [v for v in getattr(trace, "y", []) if v is not None]
        lon_values.extend(xs)
        lat_values.extend(ys)
    if not lon_values or not lat_values:
        lon_values = scoped["경도"].tolist()
        lat_values = scoped["위도"].tolist()
    lon_range = max(lon_values) - min(lon_values)
    lat_range = max(lat_values) - min(lat_values)
    pad_lon = max(lon_range * 0.08, 0.003)
    pad_lat = max(lat_range * 0.08, 0.003)

    fig.update_layout(
        title=f"{gu} 법정동별 위험군 구역도",
        xaxis=dict(
            title="",
            range=[min(lon_values) - pad_lon, max(lon_values) + pad_lon],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            range=[min(lat_values) - pad_lat, max(lat_values) + pad_lat],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        height=620,
        plot_bgcolor="#f8fafc",
        paper_bgcolor="white",
        margin=dict(l=8, r=8, t=58, b=8),
        font=dict(color=COLORS["ink"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            x=0.01,
            bgcolor="rgba(255,255,255,0.86)",
            bordercolor="#e5edf5",
            borderwidth=1,
        ),
    )
    return fig
"""


def main() -> None:
    """dashboard.py 의 함수 정의 + 안내 문구를 패치한다."""
    # 대상 파일을 UTF-8 텍스트로 로드
    text = TARGET.read_text(encoding="utf-8")
    # dong_focus_map 함수 시작/종료 위치 탐색
    start = text.find("def dong_focus_map")
    end = text.find("\n\nGWR_VARIABLE_COLORS", start)
    if start == -1 or end == -1:
        # 두 마커 중 하나라도 없으면 안전하게 중단
        raise RuntimeError("Could not locate dong_focus_map block")
    # 함수 영역만 통째로 새 버전으로 교체
    text = text[:start] + NEW_DONG_FOCUS_MAP + text[end:]
    # ── 깨진 한글 안내 문구 치환 ─────────────────────────────────────
    # (이전 버전에서 cp949 → utf-8 인코딩 깨짐으로 '?' 만 남은 문자열)
    broken = '<div class="soft-note">?? ?? ??? ??? ???? ??? ???, ? ????? ?? ?? ???? ??? ??? ????. ?? ?? ???? ????, ???? ??? ???? ??? ??? ?? ???? ??? ? ????.</div>'
    overview = '<div class="soft-note">0430 최종테이블의 선택 구 평균을 서울 10구 전체 평균과 비교합니다. 100보다 크면 전체 평균보다 높은 지표입니다.</div>'
    risk_old = '<div class="soft-note">0430 파일의 숙소 좌표만 사용해 선택 구 안의 법정동별 분포를 근사 표시합니다. 실제 법정동 경계선은 0430에 없어 포함하지 않았습니다.</div>'
    risk_new = '<div class="soft-note">선택 구의 법정동 경계를 기준으로 구역을 나누고, 각 법정동에서 가장 많이 나타나는 위험군 색으로 채웁니다. 점은 개별 숙박시설 위치이며, 마우스를 올리면 법정동별 위험군 구성과 평균 위험도를 확인할 수 있습니다.</div>'
    text = text.replace(broken, overview)
    text = text.replace(risk_old, risk_new)
    # 결과 텍스트 저장 (UTF-8)
    TARGET.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
