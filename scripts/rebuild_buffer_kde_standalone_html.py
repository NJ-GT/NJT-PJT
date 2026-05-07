# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from shapely.geometry import shape
from shapely.ops import unary_union


BASE = Path(__file__).resolve().parents[1]
INPUT_CSV = BASE / "data" / "서울10구_숙소_소방거리_유클리드.csv"
BOUNDARY_GEOJSON = BASE / "data" / "seoul_legal_dong_age_buckets_joined_0415.geojson"
OUT_HTML = BASE / "data" / "Map_Buffer_KDE_standalone.html"
OUT_PNG = BASE / "data" / "Map_Buffer_KDE_standalone.png"

SEOUL_SIGUNGU = {
    "11110": "종로구",
    "11140": "중구",
    "11170": "용산구",
    "11200": "성동구",
    "11215": "광진구",
    "11230": "동대문구",
    "11260": "중랑구",
    "11290": "성북구",
    "11305": "강북구",
    "11320": "도봉구",
    "11350": "노원구",
    "11380": "은평구",
    "11410": "서대문구",
    "11440": "마포구",
    "11470": "양천구",
    "11500": "강서구",
    "11530": "구로구",
    "11545": "금천구",
    "11560": "영등포구",
    "11590": "동작구",
    "11620": "관악구",
    "11650": "서초구",
    "11680": "강남구",
    "11710": "송파구",
    "11740": "강동구",
}


def set_korean_font() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def iter_polygon_rings(geometry: dict):
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        for ring in coords:
            yield ring
    elif gtype == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                yield ring


def iter_shapely_rings(geom):
    if geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield list(geom.exterior.coords)
        return
    if geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            yield list(part.exterior.coords)
        return
    if geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from iter_shapely_rings(part)


def infer_gu_name(props: dict) -> str | None:
    gu = props.get("구")
    if isinstance(gu, str) and gu.endswith("구"):
        return gu
    code = str(
        props.get("법정동코드") or props.get("EMD_CD") or props.get("join_code") or ""
    )
    return SEOUL_SIGUNGU.get(code[:5])


def add_boundaries(ax, path: Path) -> None:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        geo = json.load(f)
    for feature in geo.get("features", []):
        for ring in iter_polygon_rings(feature.get("geometry") or {}):
            if not ring:
                continue
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            ax.plot(xs, ys, color="#CBD5E1", linewidth=0.35, alpha=0.85, zorder=2)


def add_gu_boundaries(ax, path: Path) -> None:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        geo = json.load(f)
    grouped: dict[str, list] = {}
    for feature in geo.get("features", []):
        gu_name = infer_gu_name(feature.get("properties", {}))
        if not gu_name:
            continue
        try:
            geom = shape(feature.get("geometry"))
            if not geom.is_valid:
                geom = geom.buffer(0)
        except Exception:
            continue
        grouped.setdefault(gu_name, []).append(geom)

    for gu_name, geoms in grouped.items():
        try:
            geom = unary_union(geoms)
        except Exception:
            continue
        for ring in iter_shapely_rings(geom):
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            ax.plot(xs, ys, color="#1E293B", linewidth=1.25, alpha=0.95, zorder=5)
        centroid = geom.representative_point()
        ax.text(
            centroid.x,
            centroid.y,
            gu_name,
            fontsize=8.5,
            weight="bold",
            color="#0F172A",
            ha="center",
            va="center",
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.16",
                facecolor="white",
                edgecolor="none",
                alpha=0.62,
            ),
        )


def build_kde_image() -> tuple[str, pd.DataFrame]:
    set_korean_font()
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    required = ["위도", "경도", "반경_50m_건물수"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"필수 컬럼 없음: {missing}")

    df = df.dropna(subset=required).copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)
    df = df[
        (df["위도"].between(37.4, 37.7)) & (df["경도"].between(126.7, 127.3))
    ].copy()

    weights = df["반경_50m_건물수"].to_numpy(dtype=float)
    if np.isclose(weights.sum(), 0):
        weights = np.ones(len(df), dtype=float)
    weights = weights / weights.sum()

    lon = df["경도"].to_numpy(dtype=float)
    lat = df["위도"].to_numpy(dtype=float)

    pad_lon = 0.02
    pad_lat = 0.02
    lon_min, lon_max = lon.min() - pad_lon, lon.max() + pad_lon
    lat_min, lat_max = lat.min() - pad_lat, lat.max() + pad_lat

    grid_n = 360
    grid_lon, grid_lat = np.mgrid[
        lon_min : lon_max : grid_n * 1j, lat_min : lat_max : grid_n * 1j
    ]
    kde = gaussian_kde(np.vstack([lon, lat]), weights=weights, bw_method=0.04)
    kde_values = kde(np.vstack([grid_lon.ravel(), grid_lat.ravel()])).reshape(
        grid_n, grid_n
    )
    kde_norm = (kde_values - kde_values.min()) / (kde_values.max() - kde_values.min())

    cmap = plt.get_cmap("YlOrRd")
    cmap_alpha = cmap(np.linspace(0, 1, 256))
    cmap_alpha[:28, 3] = 0
    cmap_alpha[28:58, 3] = np.linspace(0.05, 0.45, 30)
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("YlOrRd_alpha", cmap_alpha)

    fig, ax = plt.subplots(figsize=(13.5, 9), dpi=180)
    ax.set_facecolor("#F8FAFC")
    add_boundaries(ax, BOUNDARY_GEOJSON)
    ax.contourf(
        grid_lon, grid_lat, kde_norm, levels=24, cmap=custom_cmap, alpha=0.92, zorder=3
    )
    add_gu_boundaries(ax, BOUNDARY_GEOJSON)

    bins = [
        (df["반경_50m_건물수"] > 30, "#C0392B", ">30개"),
        (df["반경_50m_건물수"].between(15, 30, inclusive="both"), "#E67E22", "15~30개"),
        (df["반경_50m_건물수"] < 15, "#27AE60", "<15개"),
    ]
    for mask, color, label in bins:
        sub = df[mask]
        ax.scatter(
            sub["경도"],
            sub["위도"],
            s=8,
            color=color,
            alpha=0.72,
            linewidth=0,
            label=label,
            zorder=4,
        )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        "버퍼(50m) + KDE 건물 밀집도", loc="left", fontsize=20, weight="bold", pad=18
    )
    ax.text(
        lon_min,
        lat_max + (lat_max - lat_min) * 0.012,
        f"총 {len(df):,}개 숙박시설 | 서울 10개구 | 가중치 = 반경 50m 건물수 | KDE bw_method=0.04",
        fontsize=10,
        color="#475569",
        va="bottom",
    )

    legend = ax.legend(
        title="개별 포인트",
        loc="lower left",
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#CBD5E1",
    )
    legend.get_title().set_fontweight("bold")

    cax = fig.add_axes([0.78, 0.16, 0.16, 0.018])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    cax.imshow(gradient, aspect="auto", cmap="YlOrRd")
    cax.set_axis_off()
    fig.text(0.78, 0.185, "KDE 밀도", fontsize=10, weight="bold", color="#0F172A")
    fig.text(0.78, 0.135, "낮음", fontsize=8, color="#475569")
    fig.text(0.925, 0.135, "높음", fontsize=8, color="#475569", ha="right")

    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.96])
    fig.savefig(OUT_PNG, bbox_inches="tight", facecolor="white")

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return image_b64, df


def write_html(image_b64: str, df: pd.DataFrame) -> None:
    top_rows = (
        df.sort_values("반경_50m_건물수", ascending=False)
        .head(12)[["구", "동", "업소명", "반경_50m_건물수", "집중도(%)"]]
        .copy()
    )
    table_rows = "\n".join(
        "<tr>"
        f"<td>{row['구']}</td>"
        f"<td>{row['동']}</td>"
        f"<td>{row['업소명']}</td>"
        f"<td>{int(row['반경_50m_건물수'])}</td>"
        f"<td>{float(row['집중도(%)']):.1f}%</td>"
        "</tr>"
        for _, row in top_rows.iterrows()
    )
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>버퍼 KDE 건물 밀집도</title>
  <style>
    body {{
      margin: 0;
      background: #edf2f7;
      color: #0f172a;
      font-family: "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
    }}
    .wrap {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
    }}
    .card {{
      background: white;
      border: 1px solid #dbe3ef;
      border-radius: 18px;
      box-shadow: 0 14px 34px rgba(15, 23, 42, 0.10);
      overflow: hidden;
    }}
    img {{
      width: 100%;
      display: block;
    }}
    .panel {{
      display: grid;
      grid-template-columns: 1fr 1.4fr;
      gap: 18px;
      padding: 18px 22px 24px;
      border-top: 1px solid #e2e8f0;
    }}
    .note {{
      font-size: 14px;
      line-height: 1.65;
      color: #334155;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #e2e8f0;
      padding: 8px 9px;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      background: #f8fafc;
      font-weight: 800;
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="card">
      <img src="data:image/png;base64,{image_b64}" alt="버퍼 50m KDE 건물 밀집도 지도">
      <div class="panel">
        <div class="note">
          <b>표시 내용</b><br>
          50m 버퍼 내 건물수를 가중치로 사용한 KDE 밀도 시각화입니다.<br>
          빨간 영역일수록 숙박시설 주변 건물 밀집 기여가 높은 구간입니다.<br>
          굵은 선은 서울시 25개 구 경계, 얇은 선은 법정동 경계입니다.<br>
          외부 CDN 없이 이미지가 HTML에 내장되어 오프라인에서도 열립니다.
        </div>
        <div>
          <b>반경 50m 건물수 상위 시설</b>
          <table>
            <thead>
              <tr><th>구</th><th>동</th><th>업소명</th><th>건물수</th><th>집중도</th></tr>
            </thead>
            <tbody>
              {table_rows}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  </main>
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")


def main() -> None:
    image_b64, df = build_kde_image()
    write_html(image_b64, df)
    print(f"saved_png={OUT_PNG}")
    print(f"saved_html={OUT_HTML}")
    print(f"rows={len(df)}")


if __name__ == "__main__":
    main()
