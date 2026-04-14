# -*- coding: utf-8 -*-
"""
Create an HTML visualization for foreign tourist counts by Seoul district.

Usage
-----
python scripts/visualize_seoul_foreign_tourists.py ^
  --input data\\tourism\\visitkorea_foreign_visitors.csv ^
  --output reports\\seoul_foreign_visitors_last_year.html

Notes
-----
- This script is designed for an officially downloaded CSV from Korea Tourism Data Lab.
- It does not scrape the website.
- The script tries to detect common Korean column names automatically.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
from pathlib import Path
from typing import Iterable


SEOUL_DISTRICTS = [
    "강남구", "강동구", "강북구", "강서구", "관악구",
    "광진구", "구로구", "금천구", "노원구", "도봉구",
    "동대문구", "동작구", "마포구", "서대문구", "서초구",
    "성동구", "성북구", "송파구", "양천구", "영등포구",
    "용산구", "은평구", "종로구", "중구", "중랑구",
]


DISTRICT_CANDIDATES = [
    "자치구", "시군구", "시군구명", "시군구명칭", "지역", "지역명", "구군", "sgg_nm",
]
YEAR_CANDIDATES = [
    "기준년도", "년도", "연도", "년", "base_year", "year",
]
MONTH_CANDIDATES = [
    "기준월", "월", "month",
]
YEARMONTH_CANDIDATES = [
    "기준년월", "년월", "ym", "yyyymm", "base_ym",
]
FOREIGN_VALUE_CANDIDATES = [
    "외국인 관광객 수", "외국인관광객수", "외국인 관광객수",
    "외국인 방문자수", "외국인방문자수", "외국인 방문객수",
    "외래객 수", "외래객수", "외래관광객 수", "외래관광객수",
    "foreign_visitors", "foreign_visitor_count", "visitor_foreign",
]


def normalize(text: str) -> str:
    return "".join(str(text).strip().lower().replace("_", "").replace(" ", "").split())


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        return list(reader)


def find_column(fieldnames: Iterable[str], candidates: list[str]) -> str | None:
    normalized = {normalize(name): name for name in fieldnames}

    for candidate in candidates:
        key = normalize(candidate)
        if key in normalized:
            return normalized[key]

    for name in fieldnames:
        name_key = normalize(name)
        if any(normalize(candidate) in name_key for candidate in candidates):
            return name

    return None


def detect_district_column(rows: list[dict[str, str]], fieldnames: list[str]) -> str:
    direct = find_column(fieldnames, DISTRICT_CANDIDATES)
    if direct:
        return direct

    for name in fieldnames:
        values = {str(row.get(name, "")).strip() for row in rows[:300] if str(row.get(name, "")).strip()}
        if sum(value in SEOUL_DISTRICTS for value in values) >= 5:
            return name

    raise ValueError("자치구 컬럼을 찾지 못했습니다.")


def detect_value_column(fieldnames: list[str]) -> str:
    direct = find_column(fieldnames, FOREIGN_VALUE_CANDIDATES)
    if direct:
        return direct

    for name in fieldnames:
        key = normalize(name)
        if ("외국인" in name or "외래" in name) and ("수" in name or "count" in key or "visitor" in key):
            return name

    raise ValueError("외국인 관광객 수 컬럼을 찾지 못했습니다.")


def parse_int(value: str) -> int | None:
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        if "." in text:
            return int(float(text))
        return int(text)
    except ValueError:
        return None


def parse_year_month(row: dict[str, str], fieldnames: list[str]) -> tuple[int | None, int | None]:
    ym_col = find_column(fieldnames, YEARMONTH_CANDIDATES)
    if ym_col:
        raw = str(row.get(ym_col, "")).strip().replace("-", "").replace(".", "")
        if len(raw) >= 6 and raw[:6].isdigit():
            return int(raw[:4]), int(raw[4:6])

    year_col = find_column(fieldnames, YEAR_CANDIDATES)
    month_col = find_column(fieldnames, MONTH_CANDIDATES)
    year = parse_int(row.get(year_col, "")) if year_col else None
    month = parse_int(row.get(month_col, "")) if month_col else None
    return year, month


def build_dataset(rows: list[dict[str, str]], target_year: int) -> tuple[list[dict[str, object]], str, str]:
    if not rows:
        raise ValueError("입력 CSV에 데이터가 없습니다.")

    fieldnames = list(rows[0].keys())
    district_col = detect_district_column(rows, fieldnames)
    value_col = detect_value_column(fieldnames)

    monthly = {(district, month): 0 for district in SEOUL_DISTRICTS for month in range(1, 13)}

    for row in rows:
        district = str(row.get(district_col, "")).strip()
        year, month = parse_year_month(row, fieldnames)
        value = parse_int(row.get(value_col, ""))
        if district not in SEOUL_DISTRICTS:
            continue
        if year != target_year or month not in range(1, 13) or value is None:
            continue
        monthly[(district, month)] += value

    data = []
    for district in SEOUL_DISTRICTS:
        month_values = [monthly[(district, month)] for month in range(1, 13)]
        data.append(
            {
                "district": district,
                "months": month_values,
                "total": sum(month_values),
            }
        )

    if not any(item["total"] > 0 for item in data):
        raise ValueError(
            f"{target_year}년 서울 자치구 외국인 관광객 데이터를 찾지 못했습니다. "
            f"CSV 컬럼을 확인해 주세요: {fieldnames}"
        )

    return data, district_col, value_col


def render_html(data: list[dict[str, object]], year: int, title: str) -> str:
    months = [f"{month}월" for month in range(1, 13)]
    totals_sorted = sorted(data, key=lambda item: item["total"], reverse=True)

    payload = {
        "months": months,
        "series": data,
        "ranking": totals_sorted,
        "year": year,
        "title": title,
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    payload_json = json.dumps(payload, ensure_ascii=False)
    safe_title = html.escape(title)

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #f7f3ec;
      --panel: #fffaf2;
      --ink: #1f1d1a;
      --sub: #6b6257;
      --accent: #00796b;
      --accent-2: #d95d39;
      --line: #e7dccd;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Pretendard", "Noto Sans KR", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(217,93,57,.12), transparent 30%),
        radial-gradient(circle at bottom right, rgba(0,121,107,.12), transparent 30%),
        var(--bg);
    }}
    .wrap {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, #fff8ef, #f4efe6);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 28px 28px 20px;
      box-shadow: 0 14px 40px rgba(67, 56, 37, 0.08);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 44px);
      line-height: 1.08;
    }}
    .sub {{
      margin: 0;
      color: var(--sub);
      font-size: 15px;
      line-height: 1.6;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 18px;
      margin-top: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 10px 24px rgba(67, 56, 37, 0.05);
    }}
    .panel h2 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .metric {{
      background: rgba(255,255,255,.6);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 16px;
    }}
    .metric .label {{
      color: var(--sub);
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .metric .value {{
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .notes {{
      margin-top: 14px;
      color: var(--sub);
      font-size: 13px;
      line-height: 1.6;
    }}
    @media (max-width: 980px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
      .meta {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>{safe_title}</h1>
      <p class="sub">서울 25개 자치구의 {year}년 1월부터 12월까지 외국인 관광객 수를 비교합니다. 아래에는 월별 추이와 연간 합계 순위를 함께 넣었습니다.</p>
      <div class="meta">
        <div class="metric">
          <div class="label">기준 연도</div>
          <div class="value">{year}</div>
        </div>
        <div class="metric">
          <div class="label">비교 자치구 수</div>
          <div class="value" id="district-count">25</div>
        </div>
        <div class="metric">
          <div class="label">총 외국인 관광객 수</div>
          <div class="value" id="total-visitors">-</div>
        </div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>월별 비교</h2>
        <div id="monthly-chart" style="height: 720px;"></div>
      </div>
      <div class="panel">
        <h2>연간 합계 순위</h2>
        <div id="ranking-chart" style="height: 720px;"></div>
      </div>
    </section>

    <section class="panel" style="margin-top: 18px;">
      <h2>해석 메모</h2>
      <div class="notes">
        <div>생성 시각: {html.escape(payload["generated_at"])}</div>
        <div>출처: 한국관광 데이터랩 공식 다운로드 CSV</div>
        <div>주의: 데이터랩 수치는 해당 원본 정의를 따르므로, 총량 해석보다는 추세 비교에 더 적합할 수 있습니다.</div>
      </div>
    </section>
  </div>

  <script>
    const payload = {payload_json};
    const palette = [
      "#0f766e", "#d95d39", "#3b82f6", "#7c3aed", "#059669",
      "#ea580c", "#2563eb", "#be123c", "#65a30d", "#a16207",
      "#0891b2", "#4338ca", "#16a34a", "#c2410c", "#4f46e5",
      "#b91c1c", "#15803d", "#92400e", "#0369a1", "#6d28d9",
      "#0d9488", "#dc2626", "#4d7c0f", "#1d4ed8", "#7e22ce"
    ];

    const monthlyTraces = payload.series.map((item, idx) => {{
      return {{
        x: payload.months,
        y: item.months,
        type: "scatter",
        mode: "lines+markers",
        name: item.district,
        line: {{ width: 2.5, color: palette[idx % palette.length] }},
        marker: {{ size: 6 }}
      }};
    }});

    Plotly.newPlot("monthly-chart", monthlyTraces, {{
      margin: {{ l: 56, r: 20, t: 10, b: 50 }},
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(255,255,255,0.6)",
      legend: {{ orientation: "h", y: -0.18 }},
      hovermode: "x unified",
      xaxis: {{ title: "월", gridcolor: "#eadfce" }},
      yaxis: {{ title: "외국인 관광객 수", separatethousands: true, gridcolor: "#eadfce" }},
    }}, {{
      responsive: true,
      displaylogo: false
    }});

    const ranking = payload.ranking.slice().reverse();
    Plotly.newPlot("ranking-chart", [{{
      x: ranking.map(item => item.total),
      y: ranking.map(item => item.district),
      type: "bar",
      orientation: "h",
      marker: {{
        color: ranking.map((_, idx) => palette[idx % palette.length])
      }}
    }}], {{
      margin: {{ l: 80, r: 20, t: 10, b: 50 }},
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(255,255,255,0.6)",
      xaxis: {{ title: "연간 외국인 관광객 수", separatethousands: true, gridcolor: "#eadfce" }},
      yaxis: {{ title: "" }}
    }}, {{
      responsive: true,
      displaylogo: false
    }});

    const total = payload.series.reduce((sum, item) => sum + item.total, 0);
    document.getElementById("total-visitors").textContent = total.toLocaleString("ko-KR");
    document.getElementById("district-count").textContent = payload.series.length.toLocaleString("ko-KR");
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Officially downloaded CSV path")
    parser.add_argument("--output", required=True, help="Output HTML path")
    parser.add_argument("--year", type=int, default=dt.date.today().year - 1, help="Target year")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = load_rows(input_path)
    data, district_col, value_col = build_dataset(rows, args.year)

    title = f"서울 자치구별 외국인 관광객 비교 ({args.year}년 1월-12월)"
    html_text = render_html(data, args.year, title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")

    print(f"Input: {input_path}")
    print(f"District column: {district_col}")
    print(f"Value column: {value_col}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
