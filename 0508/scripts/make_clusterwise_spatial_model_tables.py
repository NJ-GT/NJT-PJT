# -*- coding: utf-8 -*-
"""
위험군(저/중/고)별로 OLS → SLM → SEM → GWR/MGWR 회귀 결과 비교표를 PNG/CSV 로 생성.

목적:
    - 군집별 대표 모형(이미 산출된 평균 계수와 BW)과 동일 군집의 OLS/SLM/SEM 적합 결과를
      한 표에 나란히 배치해, 공간효과 도입에 따른 계수/유의성 변화를 한눈에 비교.

입력:
    - 0430/최종테이블0429.csv                       : 분석 마스터
    - 0430/군집별_대표모형_params_평균계수.csv      : 사전 계산된 대표 모형 평균 계수
    - 0430/가중치군집_대표모형_최종결과표.csv       : 대표 모형 BW/지표 요약

출력 (위험군별 2개 파일 × 3 = 6개):
    - 0430/{위험군}_OLS_SLM_SEM_GWR_MGWR_BW_비교표.csv
    - 0430/{위험군}_OLS_SLM_SEM_GWR_MGWR_BW_비교표.png
"""
from __future__ import annotations

import ast  # MGWR BW 컬럼이 dict 문자열로 저장되어 있어 ast.literal_eval 로 파싱
from pathlib import Path

import matplotlib

# 헤드리스 PNG 저장
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from libpysal.weights import KNN  # K-최근접 이웃 공간가중치
from matplotlib import font_manager  # 한글 폰트 등록
from sklearn.preprocessing import StandardScaler  # 표준화
from spreg import ML_Error, ML_Lag, OLS  # 공간회귀 3종 (OLS/SLM/SEM)


# 경로
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "0430" / "최종테이블0429.csv"
OUT_DIR = ROOT / "0430"
REP_PARAMS = OUT_DIR / "군집별_대표모형_params_평균계수.csv"  # 대표 모형 계수 입력
REP_SUMMARY = OUT_DIR / "가중치군집_대표모형_최종결과표.csv"  # 대표 모형 지표 입력

# 종속변수/군집/좌표
TARGET = "최종위험점수_new"
CLUSTER_COL = "cluster_label"
COORDS = ["x_5181", "y_5181"]  # 평면 좌표 (중부원점)
# 10개 설명변수 — 기존 대표 모형과 동일 순서/이름 유지
FEATURES = [
    "구조노후도",
    "단속위험도",
    "도로폭위험도",
    "최근접_소화용수_거리등급",
    "소방위험도_점수",
    "승인연도",
    "연면적",
    "집중도",
    "주변건물수",
    "총층수",
]

# 표 가독성을 위해 변수를 3개 카테고리로 묶음 (정성 의미 그룹)
CATEGORIES = {
    "위험·접근성": [
        "구조노후도",
        "단속위험도",
        "도로폭위험도",
        "최근접_소화용수_거리등급",
        "소방위험도_점수",
    ],
    "건축·시설": ["승인연도", "연면적", "총층수"],
    "입지·밀집": ["집중도", "주변건물수"],
}

# 표 출력 위험군 순서 — 위험도 오름차순
ORDER = ["저위험군", "중위험군", "고위험군"]


def read_csv(path: Path) -> pd.DataFrame:
    """다양한 인코딩에 강건한 CSV 로드 — utf-8-sig → utf-8 → cp949 순으로 시도."""
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def star(p: float | None) -> str:
    """p-value 를 별표 표기로 변환 — *** p<0.01, ** p<0.05, * p<0.1."""
    if p is None or pd.isna(p):
        return ""
    if p < 0.01:
        return " ***"
    if p < 0.05:
        return " **"
    if p < 0.1:
        return " *"
    return ""


def fmt_coef(v: float | None, p: float | None = None) -> str:
    """계수 값에 별표를 붙여 '계수 ***' 형태 문자열로 포맷팅."""
    if v is None or pd.isna(v):
        return ""
    return f"{v:.3f}{star(p)}"


def fmt_coef_bw(coef: float | None, bw: float | None) -> str:
    """대표 모형용 셀 — 계수 + 줄바꿈 후 'BW=값' 부착 (없으면 계수만)."""
    if coef is None or pd.isna(coef):
        return ""
    if bw is None or pd.isna(bw):
        return f"{coef:.3f}"
    return f"{coef:.3f}\nBW={bw:g}"


def pvals_from_stats(stats) -> list[float | None]:
    """spreg 모델의 t/z 통계 리스트에서 두 번째 요소(p값)를 추출, 실패 시 None."""
    vals: list[float | None] = []
    for item in stats:
        try:
            vals.append(float(item[1]))
        except Exception:
            vals.append(None)
    return vals


def safe_attr(model, *names: str):
    """여러 후보 속성명 중 처음 존재하는 값을 반환 — spreg 버전 호환용."""
    for name in names:
        if hasattr(model, name):
            return getattr(model, name)
    return None


def model_terms(
    model, term_names: list[str], stat_attr: str
) -> dict[str, tuple[float, float | None]]:
    """모델의 betas + p값을 (변수명 → (계수, p값)) dict 로 매핑."""
    betas = [float(x) for x in model.betas.flatten()]
    pvals = pvals_from_stats(getattr(model, stat_attr, []))
    out: dict[str, tuple[float, float | None]] = {}
    for idx, term in enumerate(term_names):
        out[term] = (betas[idx], pvals[idx] if idx < len(pvals) else None)
    return out


def load_representative_values() -> tuple[
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[str, dict[str, str]],
]:
    """
    대표 모형의 사전 계산 결과 두 CSV 를 로드 후 (위험군, 변수) → 값 dict 로 정리.

    반환:
        coef_map  : (위험군, 변수) → 평균 계수
        bw_map    : (위험군, 변수) → bandwidth (MGWR 은 변수마다 다름, 그 외는 단일값)
        metric_map: 위험군 → {대표모형, R², AICc, Residual Moran's I, Moran p}
    """
    params = read_csv(REP_PARAMS)
    summary = read_csv(REP_SUMMARY)

    # 변수별 평균 계수 매핑 (Intercept → CONSTANT 표기 통일)
    coef_map: dict[tuple[str, str], float] = {}
    for _, row in params.iterrows():
        variable = str(row["변수"])
        if variable == "Intercept":
            variable = "CONSTANT"
        coef_map[(str(row["위험군"]), variable)] = float(row["coef_mean"])

    bw_map: dict[tuple[str, str], float] = {}
    metric_map: dict[str, dict[str, str]] = {}
    for _, row in summary.iterrows():
        risk = str(row["위험군"])
        model_name = str(row["대표모형"])
        if model_name == "MGWR":
            # MGWR 의 BW 컬럼은 "{'Intercept': 50, '구조노후도': 80, ...}" 형태 dict 문자열
            bws = ast.literal_eval(str(row["BW"]))
            for key, value in bws.items():
                variable = "CONSTANT" if key == "Intercept" else key
                bw_map[(risk, variable)] = float(value)
        else:
            # GWR 등 단일 BW 모델 — 모든 변수에 동일값 적용
            for variable in ["CONSTANT"] + FEATURES:
                bw_map[(risk, variable)] = float(row["BW"])

        # 표 우측 패널에 표시할 대표 모형 지표
        metric_map[risk] = {
            "대표모형": model_name,
            "R²": f"{float(row['R2']):.3f}",
            "AICc": f"{float(row['AICc']):.3f}",
            "Residual Moran's I": f"{float(row['Residual_Moran_I']):.4f}",
            "Moran p": f"{float(row['Moran_p']):.3f}",
        }

    return coef_map, bw_map, metric_map


def metric_value(model, *names: str) -> str:
    """모델 지표(r2/aic/pr2 등)를 읽어 문자열로 — 후보명 순으로 fallback."""
    val = safe_attr(model, *names)
    if val is None:
        return ""
    try:
        return f"{float(val):.3f}"
    except Exception:
        return str(val)


def build_cluster_table(
    df: pd.DataFrame,
    risk: str,
    coef_map: dict[tuple[str, str], float],
    bw_map: dict[tuple[str, str], float],
    metric_map: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """단일 위험군에 대해 OLS/SLM/SEM 을 적합하고 표 형태 DataFrame 반환."""
    # 해당 위험군 데이터만 + 결측 제거
    work = df[df[CLUSTER_COL] == risk][FEATURES + [TARGET] + COORDS].dropna().copy()
    x = StandardScaler().fit_transform(work[FEATURES])
    # '승인연도' 는 클수록 신축 → 위험과 음의 관계.
    # 표준화 후 부호 반전해 '오래될수록' 양의 효과로 해석되도록 보정.
    x[:, FEATURES.index("승인연도")] *= -1
    y = work[TARGET].to_numpy().reshape(-1, 1)
    coords = work[COORDS].to_numpy()

    # KNN 가중치 — k=12, 표본이 12개 이하면 n-1 로 안전 처리
    k = min(12, len(work) - 1)
    w = KNN.from_array(coords, k=k)
    w.transform = "R"

    # 출력 변수 순서 — 절편 먼저
    term_names = ["CONSTANT"] + FEATURES
    # OLS / Spatial Lag / Spatial Error 적합
    ols = OLS(y, x, w=w, name_y=TARGET, name_x=FEATURES)
    slm = ML_Lag(y, x, w=w, name_y=TARGET, name_x=FEATURES)
    sem = ML_Error(y, x, w=w, name_y=TARGET, name_x=FEATURES)

    ols_terms = model_terms(ols, term_names, "t_stat")
    slm_terms = model_terms(slm, term_names, "z_stat")
    sem_terms = model_terms(sem, term_names, "z_stat")

    # 우측 컬럼 헤더 — 위험군별 대표 모형 + 평균계수(BW)
    rep_model = metric_map[risk]["대표모형"]
    rep_col = f"{risk} {rep_model}\n평균계수(BW)"

    def rep_cell(variable: str) -> str:
        """대표 모형 셀 — 변수명 별칭(승인연도) 처리 후 fmt_coef_bw 호출."""
        key = "승인연도" if variable.startswith("승인연도") else variable
        return fmt_coef_bw(coef_map.get((risk, key)), bw_map.get((risk, key)))

    def make_row(
        model_group: str,
        category: str,
        variable: str,
        ols_val: str,
        slm_val: str,
        sem_val: str,
    ) -> dict[str, str]:
        """행 dict 생성 — 첫 행에만 'Model'/'구분' 표기로 가독성 확보."""
        return {
            "Model": model_group,
            "구분": category,
            "변수": variable,
            "일반회귀(OLS)": ols_val,
            "공간시차(SLM)": slm_val,
            "공간오차(SEM)": sem_val,
            rep_col: rep_cell(variable),
        }

    # 절편(CONSTANT) 행 — 표 첫 줄
    rows: list[dict[str, str]] = [
        make_row(
            "설명변수",
            "",
            "CONSTANT",
            fmt_coef(*ols_terms["CONSTANT"]),
            fmt_coef(*slm_terms["CONSTANT"]),
            fmt_coef(*sem_terms["CONSTANT"]),
        )
    ]

    # 카테고리 → 변수 순회 — 카테고리 첫 행에만 카테고리명 표기
    for category, features in CATEGORIES.items():
        for i, feature in enumerate(features):
            # '승인연도' 부호 반전을 사용자에게 알리는 라벨
            display = "승인연도(오래될수록)" if feature == "승인연도" else feature
            rows.append(
                make_row(
                    "설명변수" if i == 0 else "",
                    category if i == 0 else "",
                    display,
                    fmt_coef(*ols_terms[feature]),
                    fmt_coef(*slm_terms[feature]),
                    fmt_coef(*sem_terms[feature]),
                )
            )

    # 공간시차/오차 파라미터 — spreg 버전마다 속성명이 달라 안전 추출
    rho_attr = safe_attr(slm, "rho")
    rho = float(rho_attr if rho_attr is not None else slm.betas.flatten()[-1])
    slm_pvals = pvals_from_stats(getattr(slm, "z_stat", []))
    rho_p = slm_pvals[-1] if slm_pvals else None
    lam_attr = safe_attr(sem, "lam", "lambda")
    lam = float(lam_attr if lam_attr is not None else sem.betas.flatten()[-1])
    sem_pvals = pvals_from_stats(getattr(sem, "z_stat", []))
    lam_p = sem_pvals[-1] if sem_pvals else None

    # 모형 지표 행들 — n / Rho / Lambda / R² / AIC / Moran I 등
    metric_rows = [
        ("n", f"{len(work):,}", f"{len(work):,}", f"{len(work):,}", f"{len(work):,}"),
        ("Rho", "", fmt_coef(rho, rho_p), "", ""),
        ("Lambda", "", "", fmt_coef(lam, lam_p), ""),
        (
            "R²",
            metric_value(ols, "r2"),
            metric_value(slm, "pr2", "r2"),
            metric_value(sem, "pr2", "r2"),
            metric_map[risk]["R²"],
        ),
        (
            "AIC/AICc",
            metric_value(ols, "aic"),
            metric_value(slm, "aic"),
            metric_value(sem, "aic"),
            metric_map[risk]["AICc"],
        ),
        ("Residual Moran's I", "", "", "", metric_map[risk]["Residual Moran's I"]),
        ("Moran p", "", "", "", metric_map[risk]["Moran p"]),
    ]
    for idx, (label, ols_val, slm_val, sem_val, rep_val) in enumerate(metric_rows):
        rows.append(
            {
                "Model": "모형" if idx == 0 else "",
                "구분": "",
                "변수": label,
                "일반회귀(OLS)": ols_val,
                "공간시차(SLM)": slm_val,
                "공간오차(SEM)": sem_val,
                rep_col: rep_val,
            }
        )

    return pd.DataFrame(rows)


def draw_png(table_df: pd.DataFrame, out_png: Path, risk: str) -> None:
    """DataFrame → matplotlib 표로 렌더링 후 PNG 저장."""
    # 한글 폰트 — 맑은 고딕 등록
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    # 행 수에 비례해 figure 높이 동적 조정
    fig, ax = plt.subplots(
        figsize=(15.5, max(8.8, 0.42 * len(table_df) + 1.25)), dpi=180
    )
    fig.patch.set_facecolor("#FFFFFF")
    ax.axis("off")

    # 메인 타이틀 + 부연 설명
    ax.text(
        0.0,
        1.035,
        f"{risk} OLS → SLM → SEM → GWR/MGWR(BW) 비교",
        transform=ax.transAxes,
        fontsize=21,
        fontweight="bold",
        color="#172033",
        va="bottom",
    )
    ax.text(
        0.0,
        1.006,
        "Y = 최종위험점수_new | X = 표준화 변수 | 해당 위험군 데이터만 사용 | KNN k=12, row-standardized",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#596579",
        va="bottom",
    )

    # 컬럼 폭 비율 — 변수명/모델 컬럼이 더 넓게
    col_widths = [0.09, 0.13, 0.245, 0.13, 0.13, 0.13, 0.145]
    mpl_table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="upper left",
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0.035, 1, 0.94],
        colWidths=col_widths,
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(8.8)

    # 셀 스타일링 — 헤더, 줄무늬, 모형 지표 영역, 변수명 좌측정렬
    for (r, c), cell in mpl_table.get_celld().items():
        cell.set_edgecolor("#8B929C")
        cell.set_linewidth(0.55)
        if r == 0:
            cell.set_facecolor("#EAF0F8")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color("#172033")
            continue
        if r % 2 == 0:
            cell.set_facecolor("#F7F9FC")
        else:
            cell.set_facecolor("#FFFFFF")
        # 11행 이후는 모형 지표 — 더 옅은 배경
        if r > 11:
            cell.set_facecolor("#FAFBFD")
        # 변수명 컬럼만 좌측정렬
        if c == 2:
            cell.get_text().set_ha("left")
            cell.PAD = 0.02

    # 별표 의미 안내 footer
    ax.text(
        0,
        0.005,
        "*** p<0.01, **p<0.05, *p<0.1",
        transform=ax.transAxes,
        fontsize=10,
        color="#343A46",
    )
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    """3개 위험군 각각에 대해 비교표 CSV/PNG 생성."""
    df = read_csv(DATA)
    coef_map, bw_map, metric_map = load_representative_values()
    for risk in ORDER:
        table = build_cluster_table(df, risk, coef_map, bw_map, metric_map)
        out_csv = OUT_DIR / f"{risk}_OLS_SLM_SEM_GWR_MGWR_BW_비교표.csv"
        out_png = OUT_DIR / f"{risk}_OLS_SLM_SEM_GWR_MGWR_BW_비교표.png"
        table.to_csv(out_csv, index=False, encoding="utf-8-sig")
        draw_png(table, out_png, risk)
        print(out_csv)
        print(out_png)


if __name__ == "__main__":
    main()
