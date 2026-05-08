# -*- coding: utf-8 -*-
"""
전체 데이터에 대한 OLS/SLM/SEM 결과와 군집별 GWR/MGWR 대표 모형 평균계수를
한 비교표(CSV/PNG)로 통합 생성.

표 구성:
    행 — CONSTANT + 카테고리 그룹(위험·접근성/건축·시설/입지·밀집)별 변수 + 모형 지표(Rho/Lambda/R²/AIC/Moran I/p)
    열 — 일반회귀(OLS) | 공간시차(SLM) | 공간오차(SEM) | 저/중/고 위험군 GWR/MGWR(BW)

목적:
    - 공간 효과 도입(SLM/SEM/GWR/MGWR)에 따른 변수 효과 변화와 잔차 자기상관 변화를
      한 표로 비교 가능하게 만들어 발표/보고서 활용.

입력:
    - 0430/최종테이블0429.csv                       : 분석 마스터
    - 0430/군집별_대표모형_params_평균계수.csv      : 군집별 평균 계수
    - 0430/가중치군집_대표모형_최종결과표.csv       : 군집별 BW + 지표

출력:
    - 0430/OLS_SLM_SEM_GWR_MGWR_BW_비교표.csv
    - 0430/OLS_SLM_SEM_GWR_MGWR_BW_비교표.png
"""
from __future__ import annotations

import ast  # MGWR BW dict 문자열 파싱
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from libpysal.weights import KNN
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler
from spreg import ML_Error, ML_Lag, OLS


# 경로
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "0430" / "최종테이블0429.csv"
OUT_DIR = ROOT / "0430"
# 작은 표 (OLS/SLM/SEM 만) — 별도 산출 시 사용
OUT_CSV = OUT_DIR / "OLS_SLM_SEM_변수계수_비교표.csv"
OUT_PNG = OUT_DIR / "OLS_SLM_SEM_변수계수_비교표.png"
# 통합 표 (GWR/MGWR 포함)
OUT_CSV_FULL = OUT_DIR / "OLS_SLM_SEM_GWR_MGWR_BW_비교표.csv"
OUT_PNG_FULL = OUT_DIR / "OLS_SLM_SEM_GWR_MGWR_BW_비교표.png"
# 대표 모형 사전 계산 결과
REP_PARAMS = OUT_DIR / "군집별_대표모형_params_평균계수.csv"
REP_SUMMARY = OUT_DIR / "가중치군집_대표모형_최종결과표.csv"

# 분석 변수 + 좌표
TARGET = "최종위험점수_new"
COORDS = ["x_5181", "y_5181"]
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

# 가독성을 위해 변수를 3개 카테고리로 묶음
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


def read_csv(path: Path) -> pd.DataFrame:
    """다중 인코딩 fallback 으로 CSV 안전 로드."""
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def star(p: float | None) -> str:
    """p값 → *** / ** / * 별표 표기."""
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
    """계수 + 별표 문자열."""
    if v is None or pd.isna(v):
        return ""
    return f"{v:.3f}{star(p)}"


def fmt_coef_bw(coef: float | None, bw: float | None) -> str:
    """대표 모형 셀 — 계수 줄바꿈 후 'BW=값' 부착 (없으면 계수만)."""
    if coef is None or pd.isna(coef):
        return ""
    if bw is None or pd.isna(bw):
        return f"{coef:.3f}"
    return f"{coef:.3f}\nBW={bw:g}"


def pvals_from_stats(stats) -> list[float | None]:
    """spreg 의 (z, p) 튜플 리스트에서 p값만 추출."""
    vals: list[float | None] = []
    for item in stats:
        try:
            vals.append(float(item[1]))
        except Exception:
            vals.append(None)
    return vals


def model_terms(
    model, term_names: list[str], stat_attr: str
) -> dict[str, tuple[float, float | None]]:
    """모델의 betas + p값을 변수명 dict 로 매핑."""
    betas = [float(x) for x in model.betas.flatten()]
    pvals = pvals_from_stats(getattr(model, stat_attr, []))
    out: dict[str, tuple[float, float | None]] = {}
    for idx, term in enumerate(term_names):
        out[term] = (betas[idx], pvals[idx] if idx < len(pvals) else None)
    return out


def safe_attr(model, *names: str):
    """후보 속성명들 중 존재하는 첫 값을 반환 — spreg 버전 호환."""
    for name in names:
        if hasattr(model, name):
            return getattr(model, name)
    return None


def load_representative_model_values() -> tuple[
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[str, dict[str, str]],
]:
    """
    군집별 대표 모형 사전 산출 결과 로드.

    반환:
        coef_map  : (위험군, 변수) → 평균 계수
        bw_map    : (위험군, 변수) → bandwidth (MGWR 은 변수마다, GWR 은 단일값)
        metric_map: 위험군 → {model, R², AIC, Moran's I, Moran p}
    """
    params = read_csv(REP_PARAMS)
    summary = read_csv(REP_SUMMARY)

    # 변수별 평균 계수
    coef_map: dict[tuple[str, str], float] = {}
    for _, row in params.iterrows():
        variable = str(row["변수"])
        if variable == "Intercept":
            variable = "CONSTANT"
        coef_map[(str(row["위험군"]), variable)] = float(row["coef_mean"])

    # BW + 모형 지표
    bw_map: dict[tuple[str, str], float] = {}
    metric_map: dict[str, dict[str, str]] = {}
    for _, row in summary.iterrows():
        risk = str(row["위험군"])
        model = str(row["대표모형"])
        bw_raw = row["BW"]
        if model == "MGWR":
            # MGWR BW 컬럼 — dict 문자열 파싱
            bws = ast.literal_eval(str(bw_raw))
            for key, value in bws.items():
                variable = "CONSTANT" if key == "Intercept" else key
                bw_map[(risk, variable)] = float(value)
        else:
            # GWR — 단일 BW 를 모든 변수에 동일 적용
            for variable in ["CONSTANT"] + FEATURES:
                bw_map[(risk, variable)] = float(bw_raw)
        metric_map[risk] = {
            "model": model,
            "R²": f"{float(row['R2']):.3f}",
            "AIC": f"{float(row['AICc']):.3f}",
            "Moran's I": f"{float(row['Residual_Moran_I']):.4f}",
            "Moran p": f"{float(row['Moran_p']):.3f}",
        }
    return coef_map, bw_map, metric_map


def build_table() -> pd.DataFrame:
    """전체 데이터로 OLS/SLM/SEM 적합 + 군집별 대표값과 결합한 비교표 DataFrame 반환."""
    df = read_csv(DATA)
    work = df[FEATURES + [TARGET] + COORDS].dropna().copy()

    # 표준화 입력
    x = StandardScaler().fit_transform(work[FEATURES])
    # '승인연도' 부호 반전 — 발표 해석상 "오래될수록 위험"이 양의 효과로 보이게
    x[:, FEATURES.index("승인연도")] *= -1
    y = work[TARGET].to_numpy().reshape(-1, 1)
    coords = work[COORDS].to_numpy()

    # KNN k=12 + 행 표준화 가중치
    w = KNN.from_array(coords, k=12)
    w.transform = "R"

    # 변수 표시명 — 부호 반전한 승인연도는 라벨로 명시
    term_names = ["CONSTANT"] + FEATURES
    display_names = ["CONSTANT"] + [
        "승인연도(오래될수록)" if f == "승인연도" else f for f in FEATURES
    ]

    # 3개 모형 적합 — KNN 가중치 동일하게 사용
    ols = OLS(y, x, w=w, name_y=TARGET, name_x=FEATURES)
    slm = ML_Lag(y, x, w=w, name_y=TARGET, name_x=FEATURES)
    sem = ML_Error(y, x, w=w, name_y=TARGET, name_x=FEATURES)

    ols_terms = model_terms(ols, term_names, "t_stat")
    slm_terms = model_terms(slm, term_names, "z_stat")
    sem_terms = model_terms(sem, term_names, "z_stat")

    # 대표 모형 결과 결합
    coef_map, bw_map, metric_map = load_representative_model_values()

    def rep_cell(risk: str, variable: str) -> str:
        """위험군×변수 셀 — fmt_coef_bw 로 계수+BW 문자열 생성."""
        return fmt_coef_bw(coef_map.get((risk, variable)), bw_map.get((risk, variable)))

    def make_row(
        model: str,
        category: str,
        variable: str,
        ols_val: str,
        slm_val: str,
        sem_val: str,
    ) -> dict[str, str]:
        """행 dict 생성 — 위험군 3개 컬럼은 rep_cell 로 자동 채움."""
        # 표시명에 "(오래될수록)" 붙은 경우는 dict 키로는 원래 변수명 사용
        variable_key = "승인연도" if variable.startswith("승인연도") else variable
        return {
            "Model": model,
            "구분": category,
            "변수": variable,
            "일반회귀(OLS)": ols_val,
            "공간시차(SLM)": slm_val,
            "공간오차(SEM)": sem_val,
            "저위험군 MGWR\n평균계수(BW)": rep_cell("저위험군", variable_key),
            "중위험군 GWR\n평균계수(BW)": rep_cell("중위험군", variable_key),
            "고위험군 MGWR\n평균계수(BW)": rep_cell("고위험군", variable_key),
        }

    rows = []
    # 절편 행
    rows.append(
        make_row(
            "설명변수",
            "",
            "CONSTANT",
            fmt_coef(*ols_terms["CONSTANT"]),
            fmt_coef(*slm_terms["CONSTANT"]),
            fmt_coef(*sem_terms["CONSTANT"]),
        )
    )

    # 카테고리 → 변수 — 카테고리 첫 행에만 카테고리 라벨 표기
    for cat, features in CATEGORIES.items():
        for i, feature in enumerate(features):
            display = "승인연도(오래될수록)" if feature == "승인연도" else feature
            rows.append(
                make_row(
                    "설명변수" if i == 0 else "",
                    cat if i == 0 else "",
                    display,
                    fmt_coef(*ols_terms[feature]),
                    fmt_coef(*slm_terms[feature]),
                    fmt_coef(*sem_terms[feature]),
                )
            )

    # SLM 의 ρ (공간시차) — 버전 호환 처리
    rho_attr = safe_attr(slm, "rho")
    rho = float(rho_attr if rho_attr is not None else slm.betas.flatten()[-1])
    slm_pvals = pvals_from_stats(getattr(slm, "z_stat", []))
    rho_p = slm_pvals[-1] if slm_pvals else None
    # SEM 의 λ (공간오차)
    lam_attr = safe_attr(sem, "lam", "lambda")
    lam = float(lam_attr if lam_attr is not None else sem.betas.flatten()[-1])
    sem_pvals = pvals_from_stats(getattr(sem, "z_stat", []))
    lam_p = sem_pvals[-1] if sem_pvals else None

    # 모형 지표 행들 — Rho/Lambda/R²/AIC/Moran I/Moran p
    metric_rows = [
        ("Rho", "", fmt_coef(rho, rho_p), "", "", "", ""),
        ("Lambda", "", "", fmt_coef(lam, lam_p), "", "", ""),
        (
            "R²",
            f"{float(safe_attr(ols, 'r2')):.3f}",
            f"{float(safe_attr(slm, 'pr2', 'r2')):.3f}",
            f"{float(safe_attr(sem, 'pr2', 'r2')):.3f}",
            metric_map["저위험군"]["R²"],
            metric_map["중위험군"]["R²"],
            metric_map["고위험군"]["R²"],
        ),
        (
            "AIC/AICc",
            f"{float(safe_attr(ols, 'aic')):.3f}",
            f"{float(safe_attr(slm, 'aic')):.3f}",
            f"{float(safe_attr(sem, 'aic')):.3f}",
            metric_map["저위험군"]["AIC"],
            metric_map["중위험군"]["AIC"],
            metric_map["고위험군"]["AIC"],
        ),
        (
            "Residual Moran's I",
            "",
            "",
            "",
            metric_map["저위험군"]["Moran's I"],
            metric_map["중위험군"]["Moran's I"],
            metric_map["고위험군"]["Moran's I"],
        ),
        (
            "Moran p",
            "",
            "",
            "",
            metric_map["저위험군"]["Moran p"],
            metric_map["중위험군"]["Moran p"],
            metric_map["고위험군"]["Moran p"],
        ),
    ]
    for idx, (
        label,
        ols_val,
        slm_val,
        sem_val,
        low_val,
        mid_val,
        high_val,
    ) in enumerate(metric_rows):
        rows.append(
            {
                "Model": "모형" if idx == 0 else "",
                "구분": "",
                "변수": label,
                "일반회귀(OLS)": ols_val,
                "공간시차(SLM)": slm_val,
                "공간오차(SEM)": sem_val,
                "저위험군 MGWR\n평균계수(BW)": low_val,
                "중위험군 GWR\n평균계수(BW)": mid_val,
                "고위험군 MGWR\n평균계수(BW)": high_val,
            }
        )

    result = pd.DataFrame(rows)
    return result


def draw_png(table_df: pd.DataFrame, out_png: Path = OUT_PNG) -> None:
    """DataFrame → matplotlib 표 → PNG 저장."""
    # 한글 폰트 — 맑은 고딕 등록
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    # figure 크기 — 행 수와 컬럼 수에 따라 자동 조정
    fig_h = max(8.5, 0.38 * len(table_df) + 1.2)
    fig_w = 18.5 if len(table_df.columns) > 6 else 13.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")
    ax.axis("off")

    # 메인 타이틀 + 부연 설명
    ax.text(
        0.0,
        1.035,
        "OLS → SLM → SEM → GWR/MGWR(BW) 공간모형 비교",
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        color="#172033",
        va="bottom",
    )
    ax.text(
        0.0,
        1.005,
        "Y = 최종위험점수_new | X = 표준화 변수 | KNN k=12, row-standardized | GWR/MGWR는 군집별 최종 대표모형의 평균계수와 BW",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#596579",
        va="bottom",
    )

    # 컬럼 폭 — 통합표(9컬럼) vs 작은표(6컬럼) 분기
    if len(table_df.columns) > 6:
        col_widths = [0.075, 0.105, 0.19, 0.105, 0.105, 0.105, 0.105, 0.105, 0.105]
    else:
        col_widths = [0.105, 0.15, 0.245, 0.165, 0.165, 0.165]
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
    mpl_table.set_fontsize(8.2 if len(table_df.columns) > 6 else 9.3)

    # 셀 스타일 — 헤더/줄무늬/모형 지표 영역/변수명 정렬
    for (r, c), cell in mpl_table.get_celld().items():
        cell.set_edgecolor("#8B929C")
        cell.set_linewidth(0.55)
        if r == 0:
            cell.set_facecolor("#EAF0F8")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color("#172033")
        else:
            # 모형 지표 영역(Model='모형' 행 이후) 또는 12행 이후는 더 옅은 배경
            model_label = table_df.iloc[r - 1]["Model"]
            if (
                model_label == "모형"
                or (r > 1 and table_df.iloc[r - 2]["Model"] == "모형")
                or r > 12
            ):
                cell.set_facecolor("#FAFBFD")
            elif r % 2 == 0:
                cell.set_facecolor("#F7F9FC")
            else:
                cell.set_facecolor("#FFFFFF")

            # 좌측 식별 컬럼은 진한 글자색
            if c in (0, 1, 2):
                cell.get_text().set_color("#172033")
            # 변수명 컬럼만 좌측 정렬
            if c == 2:
                cell.get_text().set_ha("left")
                cell.PAD = 0.02

    # 별표 의미 안내
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
    """비교표 생성 후 통합 CSV/PNG 저장."""
    table_df = build_table()
    table_df.to_csv(OUT_CSV_FULL, index=False, encoding="utf-8-sig")
    draw_png(table_df, OUT_PNG_FULL)
    print(OUT_CSV_FULL)
    print(OUT_PNG_FULL)


if __name__ == "__main__":
    main()
