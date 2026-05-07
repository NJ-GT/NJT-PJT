# -*- coding: utf-8 -*-
from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from libpysal.weights import KNN
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler
from spreg import ML_Error, ML_Lag, OLS


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "0430" / "최종테이블0429.csv"
OUT_DIR = ROOT / "0430"
REP_PARAMS = OUT_DIR / "군집별_대표모형_params_평균계수.csv"
REP_SUMMARY = OUT_DIR / "가중치군집_대표모형_최종결과표.csv"

TARGET = "최종위험점수_new"
CLUSTER_COL = "cluster_label"
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

ORDER = ["저위험군", "중위험군", "고위험군"]


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def star(p: float | None) -> str:
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
    if v is None or pd.isna(v):
        return ""
    return f"{v:.3f}{star(p)}"


def fmt_coef_bw(coef: float | None, bw: float | None) -> str:
    if coef is None or pd.isna(coef):
        return ""
    if bw is None or pd.isna(bw):
        return f"{coef:.3f}"
    return f"{coef:.3f}\nBW={bw:g}"


def pvals_from_stats(stats) -> list[float | None]:
    vals: list[float | None] = []
    for item in stats:
        try:
            vals.append(float(item[1]))
        except Exception:
            vals.append(None)
    return vals


def safe_attr(model, *names: str):
    for name in names:
        if hasattr(model, name):
            return getattr(model, name)
    return None


def model_terms(model, term_names: list[str], stat_attr: str) -> dict[str, tuple[float, float | None]]:
    betas = [float(x) for x in model.betas.flatten()]
    pvals = pvals_from_stats(getattr(model, stat_attr, []))
    out: dict[str, tuple[float, float | None]] = {}
    for idx, term in enumerate(term_names):
        out[term] = (betas[idx], pvals[idx] if idx < len(pvals) else None)
    return out


def load_representative_values() -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], dict[str, dict[str, str]]]:
    params = read_csv(REP_PARAMS)
    summary = read_csv(REP_SUMMARY)

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
            bws = ast.literal_eval(str(row["BW"]))
            for key, value in bws.items():
                variable = "CONSTANT" if key == "Intercept" else key
                bw_map[(risk, variable)] = float(value)
        else:
            for variable in ["CONSTANT"] + FEATURES:
                bw_map[(risk, variable)] = float(row["BW"])

        metric_map[risk] = {
            "대표모형": model_name,
            "R²": f"{float(row['R2']):.3f}",
            "AICc": f"{float(row['AICc']):.3f}",
            "Residual Moran's I": f"{float(row['Residual_Moran_I']):.4f}",
            "Moran p": f"{float(row['Moran_p']):.3f}",
        }

    return coef_map, bw_map, metric_map


def metric_value(model, *names: str) -> str:
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
    work = df[df[CLUSTER_COL] == risk][FEATURES + [TARGET] + COORDS].dropna().copy()
    x = StandardScaler().fit_transform(work[FEATURES])
    x[:, FEATURES.index("승인연도")] *= -1
    y = work[TARGET].to_numpy().reshape(-1, 1)
    coords = work[COORDS].to_numpy()

    # k cannot exceed n-1. Current clusters are large, but keep it safe.
    k = min(12, len(work) - 1)
    w = KNN.from_array(coords, k=k)
    w.transform = "R"

    term_names = ["CONSTANT"] + FEATURES
    ols = OLS(y, x, w=w, name_y=TARGET, name_x=FEATURES)
    slm = ML_Lag(y, x, w=w, name_y=TARGET, name_x=FEATURES)
    sem = ML_Error(y, x, w=w, name_y=TARGET, name_x=FEATURES)

    ols_terms = model_terms(ols, term_names, "t_stat")
    slm_terms = model_terms(slm, term_names, "z_stat")
    sem_terms = model_terms(sem, term_names, "z_stat")

    rep_model = metric_map[risk]["대표모형"]
    rep_col = f"{risk} {rep_model}\n평균계수(BW)"

    def rep_cell(variable: str) -> str:
        key = "승인연도" if variable.startswith("승인연도") else variable
        return fmt_coef_bw(coef_map.get((risk, key)), bw_map.get((risk, key)))

    def make_row(model_group: str, category: str, variable: str, ols_val: str, slm_val: str, sem_val: str) -> dict[str, str]:
        return {
            "Model": model_group,
            "구분": category,
            "변수": variable,
            "일반회귀(OLS)": ols_val,
            "공간시차(SLM)": slm_val,
            "공간오차(SEM)": sem_val,
            rep_col: rep_cell(variable),
        }

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

    for category, features in CATEGORIES.items():
        for i, feature in enumerate(features):
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

    rho_attr = safe_attr(slm, "rho")
    rho = float(rho_attr if rho_attr is not None else slm.betas.flatten()[-1])
    slm_pvals = pvals_from_stats(getattr(slm, "z_stat", []))
    rho_p = slm_pvals[-1] if slm_pvals else None
    lam_attr = safe_attr(sem, "lam", "lambda")
    lam = float(lam_attr if lam_attr is not None else sem.betas.flatten()[-1])
    sem_pvals = pvals_from_stats(getattr(sem, "z_stat", []))
    lam_p = sem_pvals[-1] if sem_pvals else None

    metric_rows = [
        ("n", f"{len(work):,}", f"{len(work):,}", f"{len(work):,}", f"{len(work):,}"),
        ("Rho", "", fmt_coef(rho, rho_p), "", ""),
        ("Lambda", "", "", fmt_coef(lam, lam_p), ""),
        ("R²", metric_value(ols, "r2"), metric_value(slm, "pr2", "r2"), metric_value(sem, "pr2", "r2"), metric_map[risk]["R²"]),
        ("AIC/AICc", metric_value(ols, "aic"), metric_value(slm, "aic"), metric_value(sem, "aic"), metric_map[risk]["AICc"]),
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
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(15.5, max(8.8, 0.42 * len(table_df) + 1.25)), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")
    ax.axis("off")

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
        if r > 11:
            cell.set_facecolor("#FAFBFD")
        if c == 2:
            cell.get_text().set_ha("left")
            cell.PAD = 0.02

    ax.text(0, 0.005, "*** p<0.01, **p<0.05, *p<0.1", transform=ax.transAxes, fontsize=10, color="#343A46")
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
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
