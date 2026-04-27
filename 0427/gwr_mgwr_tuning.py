# -*- coding: utf-8 -*-
"""
그룹별 GWR/MGWR 튜닝
  그룹 A: 기존숙박군 (숙박업 + 관광숙박업)
  그룹 B: 외국인관광도시민박업

  각 그룹에 대해:
    STEP 1 │ 상관·VIF 확인
    STEP 2 │ GWR Bandwidth 비교
    STEP 3 │ 최적 BW 자동 선택 (AICc 기준)
    STEP 4 │ MGWR — 변수별 최적 Bandwidth
    STEP 5 │ 결과 저장 (data/gwr_results_{slug}.csv)
"""
import glob
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from sklearn.preprocessing import StandardScaler

# ── 데이터 로드 ──────────────────────────────────────────────────────
SRC = glob.glob("C:/Users/USER/Documents/GitHub/*/NJT-PJT/0424/*/tables/*AHP3*.csv")[0]
df_all = pd.read_csv(SRC, encoding="utf-8-sig")

BASE = "C:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT"

RISK_VARS = ["구조노후도", "단속위험도", "도로폭위험도", "집중도", "주변건물수"]
TARGET = "위험점수_AHP"

for v in RISK_VARS + [TARGET]:
    df_all[v] = pd.to_numeric(df_all[v], errors="coerce")

GROUPS = {
    "기존숙박군": "A_기존숙박군",
    "외국인관광도시민박업": "B_외국인민박",
}


def run_group(group_key: str, slug: str) -> None:
    df = df_all[df_all["업종그룹"] == group_key].dropna(
        subset=RISK_VARS + [TARGET, "위도", "경도"]
    ).reset_index(drop=True)

    label = "기존숙박군 (숙박업+관광숙박업)" if group_key == "기존숙박군" else "외국인관광도시민박업"
    print(f"\n\n{'█'*65}")
    print(f"  그룹: {label}  N={len(df):,}")
    print("█" * 65)

    # ── STEP 1 │ 상관·VIF ────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("STEP 1 │ 변수 상관·VIF")
    print("━" * 60)

    Xv = StandardScaler().fit_transform(df[RISK_VARS])
    corr = df[RISK_VARS].corr().round(3)
    print("\n  상관행렬:")
    print(corr.to_string(col_space=14))

    try:
        vif = np.diag(len(Xv) * np.linalg.inv(Xv.T @ Xv))
        print("\n  VIF:")
        for var, vi in zip(RISK_VARS, vif):
            flag = "  ← 주의" if vi > 5 else ""
            print(f"    {var:<20} {vi:.2f}{flag}")
    except Exception as e:
        print(f"  VIF 오류: {e}")

    # 집중도-주변건물수 상관 확인
    corr_val = abs(corr.loc["집중도", "주변건물수"])
    if corr_val > 0.7:
        FINAL_VARS = ["구조노후도", "단속위험도", "도로폭위험도", "주변건물수"]
        print(f"\n  → 집중도↔주변건물수 상관={corr_val:.3f}>0.7: 집중도 제외, 4변수 사용")
    else:
        FINAL_VARS = RISK_VARS
        print(f"\n  → 상관 양호: 5변수 전부 사용")
    print(f"  최종 변수: {FINAL_VARS}")

    # ── 공통 샘플 설정 ───────────────────────────────────────────────
    N_GWR = min(2000, len(df))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(df), N_GWR, replace=False)

    coords = df[["위도", "경도"]].values[idx]
    yr = df[TARGET].values[idx].reshape(-1, 1)
    Xr = StandardScaler().fit_transform(df[FINAL_VARS].values[idx])
    print(f"\n  GWR 샘플: {len(idx):,}개")

    # ── STEP 2 │ GWR Bandwidth 비교 ──────────────────────────────────
    print("\n" + "━" * 60)
    print("STEP 2 │ GWR Bandwidth 비교 (bisquare adaptive NN)")
    print("━" * 60)
    print(f"\n  {'BW':>5}  {'R²':>8}  {'adj.R²':>8}  {'AICc':>10}  시간")

    bw_results = {}
    for bw in [40, 80, 120, 200]:
        if bw >= len(idx):
            continue
        t0 = time.time()
        res = GWR(coords, yr, Xr, bw=bw, kernel="bisquare", fixed=False).fit()
        t1 = time.time()
        aicc = getattr(res, "AIC", None) or getattr(res, "aicc", None) or float("nan")
        bw_results[bw] = res
        print(f"  {bw:>5}  {res.R2:>8.4f}  {res.adj_R2:>8.4f}  {aicc:>10.2f}  {t1-t0:.1f}s")

    # ── STEP 3 │ 최적 BW 자동 선택 ───────────────────────────────────
    print("\n" + "━" * 60)
    print("STEP 3 │ 최적 BW 자동 선택 (golden section, AICc 기준)")
    print("━" * 60)

    t0 = time.time()
    bw_sel = Sel_BW(coords, yr, Xr, kernel="bisquare", fixed=False)
    best_bw = bw_sel.search(search_method="golden_section")
    t1 = time.time()
    print(f"\n  최적 BW = {int(best_bw)}  ({t1-t0:.1f}s)")

    gwr_best = GWR(coords, yr, Xr, bw=best_bw, kernel="bisquare", fixed=False).fit()
    aicc_best = getattr(gwr_best, "AIC", None) or getattr(gwr_best, "aicc", None)
    print(f"  R²={gwr_best.R2:.4f}  adj.R²={gwr_best.adj_R2:.4f}  AICc={aicc_best:.2f}")
    print("\n  계수 변동 (mean ± std):")
    for i, v in enumerate(FINAL_VARS):
        p = gwr_best.params[:, i + 1]
        print(f"    {v:<20} mean={p.mean():+.4f}  std={p.std():.4f}  [{p.min():+.3f}~{p.max():+.3f}]")

    # ── STEP 4 │ MGWR — 변수별 BW ────────────────────────────────────
    print("\n" + "━" * 60)
    print("STEP 4 │ MGWR — 변수별 최적 Bandwidth")
    print("━" * 60)

    N_MGWR = min(500, len(idx))
    m_idx = rng.choice(len(idx), N_MGWR, replace=False)
    c_mgwr, y_mgwr, X_mgwr = coords[m_idx], yr[m_idx], Xr[m_idx]
    print(f"\n  MGWR 샘플: {N_MGWR}개")

    MGWR_OK = False
    try:
        t0 = time.time()
        mgwr_sel = Sel_BW(c_mgwr, y_mgwr, X_mgwr, multi=True, kernel="bisquare", fixed=False)
        mgwr_sel.search(verbose=False)
        t1 = time.time()

        bws_multi = mgwr_sel.bw
        print(f"  변수별 최적 BW ({t1-t0:.1f}s):")
        for v, bw_v in zip(["Intercept"] + FINAL_VARS, bws_multi):
            print(f"    {v:<20} BW={int(bw_v)}")

        mgwr_res = MGWR(c_mgwr, y_mgwr, X_mgwr, mgwr_sel, kernel="bisquare", fixed=False).fit()
        aicc_mgwr = getattr(mgwr_res, "AIC", None) or getattr(mgwr_res, "aicc", None)
        print(f"\n  MGWR R²={mgwr_res.R2:.4f}  adj.R²={mgwr_res.adj_R2:.4f}  AICc={aicc_mgwr:.2f}")
        print("\n  MGWR 계수 변동 (mean ± std):")
        for i, v in enumerate(FINAL_VARS):
            p = mgwr_res.params[:, i + 1]
            print(f"    {v:<20} mean={p.mean():+.4f}  std={p.std():.4f}")
        MGWR_OK = True
    except Exception as e:
        print(f"  MGWR 오류: {e}")

    # ── STEP 5 │ 최적 GWR 결과 저장 ──────────────────────────────────
    print("\n" + "━" * 60)
    print("STEP 5 │ 결과 저장")
    print("━" * 60)

    out = pd.DataFrame({
        "위도": coords[:, 0],
        "경도": coords[:, 1],
        "local_R2": gwr_best.localR2.flatten(),
        "bandwidth": int(best_bw),
        "group": group_key,
    })
    for i, v in enumerate(FINAL_VARS):
        out[f"coef_{v}"] = gwr_best.params[:, i + 1]
        out[f"tval_{v}"] = gwr_best.tvalues[:, i + 1]

    out_path = f"{BASE}/data/gwr_results_{slug}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  저장: {out_path}  ({len(out)}행, BW={int(best_bw)})")


# ── 메인 ─────────────────────────────────────────────────────────────
for group_key, slug in GROUPS.items():
    run_group(group_key, slug)

print(f"\n\n{'█'*65}")
print("  전체 완료")
print("█" * 65)
