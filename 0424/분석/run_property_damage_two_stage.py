# -*- coding: utf-8 -*-
"""Two-stage property-damage modeling around lodgings.

Stage 1: whether any fire with positive property damage exists within 50m.
Stage 2: conditional severity model for positive lodgings using log1p(total damage).

Train domain: 기존숙박군(숙박업 + 관광숙박업)
Test domain: 외국인관광도시민박업
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor


SCRIPT_PATH = Path(__file__).resolve()
ANALYSIS_DIR = SCRIPT_PATH.parent
APRIL_DIR = ANALYSIS_DIR.parent
PROJECT_DIR = APRIL_DIR.parent

LODGING_CSV = APRIL_DIR / "분석" / "tables" / "분석변수_최종테이블0423_AHP3등급비교.csv"
FIRE_CSV = PROJECT_DIR / "data" / "화재출동" / "화재출동_2021_2024.csv"

TABLE_DIR = ANALYSIS_DIR / "tables"
FIG_DIR = ANALYSIS_DIR / "figures"

ANALYSIS_YEAR_START = 2021
ANALYSIS_YEAR_END = 2024
RADIUS_OPTIONS = [50, 100, 150, 200, 300]
TARGET_RADIUS = 50
TRAIN_GROUP = "기존숙박군"
TEST_GROUP = "외국인관광도시민박업"

TARGET_EXISTS = f"r{TARGET_RADIUS}m_property_fire_exists"
TARGET_DAMAGE_SUM = f"r{TARGET_RADIUS}m_property_damage_sum_천원"
TARGET_DAMAGE_LOG = f"log1p_{TARGET_DAMAGE_SUM}"

NUMERIC_FEATURES = [
    "승인연도",
    "주변건물수",
    "집중도",
    "단속위험도",
    "구조노후도",
    "도로폭위험도",
    "위도",
    "경도",
    "총층수",
    "시설규모/연면적",
    "위험점수_AHP",
]
CATEGORICAL_FEATURES = ["구", "동", "업종", "건물용도명", "업종그룹"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def setup_style() -> None:
    font_names = {f.name for f in fm.fontManager.ttflist}
    selected_font = "Malgun Gothic"
    for font in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if font in font_names:
            selected_font = font
            break

    plt.rcParams["font.family"] = selected_font
    plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "#fcfcfb"
    plt.rcParams["axes.facecolor"] = "#fcfcfb"
    plt.rcParams["savefig.facecolor"] = "#fcfcfb"


def ensure_dirs() -> None:
    for path in (TABLE_DIR, FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def read_csv_robust(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = (
            out[col]
            .astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .replace("", pd.NA)
        )
    return out


def prepare_lodging_data(path: Path) -> pd.DataFrame:
    df = clean_text_columns(read_csv_robust(path))
    numeric_cols = NUMERIC_FEATURES
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("미상").astype(str)
    df = df.dropna(subset=["위도", "경도", "업종그룹"]).copy()
    return df.reset_index(drop=True)


def prepare_fire_data(path: Path) -> pd.DataFrame:
    df = clean_text_columns(read_csv_robust(path))
    for col in ("재산피해액(천원)", "발생연도", "위도", "경도"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        df["발생연도"].between(ANALYSIS_YEAR_START, ANALYSIS_YEAR_END, inclusive="both")
        & df["재산피해액(천원)"].gt(0)
        & df["위도"].notna()
        & df["경도"].notna()
    ].copy()
    return df.reset_index(drop=True)


def build_balltree(lat_lon_deg: np.ndarray) -> BallTree:
    return BallTree(np.deg2rad(lat_lon_deg), metric="haversine")


def radius_diagnostics(lodging: pd.DataFrame, fire: pd.DataFrame) -> pd.DataFrame:
    lodging_coords = lodging[["위도", "경도"]].to_numpy()
    fire_coords = fire[["위도", "경도"]].to_numpy()
    tree = build_balltree(fire_coords)
    earth_radius = 6_371_000.0

    rows: list[dict[str, object]] = []
    for radius in RADIUS_OPTIONS:
        idx = tree.query_radius(np.deg2rad(lodging_coords), r=radius / earth_radius)
        counts = np.array([len(i) for i in idx], dtype=int)
        temp = lodging[["업종그룹"]].copy()
        temp["hit"] = (counts > 0).astype(int)

        for group_name, group_df in temp.groupby("업종그룹"):
            rows.append(
                {
                    "radius_m": radius,
                    "group": group_name,
                    "positive_share": group_df["hit"].mean(),
                    "positive_count": int(group_df["hit"].sum()),
                    "rows": int(len(group_df)),
                }
            )

        rows.append(
            {
                "radius_m": radius,
                "group": "전체",
                "positive_share": float((counts > 0).mean()),
                "positive_count": int((counts > 0).sum()),
                "rows": int(len(counts)),
            }
        )

    return pd.DataFrame(rows)


def attach_targets(lodging: pd.DataFrame, fire: pd.DataFrame, radius_m: int = TARGET_RADIUS) -> tuple[pd.DataFrame, dict[str, float]]:
    out = lodging.copy()
    fire_coords = fire[["위도", "경도"]].to_numpy()
    fire_damage = fire["재산피해액(천원)"].to_numpy(dtype=float)
    earth_radius = 6_371_000.0

    tree = build_balltree(fire_coords)
    lodging_coords = out[["위도", "경도"]].to_numpy()
    idx = tree.query_radius(np.deg2rad(lodging_coords), r=radius_m / earth_radius)
    nearest_dist, _ = tree.query(np.deg2rad(lodging_coords), k=1)
    nearest_dist_m = nearest_dist[:, 0] * earth_radius

    fire_count = np.array([len(i) for i in idx], dtype=int)
    damage_sum = np.array([fire_damage[i].sum() if len(i) else 0.0 for i in idx], dtype=float)
    damage_max = np.array([fire_damage[i].max() if len(i) else 0.0 for i in idx], dtype=float)

    out[f"r{radius_m}m_property_fire_count"] = fire_count
    out[TARGET_EXISTS] = (fire_count > 0).astype(int)
    out[TARGET_DAMAGE_SUM] = damage_sum
    out[TARGET_DAMAGE_LOG] = np.log1p(damage_sum)
    out[f"r{radius_m}m_property_damage_max_천원"] = damage_max
    out[f"nearest_property_fire_distance_m"] = nearest_dist_m

    meta = {
        "target_radius_m": float(radius_m),
        "positive_share_overall": float(out[TARGET_EXISTS].mean()),
        "positive_share_train_group": float(out.loc[out["업종그룹"] == TRAIN_GROUP, TARGET_EXISTS].mean()),
        "positive_share_test_group": float(out.loc[out["업종그룹"] == TEST_GROUP, TARGET_EXISTS].mean()),
        "median_positive_damage_천원": float(out.loc[out[TARGET_DAMAGE_SUM] > 0, TARGET_DAMAGE_SUM].median()),
        "p90_positive_damage_천원": float(out.loc[out[TARGET_DAMAGE_SUM] > 0, TARGET_DAMAGE_SUM].quantile(0.9)),
    }
    return out.reset_index(drop=True), meta


def split_domain(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["업종그룹"] == TRAIN_GROUP].copy()
    test_df = df[df["업종그룹"] == TEST_GROUP].copy()
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def build_stage1_models() -> dict[str, object]:
    prep = make_preprocessor()
    return {
        "Dummy": Pipeline([("prep", prep), ("model", DummyClassifier(strategy="most_frequent"))]),
        "LogisticRegression": Pipeline(
            [
                ("prep", prep),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "LightGBM": Pipeline(
            [
                ("prep", prep),
                (
                    "model",
                    LGBMClassifier(
                        objective="binary",
                        n_estimators=700,
                        learning_rate=0.04,
                        num_leaves=31,
                        min_child_samples=20,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=1.2,
                        reg_alpha=0.05,
                        random_state=42,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("prep", prep),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        n_estimators=900,
                        max_depth=4,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=1.2,
                        reg_alpha=0.05,
                        min_child_weight=2,
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def build_stage2_models() -> dict[str, object]:
    prep = make_preprocessor()
    return {
        "Dummy": Pipeline([("prep", prep), ("model", DummyRegressor(strategy="mean"))]),
        "Ridge": Pipeline([("prep", prep), ("model", Ridge(alpha=10.0, random_state=42))]),
        "LightGBM": Pipeline(
            [
                ("prep", prep),
                (
                    "model",
                    LGBMRegressor(
                        objective="regression",
                        n_estimators=900,
                        learning_rate=0.04,
                        num_leaves=31,
                        min_child_samples=20,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=1.2,
                        reg_alpha=0.05,
                        random_state=42,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("prep", prep),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=1200,
                        max_depth=4,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=1.2,
                        reg_alpha=0.05,
                        min_child_weight=2,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def prepare_catboost_frames(df: pd.DataFrame) -> pd.DataFrame:
    out = df[FEATURE_COLUMNS].copy()
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].fillna("미상").astype(str)
    return out


def stage1_catboost(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = prepare_catboost_frames(train_df)
    X_test = prepare_catboost_frames(test_df)
    y_train = train_df[TARGET_EXISTS].to_numpy()
    y_test = test_df[TARGET_EXISTS].to_numpy()
    cat_idx = [X_train.columns.get_loc(col) for col in CATEGORICAL_FEATURES]

    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=1600,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=6.0,
        random_seed=42,
        verbose=False,
    )
    model.fit(Pool(X_train, y_train, cat_features=cat_idx), verbose=False)
    prob = model.predict_proba(Pool(X_test, cat_features=cat_idx))[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = pd.DataFrame(
        [
            {
                "model": "CatBoost",
                "accuracy": accuracy_score(y_test, pred),
                "f1": f1_score(y_test, pred, zero_division=0),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, prob),
                "avg_precision": average_precision_score(y_test, prob),
            }
        ]
    )
    pred_df = pd.DataFrame({"model": "CatBoost", "y_true": y_test, "y_pred": pred, "y_prob": prob})
    return metrics, pred_df


def stage2_catboost(
    train_pos: pd.DataFrame,
    test_pos: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = prepare_catboost_frames(train_pos)
    X_test = prepare_catboost_frames(test_pos)
    y_train = train_pos[TARGET_DAMAGE_LOG].to_numpy(dtype=float)
    y_test_log = test_pos[TARGET_DAMAGE_LOG].to_numpy(dtype=float)
    y_test_raw = test_pos[TARGET_DAMAGE_SUM].to_numpy(dtype=float)
    cat_idx = [X_train.columns.get_loc(col) for col in CATEGORICAL_FEATURES]

    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=1800,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=6.0,
        random_seed=42,
        verbose=False,
    )
    model.fit(Pool(X_train, y_train, cat_features=cat_idx), verbose=False)
    pred_log = model.predict(Pool(X_test, cat_features=cat_idx))
    pred_raw = np.expm1(pred_log).clip(min=0)

    metrics = pd.DataFrame(
        [
            {
                "model": "CatBoost",
                "r2_log": r2_score(y_test_log, pred_log),
                "rmse_log": math.sqrt(mean_squared_error(y_test_log, pred_log)),
                "mae_log": mean_absolute_error(y_test_log, pred_log),
                "rmse_백만원": math.sqrt(mean_squared_error(y_test_raw, pred_raw)) / 1000.0,
                "mae_백만원": mean_absolute_error(y_test_raw, pred_raw) / 1000.0,
            }
        ]
    )
    pred_df = pd.DataFrame(
        {
            "model": "CatBoost",
            "y_true_log": y_test_log,
            "y_pred_log": pred_log,
            "y_true_천원": y_test_raw,
            "y_pred_천원": pred_raw,
        }
    )
    return metrics, pred_df


def run_stage1(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    X_train = train_df[FEATURE_COLUMNS].copy()
    X_test = test_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_EXISTS].to_numpy()
    y_test = test_df[TARGET_EXISTS].to_numpy()

    metrics_rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []
    fitted: dict[str, object] = {}

    for name, model in build_stage1_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            prob = pred.astype(float)

        metrics_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "f1": f1_score(y_test, pred, zero_division=0),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else np.nan,
                "avg_precision": average_precision_score(y_test, prob),
            }
        )
        pred_frames.append(pd.DataFrame({"model": name, "y_true": y_test, "y_pred": pred, "y_prob": prob}))
        fitted[name] = model

    cat_metrics, cat_pred = stage1_catboost(train_df, test_df)
    metrics = pd.concat([pd.DataFrame(metrics_rows), cat_metrics], ignore_index=True)
    preds = pd.concat(pred_frames + [cat_pred], ignore_index=True)

    best_name = metrics.sort_values(["roc_auc", "f1", "accuracy"], ascending=[False, False, False]).iloc[0]["model"]
    return metrics.sort_values("roc_auc", ascending=False).reset_index(drop=True), preds, {"best_model_name": best_name, "fitted": fitted}


def run_stage2(
    train_pos: pd.DataFrame,
    test_pos: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    X_train = train_pos[FEATURE_COLUMNS].copy()
    X_test = test_pos[FEATURE_COLUMNS].copy()
    y_train = train_pos[TARGET_DAMAGE_LOG].to_numpy(dtype=float)
    y_test_log = test_pos[TARGET_DAMAGE_LOG].to_numpy(dtype=float)
    y_test_raw = test_pos[TARGET_DAMAGE_SUM].to_numpy(dtype=float)

    metrics_rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []
    fitted: dict[str, object] = {}

    for name, model in build_stage2_models().items():
        model.fit(X_train, y_train)
        pred_log = model.predict(X_test)
        pred_raw = np.expm1(pred_log).clip(min=0)

        metrics_rows.append(
            {
                "model": name,
                "r2_log": r2_score(y_test_log, pred_log),
                "rmse_log": math.sqrt(mean_squared_error(y_test_log, pred_log)),
                "mae_log": mean_absolute_error(y_test_log, pred_log),
                "rmse_백만원": math.sqrt(mean_squared_error(y_test_raw, pred_raw)) / 1000.0,
                "mae_백만원": mean_absolute_error(y_test_raw, pred_raw) / 1000.0,
            }
        )
        pred_frames.append(
            pd.DataFrame(
                {
                    "model": name,
                    "y_true_log": y_test_log,
                    "y_pred_log": pred_log,
                    "y_true_천원": y_test_raw,
                    "y_pred_천원": pred_raw,
                }
            )
        )
        fitted[name] = model

    cat_metrics, cat_pred = stage2_catboost(train_pos, test_pos)
    metrics = pd.concat([pd.DataFrame(metrics_rows), cat_metrics], ignore_index=True)
    preds = pd.concat(pred_frames + [cat_pred], ignore_index=True)

    best_name = metrics.sort_values(["r2_log", "mae_백만원"], ascending=[False, True]).iloc[0]["model"]
    return metrics.sort_values("r2_log", ascending=False).reset_index(drop=True), preds, {"best_model_name": best_name, "fitted": fitted}


def get_stage1_best_prob(best_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, fitted: dict[str, object]) -> np.ndarray:
    if best_name == "CatBoost":
        X_train = prepare_catboost_frames(train_df)
        X_test = prepare_catboost_frames(test_df)
        y_train = train_df[TARGET_EXISTS].to_numpy()
        cat_idx = [X_train.columns.get_loc(col) for col in CATEGORICAL_FEATURES]
        model = CatBoostClassifier(
            loss_function="Logloss",
            iterations=1600,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=6.0,
            random_seed=42,
            verbose=False,
        )
        model.fit(Pool(X_train, y_train, cat_features=cat_idx), verbose=False)
        return model.predict_proba(Pool(X_test, cat_features=cat_idx))[:, 1]

    model = fitted[best_name]
    return model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]


def get_stage2_best_log_prediction(best_name: str, train_pos: pd.DataFrame, test_df: pd.DataFrame, fitted: dict[str, object]) -> np.ndarray:
    if best_name == "CatBoost":
        X_train = prepare_catboost_frames(train_pos)
        X_test = prepare_catboost_frames(test_df)
        y_train = train_pos[TARGET_DAMAGE_LOG].to_numpy(dtype=float)
        cat_idx = [X_train.columns.get_loc(col) for col in CATEGORICAL_FEATURES]
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=1800,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=6.0,
            random_seed=42,
            verbose=False,
        )
        model.fit(Pool(X_train, y_train, cat_features=cat_idx), verbose=False)
        return model.predict(Pool(X_test, cat_features=cat_idx))

    model = fitted[best_name]
    return model.predict(test_df[FEATURE_COLUMNS])


def plot_radius_design(radius_df: pd.DataFrame, ax: plt.Axes) -> None:
    palette = {"전체": "#334155", TRAIN_GROUP: "#2563eb", TEST_GROUP: "#e11d48"}
    for group_name in ["전체", TRAIN_GROUP, TEST_GROUP]:
        sub = radius_df[radius_df["group"] == group_name].sort_values("radius_m")
        ax.plot(
            sub["radius_m"],
            sub["positive_share"],
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=group_name,
            color=palette[group_name],
        )
    ax.axvline(TARGET_RADIUS, linestyle="--", color="#64748b", linewidth=1.3)
    ax.text(TARGET_RADIUS + 4, 0.08, f"target {TARGET_RADIUS}m", color="#475569", fontsize=10)
    ax.set_title("타깃 설계: 반경별 양성 비율", loc="left", fontsize=14, fontweight="bold")
    ax.set_xlabel("반경(m)")
    ax.set_ylabel("양성 비율")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, fontsize=9)


def plot_stage1_metrics(stage1_metrics: pd.DataFrame, ax: plt.Axes) -> None:
    ordered = stage1_metrics.sort_values("roc_auc", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    colors = ["#cbd5e1", "#94a3b8", "#81b29a", "#e07a5f", "#264653"]
    ax.barh(y, ordered["roc_auc"], color=colors[: len(ordered)], alpha=0.92)
    ax.scatter(ordered["f1"], y, s=72, color="white", edgecolor="#1f2937", linewidth=1.0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["model"])
    ax.set_xlim(0.45, max(0.85, ordered["roc_auc"].max() + 0.04))
    ax.set_title("1단계: 50m 내 재산피해 화재 존재", loc="left", fontsize=14, fontweight="bold")
    ax.set_xlabel("ROC-AUC")
    ax.grid(axis="x", alpha=0.16)
    for idx, row in ordered.iterrows():
        ax.text(row["roc_auc"] + 0.006, idx, f"{row['roc_auc']:.3f}", va="center", fontsize=10, color="#334155")


def plot_stage2_metrics(stage2_metrics: pd.DataFrame, ax: plt.Axes) -> None:
    ordered = stage2_metrics.sort_values("mae_백만원", ascending=False).reset_index(drop=True)
    y = np.arange(len(ordered))
    colors = ["#cbd5e1", "#94a3b8", "#81b29a", "#e07a5f", "#264653"]
    ax.barh(y, ordered["mae_백만원"], color=colors[: len(ordered)], alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["model"])
    ax.set_title("2단계: 양성 숙소의 누적 재산피해액", loc="left", fontsize=14, fontweight="bold")
    ax.set_xlabel("MAE on original scale (million KRW)")
    ax.grid(axis="x", alpha=0.16)
    ax.set_xlim(0, ordered["mae_백만원"].max() * 1.18)
    for idx, row in ordered.iterrows():
        ax.text(
            row["mae_백만원"] + 0.08,
            idx,
            f"MAE {row['mae_백만원']:.2f} / R²(log) {row['r2_log']:.3f}",
            va="center",
            fontsize=9.8,
            color="#334155",
        )


def plot_combined_expected_damage(test_with_pred: pd.DataFrame, ax: plt.Axes) -> None:
    actual = np.log1p(test_with_pred["actual_damage_천원"].to_numpy())
    pred = np.log1p(test_with_pred["expected_damage_천원"].to_numpy())
    colors = np.where(test_with_pred["actual_positive"].to_numpy() == 1, "#e11d48", "#94a3b8")

    ax.scatter(actual, pred, s=18, alpha=0.65, c=colors, linewidths=0)
    min_v = float(min(actual.min(), pred.min()))
    max_v = float(max(actual.max(), pred.max()))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="#334155", linewidth=1.2)
    ax.set_title("결합 결과: 기대재산피해액", loc="left", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual log1p(damage in KRW-thousand)")
    ax.set_ylabel("Predicted expected log1p(damage)")
    ax.grid(alpha=0.16)

    r2_combined = r2_score(actual, pred)
    mae_combined = mean_absolute_error(
        test_with_pred["actual_damage_천원"].to_numpy(),
        test_with_pred["expected_damage_천원"].to_numpy(),
    ) / 1000.0
    ax.text(
        0.03,
        0.97,
        f"Combined R²(log) = {r2_combined:.3f}\nCombined MAE = {mae_combined:,.1f} million KRW",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        color="#1f2937",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#e2e8f0"),
    )


def plot_top20_expected_damage(test_with_pred: pd.DataFrame, out_path: Path) -> None:
    top = test_with_pred.sort_values("expected_damage_천원", ascending=False).head(20).copy()
    top["expected_damage_백만원"] = top["expected_damage_천원"] / 1000.0
    top["label"] = top["동"].fillna("") + " · " + top["숙소명"].astype(str).str.slice(0, 22)

    fig, ax = plt.subplots(figsize=(12.0, 7.6))
    y = np.arange(len(top))[::-1]
    ax.barh(y, top["expected_damage_백만원"][::-1], color="#c2410c", alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(top["label"][::-1])
    ax.set_xlabel("Expected damage (million KRW)")
    ax.set_title("외국인관광도시민박업 예상 재산피해 상위 20개", loc="left", fontsize=16, fontweight="bold")
    ax.grid(axis="x", alpha=0.16)
    max_val = float(top["expected_damage_백만원"].max()) if len(top) else 1.0
    ax.set_xlim(0, max_val * 1.18)
    for yy, value in zip(y, top["expected_damage_백만원"][::-1]):
        ax.text(value + max_val * 0.02, yy, f"{value:,.2f}", va="center", fontsize=9.2, color="#7c2d12")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_dashboard(
    radius_df: pd.DataFrame,
    stage1_metrics: pd.DataFrame,
    stage2_metrics: pd.DataFrame,
    test_with_pred: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.2))
    plot_radius_design(radius_df, axes[0, 0])
    plot_stage1_metrics(stage1_metrics, axes[0, 1])
    plot_stage2_metrics(stage2_metrics, axes[1, 0])
    plot_combined_expected_damage(test_with_pred, axes[1, 1])

    fig.suptitle(
        "재산피해액 기반 2단계 모델: 50m 반경 화재 존재 여부 + 누적 피해규모",
        fontsize=18,
        fontweight="bold",
        x=0.05,
        y=0.98,
        ha="left",
    )
    fig.text(
        0.05,
        0.945,
        "학습은 기존숙박군(숙박업+관광숙박업), 평가는 외국인관광도시민박업에서 수행했습니다. 50m는 양성비율 포화가 덜한 반경이라 2단계 구조에 사용했습니다.",
        fontsize=10.5,
        color="#475569",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    ensure_dirs()

    lodging = prepare_lodging_data(LODGING_CSV)
    fire = prepare_fire_data(FIRE_CSV)

    radius_df = radius_diagnostics(lodging, fire)
    full_df, target_meta = attach_targets(lodging, fire, TARGET_RADIUS)
    train_df, test_df = split_domain(full_df)

    train_pos = train_df[train_df[TARGET_EXISTS] == 1].copy().reset_index(drop=True)
    test_pos = test_df[test_df[TARGET_EXISTS] == 1].copy().reset_index(drop=True)

    stage1_metrics, stage1_preds, stage1_info = run_stage1(train_df, test_df)
    stage2_metrics, stage2_preds, stage2_info = run_stage2(train_pos, test_pos)

    best_stage1_name = stage1_info["best_model_name"]
    best_stage2_name = stage2_info["best_model_name"]

    best_prob = get_stage1_best_prob(best_stage1_name, train_df, test_df, stage1_info["fitted"])
    best_log = get_stage2_best_log_prediction(best_stage2_name, train_pos, test_df, stage2_info["fitted"])
    best_positive_damage = np.expm1(best_log).clip(min=0)
    expected_damage = best_prob * best_positive_damage

    test_with_pred = test_df[
        ["구", "동", "숙소명", "업종", "업종그룹", TARGET_EXISTS, TARGET_DAMAGE_SUM]
    ].copy()
    test_with_pred = test_with_pred.rename(
        columns={
            TARGET_EXISTS: "actual_positive",
            TARGET_DAMAGE_SUM: "actual_damage_천원",
        }
    )
    test_with_pred["best_stage1_model"] = best_stage1_name
    test_with_pred["best_stage2_model"] = best_stage2_name
    test_with_pred["pred_fire_probability"] = best_prob
    test_with_pred["pred_positive_damage_천원"] = best_positive_damage
    test_with_pred["expected_damage_천원"] = expected_damage

    combined_metrics = pd.DataFrame(
        [
            {
                "best_stage1_model": best_stage1_name,
                "best_stage2_model": best_stage2_name,
                "combined_r2_log": r2_score(
                    np.log1p(test_with_pred["actual_damage_천원"]),
                    np.log1p(test_with_pred["expected_damage_천원"]),
                ),
                "combined_rmse_log": math.sqrt(
                    mean_squared_error(
                        np.log1p(test_with_pred["actual_damage_천원"]),
                        np.log1p(test_with_pred["expected_damage_천원"]),
                    )
                ),
                "combined_mae_백만원": mean_absolute_error(
                    test_with_pred["actual_damage_천원"],
                    test_with_pred["expected_damage_천원"],
                )
                / 1000.0,
            }
        ]
    )

    table_target = TABLE_DIR / "lodging_targets_property_damage_50m.csv"
    table_radius = TABLE_DIR / "property_damage_radius_diagnostics.csv"
    table_stage1 = TABLE_DIR / "property_damage_stage1_metrics.csv"
    table_stage2 = TABLE_DIR / "property_damage_stage2_metrics.csv"
    table_hurdle = TABLE_DIR / "property_damage_hurdle_predictions.csv"
    table_combined = TABLE_DIR / "property_damage_hurdle_metrics.csv"
    table_stage1_pred = TABLE_DIR / "property_damage_stage1_test_predictions.csv"
    table_stage2_pred = TABLE_DIR / "property_damage_stage2_test_predictions.csv"
    meta_path = TABLE_DIR / "property_damage_two_stage_metadata.json"

    fig_dashboard = FIG_DIR / "property_damage_2stage_dashboard.png"
    fig_top20 = FIG_DIR / "property_damage_expected_top20.png"

    full_df.to_csv(table_target, index=False, encoding="utf-8-sig")
    radius_df.to_csv(table_radius, index=False, encoding="utf-8-sig")
    stage1_metrics.to_csv(table_stage1, index=False, encoding="utf-8-sig")
    stage2_metrics.to_csv(table_stage2, index=False, encoding="utf-8-sig")
    test_with_pred.to_csv(table_hurdle, index=False, encoding="utf-8-sig")
    combined_metrics.to_csv(table_combined, index=False, encoding="utf-8-sig")
    stage1_preds.to_csv(table_stage1_pred, index=False, encoding="utf-8-sig")
    stage2_preds.to_csv(table_stage2_pred, index=False, encoding="utf-8-sig")

    metadata = {
        "analysis_years": [ANALYSIS_YEAR_START, ANALYSIS_YEAR_END],
        "target_design": {
            "stage1": f"{TARGET_RADIUS}m 내 재산피해 화재 존재 여부",
            "stage2": f"{TARGET_RADIUS}m 내 누적 재산피해액(log1p, 양성 숙소만)",
            "combined": "발생확률 × 양성조건 피해규모",
        },
        "target_meta": target_meta,
        "rows_total": int(len(full_df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "rows_train_positive": int(len(train_pos)),
        "rows_test_positive": int(len(test_pos)),
        "best_stage1_model": best_stage1_name,
        "best_stage2_model": best_stage2_name,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    build_dashboard(radius_df, stage1_metrics, stage2_metrics, test_with_pred, fig_dashboard)
    plot_top20_expected_damage(test_with_pred, fig_top20)

    print(stage1_metrics.to_string(index=False))
    print()
    print(stage2_metrics.to_string(index=False))
    print()
    print(combined_metrics.to_string(index=False))
    print(f"\nSaved target table: {table_target}")
    print(f"Saved dashboard: {fig_dashboard}")
    print(f"Saved top20 figure: {fig_top20}")


if __name__ == "__main__":
    main()
