# -*- coding: utf-8 -*-
"""End-to-end injury-fire target generation and external generalization study.

Train domain: 숙박업 + 관광숙박업
Test domain: 외국인관광도시민박업
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SCRIPT_DIR = Path(__file__).resolve().parent
APRIL_DIR = SCRIPT_DIR.parent
PROJECT_DIR = APRIL_DIR.parent

LODGING_CSV = APRIL_DIR / "data" / "분석변수_최종테이블0423.csv"
FIRE_CSV = PROJECT_DIR / "data" / "화재출동" / "화재출동_2021_2024.csv"

TABLE_DIR = SCRIPT_DIR / "tables"
FIG_DIR = SCRIPT_DIR / "figures"
MODEL_DIR = SCRIPT_DIR / "models"

TARGET_RADIUS = 500
TOP3_FIRE_TYPES = ["주거", "판매/업무시설", "생활서비스"]
DOMAIN_TRAIN = ["숙박업", "관광숙박업"]
DOMAIN_TEST = "외국인관광도시민박업"
ANALYSIS_YEAR_START = 2021
ANALYSIS_YEAR_END = 2024

REG_TARGETS = [
    f"log1p_r{TARGET_RADIUS}m_injury_fire_count_all",
    f"log1p_r{TARGET_RADIUS}m_injury_fire_count_주거",
    f"log1p_r{TARGET_RADIUS}m_injury_fire_count_판매업무시설",
    f"log1p_r{TARGET_RADIUS}m_injury_fire_count_생활서비스",
]
CLS_TARGET = f"r{TARGET_RADIUS}m_injury_fire_dominant_type"


def setup_style() -> None:
    font_names = {f.name for f in fm.fontManager.ttflist}
    selected_font = "Malgun Gothic"
    for font in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if font in font_names:
            selected_font = font
            break

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.family"] = selected_font
    plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "#fcfcfb"
    plt.rcParams["axes.facecolor"] = "#fcfcfb"
    plt.rcParams["savefig.facecolor"] = "#fcfcfb"


def ensure_dirs() -> None:
    for path in (TABLE_DIR, FIG_DIR, MODEL_DIR):
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


def haversine_matrix(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius = 6_371_000
    lat1_rad = np.radians(lat1)[:, None]
    lon1_rad = np.radians(lon1)[:, None]
    lat2_rad = np.radians(lat2)[None, :]
    lon2_rad = np.radians(lon2)[None, :]
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return radius * 2 * np.arcsin(np.sqrt(a))


def prepare_lodging_data(path: Path) -> pd.DataFrame:
    df = clean_text_columns(read_csv_robust(path))

    numeric_cols = ["승인연도", "주변건물수", "집중도", "단속위험도", "구조노후도", "도로폭위험도", "위도", "경도"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["위도", "경도", "업종"]).copy()
    df["건물나이_2026"] = 2026 - df["승인연도"]
    df["건물나이_2026"] = df["건물나이_2026"].where(df["승인연도"].notna())
    return df.reset_index(drop=True)


def prepare_fire_data(path: Path) -> pd.DataFrame:
    df = clean_text_columns(read_csv_robust(path))
    for col in ("인명피해계", "발생연도", "위도", "경도"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        df["발생연도"].between(ANALYSIS_YEAR_START, ANALYSIS_YEAR_END, inclusive="both")
        & (df["인명피해계"] >= 1)
        & df["발화장소_대분류"].isin(TOP3_FIRE_TYPES)
        & df["위도"].notna()
        & df["경도"].notna()
    ].copy()

    df["발화장소_대분류_norm"] = df["발화장소_대분류"].replace({"판매/업무시설": "판매업무시설"})
    return df.reset_index(drop=True)


def attach_targets(lodging: pd.DataFrame, fire: pd.DataFrame, radii: Iterable[int] = (300, 500, 1000)) -> tuple[pd.DataFrame, dict[str, int]]:
    out = lodging.copy()

    dist = haversine_matrix(
        out["위도"].to_numpy(),
        out["경도"].to_numpy(),
        fire["위도"].to_numpy(),
        fire["경도"].to_numpy(),
    )

    tie_count = 0
    for radius in radii:
        mask_all = dist <= radius
        out[f"r{radius}m_injury_fire_count_all"] = mask_all.sum(axis=1)
        out[f"log1p_r{radius}m_injury_fire_count_all"] = np.log1p(out[f"r{radius}m_injury_fire_count_all"])

        per_type_cols: list[str] = []
        for fire_type in ("주거", "판매업무시설", "생활서비스"):
            type_mask = fire["발화장소_대분류_norm"].to_numpy() == fire_type
            col = f"r{radius}m_injury_fire_count_{fire_type}"
            out[col] = (mask_all[:, type_mask]).sum(axis=1)
            out[f"log1p_{col}"] = np.log1p(out[col])
            per_type_cols.append(col)

        if radius == TARGET_RADIUS:
            dominant_labels: list[str] = []
            per_type_values = out[per_type_cols].to_numpy()
            label_names = np.array(["주거", "판매업무시설", "생활서비스"])
            for counts in per_type_values:
                if counts.sum() == 0:
                    dominant_labels.append("없음")
                    continue
                max_count = counts.max()
                winner_idx = np.where(counts == max_count)[0]
                if len(winner_idx) > 1:
                    tie_count += 1
                dominant_labels.append(label_names[winner_idx[0]])
            out[CLS_TARGET] = dominant_labels

    out["r500m_injury_fire_count_top3_sum"] = (
        out["r500m_injury_fire_count_주거"]
        + out["r500m_injury_fire_count_판매업무시설"]
        + out["r500m_injury_fire_count_생활서비스"]
    )

    meta = {
        "lodging_rows": int(len(out)),
        "fire_rows_filtered": int(len(fire)),
        "tie_count_for_classification": int(tie_count),
    }
    return out, meta


def make_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "승인연도",
        "건물나이_2026",
        "주변건물수",
        "집중도",
        "단속위험도",
        "구조노후도",
        "도로폭위험도",
        "위도",
        "경도",
    ]
    categorical_features = ["구"]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def split_domain(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["업종"].isin(DOMAIN_TRAIN)].copy()
    test_df = df[df["업종"] == DOMAIN_TEST].copy()
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names: list[str] = []
    for _, transformer, columns in preprocessor.transformers_:
        if transformer == "drop":
            continue
        if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
            onehot = transformer.named_steps["onehot"]
            names.extend(onehot.get_feature_names_out(columns).tolist())
        else:
            names.extend(list(columns))
    return names


def train_regressions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = train_df[["구", "승인연도", "건물나이_2026", "주변건물수", "집중도", "단속위험도", "구조노후도", "도로폭위험도", "위도", "경도"]]
    X_test = test_df[X_train.columns]

    metric_rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []

    for target in REG_TARGETS:
        y_train = train_df[target].to_numpy()
        y_test = test_df[target].to_numpy()

        baseline = Pipeline(
            [
                ("prep", make_preprocessor()),
                ("model", DummyRegressor(strategy="mean")),
            ]
        )
        rf = Pipeline(
            [
                ("prep", make_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        max_depth=10,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )

        for name, model in (("baseline_mean", baseline), ("random_forest", rf)):
            model.fit(X_train, y_train)
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)

            metric_rows.append(
                {
                    "task": "regression",
                    "target": target,
                    "model": name,
                    "train_mae": mean_absolute_error(y_train, pred_train),
                    "train_rmse": rmse(y_train, pred_train),
                    "train_r2": r2_score(y_train, pred_train),
                    "test_mae": mean_absolute_error(y_test, pred_test),
                    "test_rmse": rmse(y_test, pred_test),
                    "test_r2": r2_score(y_test, pred_test),
                }
            )

            pred_frame = pd.DataFrame(
                {
                    "숙소명": test_df["숙소명"],
                    "구": test_df["구"],
                    "업종": test_df["업종"],
                    "target": target,
                    "model": name,
                    "actual": y_test,
                    "predicted": pred_test,
                }
            )
            pred_frames.append(pred_frame)

            if name == "random_forest":
                joblib.dump(model, MODEL_DIR / f"{target}_random_forest.joblib")
                prep = model.named_steps["prep"]
                feature_names = get_feature_names(prep)
                importances = model.named_steps["model"].feature_importances_
                fi_df = (
                    pd.DataFrame({"feature": feature_names, "importance": importances})
                    .sort_values("importance", ascending=False)
                    .head(20)
                    .assign(target=target, model=name)
                )
                fi_path = TABLE_DIR / f"{target}_feature_importance.csv"
                fi_df.to_csv(fi_path, index=False, encoding="utf-8-sig")

    return pd.DataFrame(metric_rows), pd.concat(pred_frames, ignore_index=True)


def train_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = train_df[["구", "승인연도", "건물나이_2026", "주변건물수", "집중도", "단속위험도", "구조노후도", "도로폭위험도", "위도", "경도"]]
    X_test = test_df[X_train.columns]
    y_train = train_df[CLS_TARGET].to_numpy()
    y_test = test_df[CLS_TARGET].to_numpy()

    models = {
        "baseline_most_frequent": DummyClassifier(strategy="most_frequent"),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        ),
    }

    metric_rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []
    report_rows: list[dict[str, object]] = []

    labels = ["없음", "주거", "판매업무시설", "생활서비스"]

    for name, base_model in models.items():
        model = Pipeline([("prep", make_preprocessor()), ("model", base_model)])
        model.fit(X_train, y_train)
        pred_test = model.predict(X_test)
        pred_train = model.predict(X_train)

        metric_rows.append(
            {
                "task": "classification",
                "target": CLS_TARGET,
                "model": name,
                "train_accuracy": accuracy_score(y_train, pred_train),
                "test_accuracy": accuracy_score(y_test, pred_test),
                "test_macro_f1": f1_score(y_test, pred_test, average="macro", zero_division=0),
                "test_weighted_f1": f1_score(y_test, pred_test, average="weighted", zero_division=0),
                "test_macro_precision": precision_score(y_test, pred_test, average="macro", zero_division=0),
                "test_macro_recall": recall_score(y_test, pred_test, average="macro", zero_division=0),
            }
        )

        pred_frames.append(
            pd.DataFrame(
                {
                    "숙소명": test_df["숙소명"],
                    "구": test_df["구"],
                    "업종": test_df["업종"],
                    "model": name,
                    "actual": y_test,
                    "predicted": pred_test,
                }
            )
        )

        report = classification_report(y_test, pred_test, output_dict=True, zero_division=0)
        for label, stats in report.items():
            if isinstance(stats, dict):
                report_rows.append(
                    {
                        "model": name,
                        "label": label,
                        "precision": stats["precision"],
                        "recall": stats["recall"],
                        "f1_score": stats["f1-score"],
                        "support": stats["support"],
                    }
                )

        cm = confusion_matrix(y_test, pred_test, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"actual_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
        cm_df.to_csv(TABLE_DIR / f"{name}_classification_confusion_matrix.csv", encoding="utf-8-sig")

        if name == "random_forest":
            joblib.dump(model, MODEL_DIR / f"{CLS_TARGET}_random_forest.joblib")
            prep = model.named_steps["prep"]
            feature_names = get_feature_names(prep)
            importances = model.named_steps["model"].feature_importances_
            fi_df = (
                pd.DataFrame({"feature": feature_names, "importance": importances})
                .sort_values("importance", ascending=False)
                .head(20)
                .assign(target=CLS_TARGET, model=name)
            )
            fi_df.to_csv(TABLE_DIR / f"{CLS_TARGET}_feature_importance.csv", index=False, encoding="utf-8-sig")

    return pd.DataFrame(metric_rows), pd.concat(pred_frames, ignore_index=True), pd.DataFrame(report_rows)


def plot_target_distribution(lodging_with_targets: pd.DataFrame, fire_filtered: pd.DataFrame) -> Path:
    split_counts = lodging_with_targets["업종"].value_counts().reindex(["관광숙박업", "숙박업", "외국인관광도시민박업"])
    dominant_counts = (
        lodging_with_targets.groupby("업종")[CLS_TARGET]
        .value_counts()
        .rename("count")
        .reset_index()
    )
    fire_counts = fire_filtered["발화장소_대분류_norm"].value_counts().reindex(["주거", "판매업무시설", "생활서비스"])

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6))

    sns.barplot(
        x=split_counts.index,
        y=split_counts.values,
        ax=axes[0],
        palette=["#2563eb", "#0f766e", "#dc2626"],
    )
    axes[0].set_title("숙소 데이터 분할", loc="left", fontsize=16, fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("숙소 수")
    axes[0].tick_params(axis="x", rotation=14)
    for idx, value in enumerate(split_counts.values):
        axes[0].text(idx, value + max(split_counts.values) * 0.015, f"{int(value):,}", ha="center", va="bottom", fontsize=10)

    sns.barplot(
        x=fire_counts.index,
        y=fire_counts.values,
        ax=axes[1],
        palette=["#1d4ed8", "#7c3aed", "#ea580c"],
    )
    axes[1].set_title("2021-2024 인명피해 화재 top3", loc="left", fontsize=16, fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("화재 건수")
    axes[1].tick_params(axis="x", rotation=12)
    for idx, value in enumerate(fire_counts.values):
        axes[1].text(idx, value + max(fire_counts.values) * 0.02, f"{int(value):,}", ha="center", va="bottom", fontsize=10)

    dom_pivot = dominant_counts.pivot(index="업종", columns=CLS_TARGET, values="count").fillna(0)
    dom_pivot = dom_pivot.reindex(index=["관광숙박업", "숙박업", "외국인관광도시민박업"], columns=["없음", "주거", "판매업무시설", "생활서비스"], fill_value=0)
    bottom = np.zeros(len(dom_pivot))
    colors = {"없음": "#cbd5e1", "주거": "#2563eb", "판매업무시설": "#8b5cf6", "생활서비스": "#f97316"}
    for col in dom_pivot.columns:
        axes[2].bar(dom_pivot.index, dom_pivot[col].values, bottom=bottom, label=col, color=colors[col], alpha=0.95)
        bottom += dom_pivot[col].values
    axes[2].set_title("500m 우세 발화장소 라벨 분포", loc="left", fontsize=16, fontweight="bold")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("숙소 수")
    axes[2].tick_params(axis="x", rotation=14)
    axes[2].legend(frameon=False, fontsize=9, ncol=2, loc="upper right")

    fig.suptitle("외부 일반화 분석용 데이터 구성", x=0.055, y=1.02, ha="left", fontsize=22, fontweight="bold")
    fig.text(0.055, 0.96, "회귀 타깃은 500m 인명피해 화재 노출량(log1p), 분류 타깃은 500m 우세 발화장소 유형입니다.", fontsize=10.8, color="#475569")
    fig.tight_layout()

    out = FIG_DIR / "01_target_distribution.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_regression_results(metric_df: pd.DataFrame, pred_df: pd.DataFrame) -> Path:
    rf_metrics = metric_df[metric_df["model"] == "random_forest"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    title_map = {
        REG_TARGETS[0]: "전체 인명피해 화재 노출",
        REG_TARGETS[1]: "주거 인명피해 화재 노출",
        REG_TARGETS[2]: "판매·업무시설 인명피해 화재 노출",
        REG_TARGETS[3]: "생활서비스 인명피해 화재 노출",
    }

    for ax, target in zip(axes, REG_TARGETS):
        sub_pred = pred_df[(pred_df["target"] == target) & (pred_df["model"] == "random_forest")]
        sub_metric = rf_metrics[rf_metrics["target"] == target].iloc[0]
        ax.scatter(sub_pred["actual"], sub_pred["predicted"], s=20, alpha=0.55, color="#2563eb", edgecolor="none")
        min_val = min(sub_pred["actual"].min(), sub_pred["predicted"].min())
        max_val = max(sub_pred["actual"].max(), sub_pred["predicted"].max())
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#dc2626", linewidth=1.4)
        ax.set_title(title_map[target], loc="left", fontsize=14, fontweight="bold")
        ax.set_xlabel("실제값")
        ax.set_ylabel("예측값")
        ax.text(
            0.03,
            0.97,
            f"Test MAE  {sub_metric['test_mae']:.3f}\nTest RMSE {sub_metric['test_rmse']:.3f}\nTest R²   {sub_metric['test_r2']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#cbd5e1"),
        )

    fig.suptitle("외국인관광도시민박업 외부 테스트: 회귀 결과", x=0.06, y=0.995, ha="left", fontsize=22, fontweight="bold")
    fig.text(0.06, 0.965, "학습: 숙박업 + 관광숙박업 / 테스트: 외국인관광도시민박업 / 모델: Random Forest", fontsize=10.8, color="#475569")
    fig.tight_layout()

    out = FIG_DIR / "02_regression_results.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_classification_results(metric_df: pd.DataFrame, pred_df: pd.DataFrame, report_df: pd.DataFrame) -> Path:
    labels = ["없음", "주거", "판매업무시설", "생활서비스"]
    rf_pred = pred_df[pred_df["model"] == "random_forest"]
    cm = confusion_matrix(rf_pred["actual"], rf_pred["predicted"], labels=labels)
    cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    rf_metrics = metric_df[metric_df["model"] == "random_forest"].iloc[0]
    report_plot = report_df[(report_df["model"] == "random_forest") & (~report_df["label"].isin(["accuracy", "macro avg", "weighted avg"]))].copy()

    fig = plt.figure(figsize=(15.5, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.22)
    ax_cm = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    sns.heatmap(
        cm_pct,
        annot=cm,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "행 기준 비율"},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax_cm,
        linewidths=0.4,
        linecolor="#f8fafc",
    )
    ax_cm.set_title("분류 혼동행렬", loc="left", fontsize=16, fontweight="bold")
    ax_cm.set_xlabel("예측 라벨")
    ax_cm.set_ylabel("실제 라벨")

    report_long = report_plot.melt(id_vars="label", value_vars=["precision", "recall", "f1_score"], var_name="metric", value_name="score")
    sns.barplot(data=report_long, x="score", y="label", hue="metric", ax=ax_bar, orient="h", palette=["#2563eb", "#0f766e", "#dc2626"])
    ax_bar.set_xlim(0, 1.02)
    ax_bar.set_title("클래스별 정밀도·재현율·F1", loc="left", fontsize=16, fontweight="bold")
    ax_bar.set_xlabel("점수")
    ax_bar.set_ylabel("")
    ax_bar.legend(title="", frameon=False, loc="lower right")
    ax_bar.text(
        0.03,
        0.97,
        f"Accuracy      {rf_metrics['test_accuracy']:.3f}\nMacro F1      {rf_metrics['test_macro_f1']:.3f}\nWeighted F1   {rf_metrics['test_weighted_f1']:.3f}",
        transform=ax_bar.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#cbd5e1"),
    )

    fig.suptitle("외국인관광도시민박업 외부 테스트: 분류 결과", x=0.06, y=0.995, ha="left", fontsize=22, fontweight="bold")
    fig.text(0.06, 0.962, "500m 반경에서 가장 많이 나타난 인명피해 발화장소 유형을 예측했습니다.", fontsize=10.8, color="#475569")
    fig.tight_layout()

    out = FIG_DIR / "03_classification_results.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_final_dashboard(metadata: dict[str, object], reg_metrics: pd.DataFrame, cls_metrics: pd.DataFrame) -> Path:
    reg_rf = reg_metrics[reg_metrics["model"] == "random_forest"].copy()
    cls_rf = cls_metrics[cls_metrics["model"] == "random_forest"].iloc[0]

    pretty_target = {
        REG_TARGETS[0]: "전체",
        REG_TARGETS[1]: "주거",
        REG_TARGETS[2]: "판매·업무",
        REG_TARGETS[3]: "생활서비스",
    }
    reg_rf["target_short"] = reg_rf["target"].map(pretty_target)

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 1.2], width_ratios=[1, 1, 1], hspace=0.28, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, :2])
    ax_e = fig.add_subplot(gs[1, 2])

    fig.text(0.05, 0.965, "외국인관광도시민박업 화재 노출 외부 일반화 분석", fontsize=25, fontweight="bold", ha="left", color="#0f172a")
    fig.text(
        0.05,
        0.932,
        "학습은 기존 숙박업·관광숙박업에서, 평가는 외국인관광도시민박업에서 수행했습니다. 종속변수는 숙소 주변 500m 인명피해 화재 노출량과 우세 발화장소 유형입니다.",
        fontsize=11,
        color="#475569",
        ha="left",
    )

    split_series = pd.Series(
        {
            "학습 숙소": metadata["train_rows"],
            "테스트 숙소": metadata["test_rows"],
            "필터된 화재": metadata["filtered_fire_rows"],
        }
    )
    sns.barplot(x=split_series.index, y=split_series.values, ax=ax_a, palette=["#2563eb", "#dc2626", "#0f766e"])
    ax_a.set_title("분석 규모", loc="left", fontsize=15, fontweight="bold")
    ax_a.set_xlabel("")
    ax_a.set_ylabel("건수")
    for idx, value in enumerate(split_series.values):
        ax_a.text(idx, value + max(split_series.values) * 0.02, f"{int(value):,}", ha="center", fontsize=10)

    cards = [
        ("분류 정확도", cls_rf["test_accuracy"], "#2563eb"),
        ("분류 Macro F1", cls_rf["test_macro_f1"], "#7c3aed"),
        ("동률 처리 건수", metadata["tie_count_for_classification"], "#ea580c"),
    ]
    ax_b.axis("off")
    ax_b.set_title("핵심 수치", loc="left", fontsize=15, fontweight="bold")
    for idx, (label, value, color) in enumerate(cards):
        y = 0.82 - idx * 0.28
        ax_b.add_patch(plt.Rectangle((0.02, y - 0.13), 0.96, 0.2, color="#ffffff", ec="#e2e8f0", lw=1.2, transform=ax_b.transAxes))
        ax_b.text(0.06, y + 0.03, label, transform=ax_b.transAxes, fontsize=11, color="#475569")
        text = f"{value:.3f}" if isinstance(value, float) else f"{int(value):,}"
        ax_b.text(0.06, y - 0.06, text, transform=ax_b.transAxes, fontsize=20, fontweight="bold", color=color)

    best_reg = reg_rf.sort_values("test_r2", ascending=False)
    sns.barplot(data=best_reg, x="test_r2", y="target_short", ax=ax_c, color="#0f766e")
    ax_c.axvline(0, color="#94a3b8", linewidth=1.2)
    ax_c.set_title("회귀 외부 테스트 R²", loc="left", fontsize=15, fontweight="bold")
    ax_c.set_xlabel("R²")
    ax_c.set_ylabel("")
    for idx, (_, row) in enumerate(best_reg.iterrows()):
        ax_c.text(row["test_r2"], idx, f" {row['test_r2']:.3f}", va="center", ha="left" if row["test_r2"] >= 0 else "right", fontsize=10)

    heat = reg_rf[["target_short", "test_mae", "test_rmse", "test_r2"]].set_index("target_short")
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.4, linecolor="#f8fafc", cbar=False, ax=ax_d)
    ax_d.set_title("회귀 타깃별 외부 테스트 성능", loc="left", fontsize=16, fontweight="bold")
    ax_d.set_xlabel("")
    ax_d.set_ylabel("")

    ax_e.axis("off")
    ax_e.set_title("해석 포인트", loc="left", fontsize=15, fontweight="bold")
    bullet_lines = [
        f"분석 연도는 {ANALYSIS_YEAR_START}–{ANALYSIS_YEAR_END}년으로 고정했습니다.",
        "회귀 타깃은 500m 반경 인명피해 화재 노출량(log1p)입니다.",
        "분류 타깃은 500m 반경에서 가장 우세한 발화장소 유형입니다.",
        "업종 변수는 학습에 넣지 않아, 유형 전이 가능성만 보도록 설계했습니다.",
        "외국인관광도시민박업은 별도 외부 테스트셋으로만 사용했습니다.",
    ]
    ax_e.text(
        0.03,
        0.95,
        "\n".join(f"• {line}" for line in bullet_lines),
        transform=ax_e.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="#334155",
        linespacing=1.75,
    )

    out = FIG_DIR / "04_final_dashboard.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    setup_style()
    ensure_dirs()

    lodging = prepare_lodging_data(LODGING_CSV)
    fire = prepare_fire_data(FIRE_CSV)
    lodging_targets, meta_targets = attach_targets(lodging, fire)
    train_df, test_df = split_domain(lodging_targets)

    target_dataset_path = TABLE_DIR / "lodging_targets_injury_fire_top3.csv"
    lodging_targets.to_csv(target_dataset_path, index=False, encoding="utf-8-sig")

    reg_metrics, reg_preds = train_regressions(train_df, test_df)
    cls_metrics, cls_preds, cls_report = train_classifier(train_df, test_df)

    reg_metrics.to_csv(TABLE_DIR / "regression_metrics.csv", index=False, encoding="utf-8-sig")
    reg_preds.to_csv(TABLE_DIR / "regression_test_predictions.csv", index=False, encoding="utf-8-sig")
    cls_metrics.to_csv(TABLE_DIR / "classification_metrics.csv", index=False, encoding="utf-8-sig")
    cls_preds.to_csv(TABLE_DIR / "classification_test_predictions.csv", index=False, encoding="utf-8-sig")
    cls_report.to_csv(TABLE_DIR / "classification_report.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "lodging_rows": int(len(lodging_targets)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "filtered_fire_rows": int(len(fire)),
        "fire_year_min": int(fire["발생연도"].min()) if len(fire) else None,
        "fire_year_max": int(fire["발생연도"].max()) if len(fire) else None,
        **meta_targets,
        "train_domain": DOMAIN_TRAIN,
        "test_domain": DOMAIN_TEST,
        "regression_targets": REG_TARGETS,
        "classification_target": CLS_TARGET,
    }
    with (TABLE_DIR / "analysis_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)

    fig_paths = [
        plot_target_distribution(lodging_targets, fire),
        plot_regression_results(reg_metrics, reg_preds),
        plot_classification_results(cls_metrics, cls_preds, cls_report),
        plot_final_dashboard(metadata, reg_metrics, cls_metrics),
    ]

    print("[완료] 숙소 단위 타깃 생성 및 외부 일반화 분석")
    print(f"  target dataset : {target_dataset_path}")
    for path in fig_paths:
        print(f"  figure         : {path}")


if __name__ == "__main__":
    main()
