from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "0424" / "분석" / "tables" / "분석변수_최종테이블0423_AHP3등급비교.csv"
OUTPUT_DIR = ROOT / "0424" / "분석"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"

GRADE_ORDER = ["안전", "보통", "위험"]
TARGETS = {
    "공통_3등급": "공통 기준 3등급",
    "업종군별_3등급": "업종군별 기준 3등급",
}
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
]
CATEGORICAL_FEATURES = [
    "구",
    "동",
    "업종",
    "건물용도명",
    "업종그룹",
]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    df = df.dropna(subset=list(TARGETS.keys())).copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("미상").astype(str)
    return df


def build_preprocessor() -> ColumnTransformer:
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


def build_baseline_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            (
                "model",
                XGBClassifier(
                    n_estimators=500,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    tree_method="hist",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_search_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    tree_method="hist",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def tune_one_target(df: pd.DataFrame, target_col: str, target_label: str) -> tuple[dict[str, object], pd.DataFrame]:
    label_to_int = {label: idx for idx, label in enumerate(GRADE_ORDER)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[target_col].map(label_to_int).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    baseline = build_baseline_pipeline()
    baseline.fit(X_train, y_train)
    baseline_pred = pd.Series(baseline.predict(X_test), index=y_test.index).astype(int)
    baseline_scores = evaluate_predictions(y_test, baseline_pred)

    search = RandomizedSearchCV(
        estimator=build_search_pipeline(),
        param_distributions={
            "model__n_estimators": [200, 300, 400, 500, 700, 900],
            "model__max_depth": [3, 4, 5, 6, 7, 8],
            "model__learning_rate": [0.02, 0.03, 0.05, 0.07, 0.1],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "model__min_child_weight": [1, 2, 4, 6, 8],
            "model__gamma": [0.0, 0.05, 0.1, 0.2, 0.4],
            "model__reg_alpha": [0.0, 0.001, 0.01, 0.05, 0.1],
            "model__reg_lambda": [0.8, 1.0, 1.2, 1.5, 2.0],
        },
        n_iter=24,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=1,
        verbose=0,
        refit=True,
    )
    search.fit(X_train, y_train)
    tuned_pred = pd.Series(search.best_estimator_.predict(X_test), index=y_test.index).astype(int)
    tuned_scores = evaluate_predictions(y_test, tuned_pred)

    compare_rows = [
        {
            "target_col": target_col,
            "target_label": target_label,
            "variant": "baseline",
            "cv_best_macro_f1": np.nan,
            "accuracy": round(baseline_scores["accuracy"], 4),
            "balanced_accuracy": round(baseline_scores["balanced_accuracy"], 4),
            "macro_f1": round(baseline_scores["macro_f1"], 4),
            "weighted_f1": round(baseline_scores["weighted_f1"], 4),
        },
        {
            "target_col": target_col,
            "target_label": target_label,
            "variant": "tuned",
            "cv_best_macro_f1": round(float(search.best_score_), 4),
            "accuracy": round(tuned_scores["accuracy"], 4),
            "balanced_accuracy": round(tuned_scores["balanced_accuracy"], 4),
            "macro_f1": round(tuned_scores["macro_f1"], 4),
            "weighted_f1": round(tuned_scores["weighted_f1"], 4),
        },
    ]

    param_df = pd.DataFrame(
        [
            {
                "target_col": target_col,
                "target_label": target_label,
                "variant": "baseline",
                "params": str(baseline.named_steps["model"].get_params()),
            },
            {
                "target_col": target_col,
                "target_label": target_label,
                "variant": "tuned",
                "params": str(search.best_estimator_.named_steps["model"].get_params()),
            },
        ]
    )

    run_info = {
        "target_col": target_col,
        "target_label": target_label,
        "baseline_true": y_test.map(int_to_label),
        "baseline_pred": baseline_pred.map(int_to_label),
        "tuned_true": y_test.map(int_to_label),
        "tuned_pred": tuned_pred.map(int_to_label),
        "baseline_scores": baseline_scores,
        "tuned_scores": tuned_scores,
        "best_cv_macro_f1": float(search.best_score_),
    }
    return run_info, pd.concat([pd.DataFrame(compare_rows), param_df], axis=0)


def build_outputs(df: pd.DataFrame) -> tuple[list[dict[str, object]], pd.DataFrame]:
    runs: list[dict[str, object]] = []
    table_parts: list[pd.DataFrame] = []
    for target_col, target_label in TARGETS.items():
        run_info, table_df = tune_one_target(df, target_col, target_label)
        runs.append(run_info)
        table_parts.append(table_df)
    return runs, pd.concat(table_parts, ignore_index=True)


def plot_metric_comparison(runs: list[dict[str, object]], output_path: Path) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), facecolor="#F7F3EC")
    fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.22)
    metric_names = ["accuracy", "balanced_accuracy", "macro_f1"]
    metric_titles = ["Accuracy", "Balanced Accuracy", "Macro F1"]
    colors = {"baseline": "#7C8DA6", "tuned": "#D1495B"}

    for ax, metric_name, metric_title in zip(axes, metric_names, metric_titles):
        x = np.arange(len(runs))
        width = 0.28
        baseline_vals = [run["baseline_scores"][metric_name] for run in runs]
        tuned_vals = [run["tuned_scores"][metric_name] for run in runs]

        ax.bar(x - width / 2, baseline_vals, width=width, color=colors["baseline"], label="baseline", edgecolor="white")
        ax.bar(x + width / 2, tuned_vals, width=width, color=colors["tuned"], label="tuned", edgecolor="white")

        for xpos, val in zip(x - width / 2, baseline_vals):
            ax.text(xpos, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        for xpos, val in zip(x + width / 2, tuned_vals):
            ax.text(xpos, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x, [run["target_label"] for run in runs])
        ax.set_ylim(0, 1.02)
        ax.set_title(metric_title, fontsize=14, fontweight="bold")
        ax.grid(alpha=0.18, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("XGBoost 튜닝 전후 성능 비교", fontsize=20, fontweight="bold", y=0.985)
    fig.text(
        0.02,
        0.03,
        "같은 테스트셋에서 baseline XGBoost와 튜닝된 XGBoost를 비교. 튜닝은 훈련셋 내부 3-fold CV에서 macro F1 기준으로 탐색함.",
        fontsize=11,
        color="#5A5A5A",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusions(runs: list[dict[str, object]], output_path: Path) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 11.2), facecolor="#F7F3EC")
    cmap = "YlOrRd"

    for row_idx, run in enumerate(runs):
        disp_base = ConfusionMatrixDisplay.from_predictions(
            run["baseline_true"],
            run["baseline_pred"],
            labels=GRADE_ORDER,
            normalize="true",
            cmap=cmap,
            ax=axes[row_idx, 0],
            colorbar=False,
            values_format=".2f",
        )
        axes[row_idx, 0].set_title(
            f"{run['target_label']} baseline\nMacro F1 {run['baseline_scores']['macro_f1']:.3f}",
            fontsize=13,
            fontweight="bold",
        )
        axes[row_idx, 0].set_xlabel("예측 등급")
        axes[row_idx, 0].set_ylabel("실제 등급")

        disp_tuned = ConfusionMatrixDisplay.from_predictions(
            run["tuned_true"],
            run["tuned_pred"],
            labels=GRADE_ORDER,
            normalize="true",
            cmap=cmap,
            ax=axes[row_idx, 1],
            colorbar=False,
            values_format=".2f",
        )
        axes[row_idx, 1].set_title(
            f"{run['target_label']} tuned\nMacro F1 {run['tuned_scores']['macro_f1']:.3f}",
            fontsize=13,
            fontweight="bold",
        )
        axes[row_idx, 1].set_xlabel("예측 등급")
        axes[row_idx, 1].set_ylabel("실제 등급")

    fig.suptitle("XGBoost 튜닝 전후 혼동행렬", fontsize=20, fontweight="bold", y=0.985)
    fig.text(
        0.02,
        0.02,
        "행 기준 정규화 혼동행렬. 같은 테스트셋에서 baseline과 tuned를 직접 비교함.",
        fontsize=11,
        color="#5A5A5A",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    runs, result_df = build_outputs(df)

    result_path = TABLE_DIR / "AHP_3등급_XGBoost_튜닝비교.csv"
    metric_fig_path = FIG_DIR / "AHP_3등급_XGBoost_튜닝성능비교.png"
    cm_fig_path = FIG_DIR / "AHP_3등급_XGBoost_튜닝혼동행렬.png"

    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    plot_metric_comparison(runs, metric_fig_path)
    plot_confusions(runs, cm_fig_path)

    print(f"Saved tuning table: {result_path}")
    print(f"Saved tuning metric figure: {metric_fig_path}")
    print(f"Saved tuning confusion figure: {cm_fig_path}")
    for run in runs:
        print(
            {
                "target": run["target_label"],
                "baseline_macro_f1": round(run["baseline_scores"]["macro_f1"], 4),
                "tuned_macro_f1": round(run["tuned_scores"]["macro_f1"], 4),
                "best_cv_macro_f1": round(run["best_cv_macro_f1"], 4),
            }
        )


if __name__ == "__main__":
    main()
