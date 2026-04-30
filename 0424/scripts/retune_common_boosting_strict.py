from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "0424" / "분석" / "tables" / "분석변수_최종테이블0423_AHP3등급비교.csv"
OUTPUT_DIR = ROOT / "0424" / "분석"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"

TARGET_COL = "공통_3등급"
TARGET_LABEL = "공통 기준 3등급"
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
    "연면적",
]
CATEGORICAL_FEATURES = [
    "구",
    "동",
    "업종",
    "건물용도명",
    "업종그룹",
]
GRADE_ORDER = ["안전", "보통", "위험"]
COLORS = {"baseline": "#7C8DA6", "tuned": "#D1495B"}

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    df = df.dropna(subset=[TARGET_COL]).copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("미상").astype(str)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
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


def score_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def get_best_iteration(model, fallback: int) -> int:
    for attr in ["best_iteration_", "best_iteration"]:
        value = getattr(model, attr, None)
        if value is not None:
            return max(int(value), 1)
    return fallback


def train_lightgbm(params: dict[str, object], X_train, y_train, X_val, y_val):
    model = LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
        n_jobs=1,
        **params,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb_early_stopping(stopping_rounds=60, verbose=False)],
    )
    train_scores = score_metrics(y_train, model.predict(X_train))
    val_scores = score_metrics(y_val, model.predict(X_val))
    gap = train_scores["accuracy"] - val_scores["accuracy"]
    selection = val_scores["accuracy"] - 0.20 * max(gap, 0)
    return model, train_scores, val_scores, gap, selection


def train_xgboost(params: dict[str, object], X_train, y_train, X_val, y_val):
    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        early_stopping_rounds=60,
        **params,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    train_scores = score_metrics(y_train, model.predict(X_train))
    val_scores = score_metrics(y_val, model.predict(X_val))
    gap = train_scores["accuracy"] - val_scores["accuracy"]
    selection = val_scores["accuracy"] - 0.12 * max(gap, 0)
    return model, train_scores, val_scores, gap, selection


def fit_final_lightgbm(params: dict[str, object], X_train_full, y_train_full) -> LGBMClassifier:
    model = LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
        n_jobs=1,
        **params,
    )
    model.fit(X_train_full, y_train_full)
    return model


def fit_final_xgboost(params: dict[str, object], X_train_full, y_train_full) -> XGBClassifier:
    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        **params,
    )
    model.fit(X_train_full, y_train_full)
    return model


def evaluate_final(model, X_train_full, y_train_full, X_test, y_test) -> dict[str, float]:
    train_scores = score_metrics(y_train_full, model.predict(X_train_full))
    test_scores = score_metrics(y_test, model.predict(X_test))
    return {
        "train_accuracy": train_scores["accuracy"],
        "test_accuracy": test_scores["accuracy"],
        "gap": train_scores["accuracy"] - test_scores["accuracy"],
        "test_macro_f1": test_scores["macro_f1"],
        "test_balanced_accuracy": test_scores["balanced_accuracy"],
    }


def tune_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_to_int = {label: idx for idx, label in enumerate(GRADE_ORDER)}
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COL].map(label_to_int).astype(int)

    X_train_full_raw, X_test_raw, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_full_raw,
        y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full,
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_train_full = preprocessor.transform(X_train_full_raw)
    X_test = preprocessor.transform(X_test_raw)

    xgb_baseline = {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 2,
        "gamma": 0.1,
        "reg_alpha": 0.05,
        "reg_lambda": 1.5,
    }
    xgb_dist = {
        "n_estimators": [300, 450, 600, 800, 1000, 1200],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.015, 0.02, 0.03, 0.05],
        "subsample": [0.65, 0.75, 0.85, 0.95],
        "colsample_bytree": [0.65, 0.75, 0.85, 0.95],
        "min_child_weight": [2, 4, 6, 8, 10],
        "gamma": [0.05, 0.1, 0.2, 0.4, 0.6],
        "reg_alpha": [0.01, 0.05, 0.1, 0.3, 0.6],
        "reg_lambda": [1.0, 1.5, 2.0, 3.0, 5.0],
    }
    lgb_baseline = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 11,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 60,
        "min_split_gain": 0.1,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
    }
    lgb_dist = {
        "n_estimators": [300, 500, 700, 900, 1200],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "num_leaves": [5, 7, 9, 11, 15],
        "max_depth": [3, 4, 5, 6],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "min_child_samples": [40, 60, 80, 120, 160],
        "min_split_gain": [0.05, 0.1, 0.2, 0.4],
        "reg_alpha": [0.1, 0.5, 1.0, 2.0, 4.0],
        "reg_lambda": [1.0, 2.0, 4.0, 8.0, 12.0],
    }

    summary_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for model_name, baseline_params, dist, trainer, fitter in [
        ("XGBoost", xgb_baseline, xgb_dist, train_xgboost, fit_final_xgboost),
        ("LightGBM", lgb_baseline, lgb_dist, train_lightgbm, fit_final_lightgbm),
    ]:
        baseline_model, _, baseline_val_scores, baseline_val_gap, baseline_sel = trainer(
            baseline_params, X_train, y_train, X_val, y_val
        )
        baseline_iter = get_best_iteration(baseline_model, int(baseline_params["n_estimators"]))
        baseline_final_params = dict(baseline_params)
        baseline_final_params["n_estimators"] = baseline_iter
        baseline_final = fitter(baseline_final_params, X_train_full, y_train_full)
        baseline_final_scores = evaluate_final(baseline_final, X_train_full, y_train_full, X_test, y_test)

        best_score = -1e9
        best_params = None
        best_model = None
        best_val_scores = None
        best_val_gap = None

        for idx, params in enumerate(ParameterSampler(dist, n_iter=12, random_state=42), start=1):
            model, train_scores, val_scores, gap, selection = trainer(params, X_train, y_train, X_val, y_val)
            candidate_rows.append(
                {
                    "model": model_name,
                    "candidate_idx": idx,
                    "params": str(params),
                    "val_accuracy": round(val_scores["accuracy"], 4),
                    "val_macro_f1": round(val_scores["macro_f1"], 4),
                    "val_gap": round(gap, 4),
                    "selection_score": round(selection, 4),
                }
            )
            if selection > best_score:
                best_score = selection
                best_params = params
                best_model = model
                best_val_scores = val_scores
                best_val_gap = gap

        assert best_params is not None and best_model is not None
        tuned_iter = get_best_iteration(best_model, int(best_params["n_estimators"]))
        tuned_final_params = dict(best_params)
        tuned_final_params["n_estimators"] = tuned_iter
        tuned_final = fitter(tuned_final_params, X_train_full, y_train_full)
        tuned_final_scores = evaluate_final(tuned_final, X_train_full, y_train_full, X_test, y_test)

        summary_rows.append(
            {
                "target_col": TARGET_COL,
                "target_label": TARGET_LABEL,
                "model": model_name,
                "baseline_params": str(baseline_final_params),
                "baseline_val_accuracy": round(baseline_val_scores["accuracy"], 4),
                "baseline_val_gap": round(baseline_val_gap, 4),
                "baseline_test_accuracy": round(baseline_final_scores["test_accuracy"], 4),
                "baseline_gap": round(baseline_final_scores["gap"], 4),
                "baseline_macro_f1": round(baseline_final_scores["test_macro_f1"], 4),
                "tuned_params": str(tuned_final_params),
                "tuned_val_accuracy": round(best_val_scores["accuracy"], 4),
                "tuned_val_gap": round(best_val_gap, 4),
                "tuned_test_accuracy": round(tuned_final_scores["test_accuracy"], 4),
                "tuned_gap": round(tuned_final_scores["gap"], 4),
                "tuned_macro_f1": round(tuned_final_scores["test_macro_f1"], 4),
                "test_accuracy_delta": round(tuned_final_scores["test_accuracy"] - baseline_final_scores["test_accuracy"], 4),
                "gap_delta": round(tuned_final_scores["gap"] - baseline_final_scores["gap"], 4),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(candidate_rows)


def plot_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), facecolor="#F7F3EC")
    fig.subplots_adjust(top=0.82, bottom=0.15, wspace=0.24)
    x = np.arange(len(summary_df))
    width = 0.28

    axes[0].bar(x - width / 2, summary_df["baseline_test_accuracy"], width=width, color=COLORS["baseline"], edgecolor="white", label="baseline")
    axes[0].bar(x + width / 2, summary_df["tuned_test_accuracy"], width=width, color=COLORS["tuned"], edgecolor="white", label="tuned")
    for xpos, base, tuned, delta in zip(x, summary_df["baseline_test_accuracy"], summary_df["tuned_test_accuracy"], summary_df["test_accuracy_delta"]):
        axes[0].text(xpos - width / 2, base + 0.012, f"{base:.3f}", ha="center", va="bottom", fontsize=9)
        axes[0].text(xpos + width / 2, tuned + 0.012, f"{tuned:.3f}", ha="center", va="bottom", fontsize=9)
        axes[0].text(xpos, max(base, tuned) + 0.045, f"Δ {delta:+.3f}", ha="center", va="bottom", fontsize=10, color="#233D4D")
    axes[0].set_xticks(x, summary_df["model"])
    axes[0].set_ylim(0.68, 0.84)
    axes[0].set_title("공통 기준 test accuracy", fontsize=15, fontweight="bold")
    axes[0].grid(alpha=0.18, axis="y")

    axes[1].bar(x - width / 2, summary_df["baseline_gap"], width=width, color=COLORS["baseline"], edgecolor="white")
    axes[1].bar(x + width / 2, summary_df["tuned_gap"], width=width, color=COLORS["tuned"], edgecolor="white")
    for xpos, base_gap, tuned_gap, delta in zip(x, summary_df["baseline_gap"], summary_df["tuned_gap"], summary_df["gap_delta"]):
        axes[1].text(xpos - width / 2, base_gap + 0.006, f"{base_gap:.3f}", ha="center", va="bottom", fontsize=9)
        axes[1].text(xpos + width / 2, tuned_gap + 0.006, f"{tuned_gap:.3f}", ha="center", va="bottom", fontsize=9)
        axes[1].text(xpos, max(base_gap, tuned_gap) + 0.028, f"Δ {delta:+.3f}", ha="center", va="bottom", fontsize=10, color="#233D4D")
    axes[1].set_xticks(x, summary_df["model"])
    axes[1].set_ylim(0.14, 0.28)
    axes[1].set_title("공통 기준 train-test gap", fontsize=15, fontweight="bold")
    axes[1].grid(alpha=0.18, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("공통 기준 집중 재튜닝: 얕은 XGBoost vs 강한 규제 LightGBM", fontsize=20, fontweight="bold", y=0.985)
    fig.text(
        0.02,
        0.03,
        "왼쪽은 test accuracy, 오른쪽은 train-test gap. accuracy는 올라가고 gap은 내려가면 가장 이상적임.",
        fontsize=11,
        color="#5A5A5A",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    summary_df, candidate_df = tune_models(df)

    summary_path = TABLE_DIR / "공통기준_부스팅집중재튜닝_요약.csv"
    candidate_path = TABLE_DIR / "공통기준_부스팅집중재튜닝_후보.csv"
    fig_path = FIG_DIR / "공통기준_부스팅집중재튜닝_비교.png"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    candidate_df.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    plot_summary(summary_df, fig_path)

    print(f"Saved summary: {summary_path}")
    print(f"Saved candidates: {candidate_path}")
    print(f"Saved figure: {fig_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
