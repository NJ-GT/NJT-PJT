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

TARGETS = {
    "공통_3등급": "공통 기준 3등급",
    "업종군별_3등급": "업종군별 기준 3등급",
}
GRADE_ORDER = ["안전", "보통", "위험"]
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
MODEL_COLORS = {"baseline": "#7C8DA6", "tuned": "#D1495B"}

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


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
        steps=[("imputer", SimpleImputer(strategy="median"))]
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


def train_lgbm_with_params(params: dict[str, object], X_train, y_train, X_val, y_val) -> tuple[LGBMClassifier, float, dict[str, float]]:
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
    val_pred = model.predict(X_val)
    val_scores = score_metrics(y_val, val_pred)
    gap = accuracy_score(y_train, model.predict(X_train)) - val_scores["accuracy"]
    score = val_scores["accuracy"] - 0.08 * max(gap, 0)
    return model, score, val_scores


def train_xgb_with_params(params: dict[str, object], X_train, y_train, X_val, y_val) -> tuple[XGBClassifier, float, dict[str, float]]:
    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        **params,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    val_pred = model.predict(X_val)
    val_scores = score_metrics(y_val, val_pred)
    gap = accuracy_score(y_train, model.predict(X_train)) - val_scores["accuracy"]
    score = val_scores["accuracy"] - 0.08 * max(gap, 0)
    return model, score, val_scores


def tune_one_model(
    model_name: str,
    baseline_params: dict[str, object],
    param_distributions: dict[str, list[object]],
    X_train,
    y_train,
    X_val,
    y_val,
    X_train_full,
    y_train_full,
    X_test,
    y_test,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if model_name == "LightGBM":
        trainer = train_lgbm_with_params
    else:
        trainer = train_xgb_with_params

    baseline_model, _, baseline_val_scores = trainer(baseline_params, X_train, y_train, X_val, y_val)
    baseline_best_iter = get_best_iteration(baseline_model, int(baseline_params.get("n_estimators", 500)))
    baseline_refit_params = dict(baseline_params)
    baseline_refit_params["n_estimators"] = baseline_best_iter

    if model_name == "LightGBM":
        final_baseline = LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
            n_jobs=1,
            **baseline_refit_params,
        )
    else:
        final_baseline = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=1,
            **baseline_refit_params,
        )
    final_baseline.fit(X_train_full, y_train_full)
    baseline_train_pred = final_baseline.predict(X_train_full)
    baseline_test_pred = final_baseline.predict(X_test)
    baseline_train_scores = score_metrics(y_train_full, baseline_train_pred)
    baseline_test_scores = score_metrics(y_test, baseline_test_pred)

    sampled_params = list(ParameterSampler(param_distributions, n_iter=8, random_state=42))
    candidate_rows: list[dict[str, object]] = []
    best_model = None
    best_score = -1e9
    best_params = None
    best_val_scores = None

    for idx, params in enumerate(sampled_params, start=1):
        model, score, val_scores = trainer(params, X_train, y_train, X_val, y_val)
        candidate_rows.append(
            {
                "model": model_name,
                "candidate_idx": idx,
                "params": str(params),
                "val_accuracy": round(val_scores["accuracy"], 4),
                "val_macro_f1": round(val_scores["macro_f1"], 4),
                "selection_score": round(score, 4),
            }
        )
        if score > best_score:
            best_score = score
            best_model = model
            best_params = params
            best_val_scores = val_scores

    assert best_model is not None
    tuned_best_iter = get_best_iteration(best_model, int(best_params.get("n_estimators", 500)))
    tuned_refit_params = dict(best_params)
    tuned_refit_params["n_estimators"] = tuned_best_iter

    if model_name == "LightGBM":
        final_tuned = LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
            n_jobs=1,
            **tuned_refit_params,
        )
    else:
        final_tuned = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=1,
            **tuned_refit_params,
        )
    final_tuned.fit(X_train_full, y_train_full)
    tuned_train_pred = final_tuned.predict(X_train_full)
    tuned_test_pred = final_tuned.predict(X_test)
    tuned_train_scores = score_metrics(y_train_full, tuned_train_pred)
    tuned_test_scores = score_metrics(y_test, tuned_test_pred)

    summary = {
        "model": model_name,
        "baseline_params": str(baseline_refit_params),
        "baseline_train_accuracy": baseline_train_scores["accuracy"],
        "baseline_test_accuracy": baseline_test_scores["accuracy"],
        "baseline_gap": baseline_train_scores["accuracy"] - baseline_test_scores["accuracy"],
        "baseline_macro_f1": baseline_test_scores["macro_f1"],
        "tuned_params": str(tuned_refit_params),
        "tuned_train_accuracy": tuned_train_scores["accuracy"],
        "tuned_test_accuracy": tuned_test_scores["accuracy"],
        "tuned_gap": tuned_train_scores["accuracy"] - tuned_test_scores["accuracy"],
        "tuned_macro_f1": tuned_test_scores["macro_f1"],
        "best_val_accuracy": best_val_scores["accuracy"],
        "best_val_macro_f1": best_val_scores["macro_f1"],
    }
    return summary, candidate_rows


def tune_target(df: pd.DataFrame, target_col: str, target_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_to_int = {label: idx for idx, label in enumerate(GRADE_ORDER)}
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[target_col].map(label_to_int).astype(int)

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
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }
    xgb_dist = {
        "n_estimators": [300, 500, 700, 900, 1100],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.02, 0.03, 0.05, 0.07],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 2, 4, 6, 8],
        "gamma": [0.0, 0.05, 0.1, 0.2, 0.4],
        "reg_alpha": [0.0, 0.001, 0.01, 0.05, 0.1, 0.5],
        "reg_lambda": [0.8, 1.0, 1.2, 1.5, 2.0, 3.0],
    }
    lgb_baseline = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_samples": 20,
        "min_split_gain": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    }
    lgb_dist = {
        "n_estimators": [300, 500, 700, 900, 1200],
        "learning_rate": [0.02, 0.03, 0.05, 0.07],
        "num_leaves": [7, 11, 15, 21, 31],
        "max_depth": [3, 4, 5, 6, 8, -1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_samples": [20, 40, 60, 80, 120],
        "min_split_gain": [0.0, 0.02, 0.05, 0.1, 0.2],
        "reg_alpha": [0.0, 0.001, 0.01, 0.05, 0.1, 0.5],
        "reg_lambda": [0.0, 0.1, 0.5, 1.0, 2.0, 4.0],
    }

    summaries = []
    candidate_rows_all: list[dict[str, object]] = []

    for model_name, baseline_params, dist in [
        ("XGBoost", xgb_baseline, xgb_dist),
        ("LightGBM", lgb_baseline, lgb_dist),
    ]:
        summary, candidate_rows = tune_one_model(
            model_name=model_name,
            baseline_params=baseline_params,
            param_distributions=dist,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_train_full=X_train_full,
            y_train_full=y_train_full,
            X_test=X_test,
            y_test=y_test,
        )
        summary["target_col"] = target_col
        summary["target_label"] = target_label
        summaries.append(summary)
        for row in candidate_rows:
            row["target_col"] = target_col
            row["target_label"] = target_label
        candidate_rows_all.extend(candidate_rows)

    return pd.DataFrame(summaries), pd.DataFrame(candidate_rows_all)


def plot_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), facecolor="#F7F3EC")
    fig.subplots_adjust(top=0.86, bottom=0.10, wspace=0.22, hspace=0.28)

    for row_idx, target_label in enumerate(TARGETS.values()):
        sub = summary_df.loc[summary_df["target_label"].eq(target_label)].copy()
        x = np.arange(len(sub))
        width = 0.28

        axes[row_idx, 0].bar(x - width / 2, sub["baseline_test_accuracy"], width=width, color=MODEL_COLORS["baseline"], label="baseline", edgecolor="white")
        axes[row_idx, 0].bar(x + width / 2, sub["tuned_test_accuracy"], width=width, color=MODEL_COLORS["tuned"], label="tuned", edgecolor="white")
        for xpos, base, tuned in zip(x, sub["baseline_test_accuracy"], sub["tuned_test_accuracy"]):
            axes[row_idx, 0].text(xpos - width / 2, base + 0.012, f"{base:.3f}", ha="center", va="bottom", fontsize=9)
            axes[row_idx, 0].text(xpos + width / 2, tuned + 0.012, f"{tuned:.3f}", ha="center", va="bottom", fontsize=9)
        axes[row_idx, 0].set_xticks(x, sub["model"])
        axes[row_idx, 0].set_ylim(0.55, 0.9)
        axes[row_idx, 0].set_title(f"{target_label} test accuracy", fontsize=14, fontweight="bold")
        axes[row_idx, 0].grid(alpha=0.18, axis="y")

        axes[row_idx, 1].bar(x - width / 2, sub["baseline_gap"], width=width, color=MODEL_COLORS["baseline"], edgecolor="white")
        axes[row_idx, 1].bar(x + width / 2, sub["tuned_gap"], width=width, color=MODEL_COLORS["tuned"], edgecolor="white")
        for xpos, base_gap, tuned_gap in zip(x, sub["baseline_gap"], sub["tuned_gap"]):
            axes[row_idx, 1].text(xpos - width / 2, base_gap + 0.006, f"{base_gap:.3f}", ha="center", va="bottom", fontsize=9)
            axes[row_idx, 1].text(xpos + width / 2, tuned_gap + 0.006, f"{tuned_gap:.3f}", ha="center", va="bottom", fontsize=9)
        axes[row_idx, 1].set_xticks(x, sub["model"])
        axes[row_idx, 1].set_ylim(0.0, max(sub["baseline_gap"].max(), sub["tuned_gap"].max()) + 0.08)
        axes[row_idx, 1].set_title(f"{target_label} train-test gap", fontsize=14, fontweight="bold")
        axes[row_idx, 1].grid(alpha=0.18, axis="y")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS["baseline"]),
        plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS["tuned"]),
    ]
    fig.legend(handles, ["baseline", "tuned"], frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("부스팅 모델 튜닝 결과: 정확도와 과적합 변화", fontsize=20, fontweight="bold", y=0.985)
    fig.text(
        0.02,
        0.03,
        "왼쪽은 test accuracy, 오른쪽은 train-test accuracy gap. tuned가 왼쪽은 올라가고 오른쪽은 내려가면 가장 이상적임.",
        fontsize=11,
        color="#5A5A5A",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    summary_parts = []
    candidate_parts = []
    for target_col, target_label in TARGETS.items():
        summary_df, candidate_df = tune_target(df, target_col, target_label)
        summary_parts.append(summary_df)
        candidate_parts.append(candidate_df)

    all_summary = pd.concat(summary_parts, ignore_index=True)
    all_candidates = pd.concat(candidate_parts, ignore_index=True)

    summary_path = TABLE_DIR / "AHP_3등급_부스팅튜닝_요약.csv"
    candidate_path = TABLE_DIR / "AHP_3등급_부스팅튜닝_후보결과.csv"
    fig_path = FIG_DIR / "AHP_3등급_부스팅튜닝_비교.png"

    all_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    all_candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    plot_summary(all_summary, fig_path)

    print(f"Saved summary: {summary_path}")
    print(f"Saved candidates: {candidate_path}")
    print(f"Saved figure: {fig_path}")
    print(all_summary.to_string(index=False))


if __name__ == "__main__":
    main()
