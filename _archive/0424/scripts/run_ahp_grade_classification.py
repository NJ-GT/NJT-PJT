from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None


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
MODEL_ORDER = [
    "Dummy",
    "LogisticRegression",
    "LinearSVC",
    "RandomForest",
    "ExtraTrees",
    "XGBoost",
    "LightGBM",
]
MODEL_COLORS = {
    "Dummy": "#B8B8B8",
    "LogisticRegression": "#3B82F6",
    "LinearSVC": "#6C5CE7",
    "RandomForest": "#D1495B",
    "ExtraTrees": "#2A9D8F",
    "XGBoost": "#F4A261",
    "LightGBM": "#5B8E7D",
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
    "연면적",
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


def build_models() -> dict[str, Pipeline]:
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
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    models: dict[str, Pipeline] = {
        "Dummy": Pipeline(
            steps=[
                ("prep", preprocess),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "LogisticRegression": Pipeline(
            steps=[
                ("prep", preprocess),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "LinearSVC": Pipeline(
            steps=[
                ("prep", preprocess),
                (
                    "model",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=42,
                        dual="auto",
                        max_iter=6000,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("prep", preprocess),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        random_state=42,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "ExtraTrees": Pipeline(
            steps=[
                ("prep", preprocess),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=700,
                        min_samples_leaf=2,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline(
            steps=[
                ("prep", preprocess),
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
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    if LGBMClassifier is not None:
        models["LightGBM"] = Pipeline(
            steps=[
                ("prep", preprocess),
                (
                    "model",
                    LGBMClassifier(
                        objective="multiclass",
                        n_estimators=500,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                        class_weight="balanced",
                        verbosity=-1,
                    ),
                ),
            ]
        )

    return models


def evaluate_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    metrics_rows: list[dict[str, object]] = []
    report_rows: list[dict[str, object]] = []
    best_runs: dict[str, dict[str, object]] = {}

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    models = build_models()

    label_to_int = {label: idx for idx, label in enumerate(GRADE_ORDER)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}

    for target_col, target_label in TARGETS.items():
        y = df[target_col].astype(str)
        y_encoded = y.map(label_to_int).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=0.25,
            random_state=42,
            stratify=y_encoded,
        )

        for model_name, pipeline in models.items():
            pipeline.fit(X_train, y_train)
            train_pred = pipeline.predict(X_train)
            pred = pipeline.predict(X_test)
            train_pred = pd.Series(train_pred, index=y_train.index).astype(int)
            pred = pd.Series(pred, index=y_test.index).astype(int)
            train_accuracy = accuracy_score(y_train.astype(int), train_pred)
            y_test_series = y_test.astype(int)
            pred_labels = pred.map(int_to_label)
            y_test_labels = y_test_series.map(int_to_label)

            accuracy = accuracy_score(y_test_series, pred)
            balanced_acc = balanced_accuracy_score(y_test_series, pred)
            macro_f1 = f1_score(y_test_series, pred, average="macro")
            weighted_f1 = f1_score(y_test_series, pred, average="weighted")

            metrics_rows.append(
                {
                    "target_col": target_col,
                    "target_label": target_label,
                    "model": model_name,
                    "train_rows": int(len(X_train)),
                    "test_rows": int(len(X_test)),
                    "train_accuracy": round(float(train_accuracy), 4),
                    "accuracy": round(float(accuracy), 4),
                    "accuracy_gap": round(float(train_accuracy - accuracy), 4),
                    "balanced_accuracy": round(float(balanced_acc), 4),
                    "macro_f1": round(float(macro_f1), 4),
                    "weighted_f1": round(float(weighted_f1), 4),
                }
            )

            report = classification_report(
                y_test_labels,
                pred_labels,
                labels=GRADE_ORDER,
                output_dict=True,
                zero_division=0,
            )
            for label in GRADE_ORDER:
                report_rows.append(
                    {
                        "target_col": target_col,
                        "target_label": target_label,
                        "model": model_name,
                        "class": label,
                        "precision": round(float(report[label]["precision"]), 4),
                        "recall": round(float(report[label]["recall"]), 4),
                        "f1_score": round(float(report[label]["f1-score"]), 4),
                        "support": int(report[label]["support"]),
                    }
                )

            if model_name != "Dummy":
                current_best = best_runs.get(target_col)
                if current_best is None or macro_f1 > current_best["macro_f1"]:
                    best_runs[target_col] = {
                        "model_name": model_name,
                        "pipeline": pipeline,
                        "X_test": X_test.copy(),
                        "y_test": y_test_labels.copy(),
                        "pred": pred_labels.copy(),
                        "macro_f1": float(macro_f1),
                    }

    metrics_df = pd.DataFrame(metrics_rows)
    report_df = pd.DataFrame(report_rows)
    return metrics_df, report_df, best_runs


def plot_metric_dashboard(metrics_df: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.3), facecolor="#F7F3EC")
    fig.subplots_adjust(top=0.78, bottom=0.14, wspace=0.22)
    metric_names = ["accuracy", "balanced_accuracy", "macro_f1"]
    metric_titles = ["Accuracy", "Balanced Accuracy", "Macro F1"]
    model_order = [m for m in MODEL_ORDER if m in metrics_df["model"].unique()]

    for ax, metric_name, metric_title in zip(axes, metric_names, metric_titles):
        pivot = (
            metrics_df.pivot(index="target_label", columns="model", values=metric_name)
            .reindex(TARGETS.values())
        )
        x = np.arange(len(pivot.index))
        width = 0.72 / len(model_order)
        for idx, model_name in enumerate(model_order):
            vals = pivot[model_name].values
            ax.bar(
                x + (idx - (len(model_order) - 1) / 2) * width,
                vals,
                width=width * 0.96,
                color=MODEL_COLORS[model_name],
                label=model_name,
                edgecolor="white",
            )
            for xpos, val in zip(x + (idx - (len(model_order) - 1) / 2) * width, vals):
                ax.text(xpos, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x, pivot.index)
        ax.set_ylim(0, 1.02)
        ax.set_title(metric_title, fontsize=14, fontweight="bold")
        ax.grid(alpha=0.18, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=min(6, len(labels)), loc="upper center", bbox_to_anchor=(0.5, 0.92))
    fig.suptitle("위험점수_AHP 3등급 분류모델 성능 비교", fontsize=20, fontweight="bold", y=0.985)
    fig.text(
        0.02,
        0.02,
        "같은 입력변수로 공통 기준 3등급과 업종군별 기준 3등급을 각각 분류한 결과. Dummy는 최빈값 기준선이고, XGBoost는 설치된 경우 함께 비교함.",
        fontsize=11,
        color="#5A5A5A",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_train_test_gap(metrics_df: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    model_order = [m for m in MODEL_ORDER if m in metrics_df["model"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), facecolor="#F7F3EC")
    fig.subplots_adjust(top=0.80, bottom=0.16, wspace=0.20)

    for ax, target_label in zip(axes, TARGETS.values()):
        sub = metrics_df.loc[metrics_df["target_label"].eq(target_label)].copy()
        sub["model"] = pd.Categorical(sub["model"], categories=model_order, ordered=True)
        sub = sub.sort_values("model")
        x = np.arange(len(sub))
        width = 0.34

        ax.bar(
            x - width / 2,
            sub["train_accuracy"],
            width=width,
            color="#7C8DA6",
            edgecolor="white",
            label="train",
        )
        ax.bar(
            x + width / 2,
            sub["accuracy"],
            width=width,
            color="#D1495B",
            edgecolor="white",
            label="test",
        )

        for xpos, tr, te, gap in zip(x, sub["train_accuracy"], sub["accuracy"], sub["accuracy_gap"]):
            ax.plot([xpos - width / 2, xpos + width / 2], [tr, te], color="#233D4D", linewidth=1.4, alpha=0.75)
            ax.text(xpos, max(tr, te) + 0.018, f"gap {gap:.3f}", ha="center", va="bottom", fontsize=8.8, color="#233D4D")

        ax.set_xticks(x, sub["model"], rotation=18)
        ax.set_ylim(0, 1.05)
        ax.set_title(target_label, fontsize=15, fontweight="bold")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.16, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("모델별 Train-Test 정확도 차이", fontsize=20, fontweight="bold", y=0.985)
    fig.text(
        0.02,
        0.03,
        "막대는 train/test accuracy, 선과 라벨은 정확도 차이(gap). gap이 클수록 과적합 가능성이 큼.",
        fontsize=11,
        color="#5A5A5A",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(best_runs: dict[str, dict[str, object]], output_path: Path) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), facecolor="#F7F3EC")
    cmap = "YlOrRd"

    for ax, target_col in zip(axes, TARGETS.keys()):
        run = best_runs[target_col]
        disp = ConfusionMatrixDisplay.from_predictions(
            run["y_test"],
            run["pred"],
            labels=GRADE_ORDER,
            normalize="true",
            cmap=cmap,
            ax=ax,
            colorbar=False,
            values_format=".2f",
        )
        ax.set_title(
            f"{TARGETS[target_col]}\nBest: {run['model_name']} (Macro F1 {run['macro_f1']:.3f})",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("예측 등급")
        ax.set_ylabel("실제 등급")

    fig.suptitle("최우수 분류모델 혼동행렬", fontsize=19, fontweight="bold", y=0.98)
    fig.text(
        0.02,
        0.02,
        "행 기준 정규화 혼동행렬. 각 실제 등급이 어느 등급으로 분류됐는지 비율로 표시함.",
        fontsize=11,
        color="#5A5A5A",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_best_predictions(best_runs: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for target_col, run in best_runs.items():
        pred_df = run["X_test"].copy()
        pred_df["실제등급"] = run["y_test"].values
        pred_df["예측등급"] = run["pred"].values
        pred_df["target_col"] = target_col
        pred_df["target_label"] = TARGETS[target_col]
        pred_df["best_model"] = run["model_name"]
        rows.append(pred_df)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    metrics_df, report_df, best_runs = evaluate_models(df)
    pred_df = save_best_predictions(best_runs)

    metrics_path = TABLE_DIR / "AHP_3등급_분류성능.csv"
    report_path = TABLE_DIR / "AHP_3등급_분류리포트.csv"
    pred_path = TABLE_DIR / "AHP_3등급_최우수모델_예측결과.csv"
    metric_fig_path = FIG_DIR / "AHP_3등급_분류성능비교.png"
    gap_fig_path = FIG_DIR / "AHP_3등급_TrainTest_정확도차이.png"
    cm_fig_path = FIG_DIR / "AHP_3등급_혼동행렬.png"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    plot_metric_dashboard(metrics_df, metric_fig_path)
    plot_train_test_gap(metrics_df, gap_fig_path)
    plot_confusion_matrices(best_runs, cm_fig_path)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved reports: {report_path}")
    print(f"Saved predictions: {pred_path}")
    print(f"Saved metric figure: {metric_fig_path}")
    print(f"Saved train-test gap figure: {gap_fig_path}")
    print(f"Saved confusion figure: {cm_fig_path}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
