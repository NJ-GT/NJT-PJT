from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "0424" / "분석" / "tables" / "분석변수_최종테이블0423_AHP3등급비교.csv"
TABLE_DIR = ROOT / "0424" / "분석" / "tables"
FIG_DIR = ROOT / "0424" / "분석" / "figures"

TARGET_COL = "공통_3등급"
GRADE_ORDER = ["안전", "보통", "위험"]
DISPLAY_GRADES = ["Safe", "Moderate", "Risk"]
GRADE_TO_INT = {grade: idx for idx, grade in enumerate(GRADE_ORDER)}

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
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

COLORS = {
    "LightGBM": "#81B29A",
    "XGBoost": "#E07A5F",
    "CatBoost": "#F2CC8F",
    "StackingMeta": "#264653",
}


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    df = df.dropna(subset=[TARGET_COL]).copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("미상").astype(str)
    df[TARGET_COL] = df[TARGET_COL].astype(str)
    return df.reset_index(drop=True)


def split_dataset(y: np.ndarray) -> dict[str, np.ndarray]:
    all_idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    meta_train_idx, meta_val_idx = train_test_split(
        train_idx,
        test_size=0.20,
        random_state=42,
        stratify=y[train_idx],
    )
    final_fit_idx, final_val_idx = train_test_split(
        train_idx,
        test_size=0.15,
        random_state=42,
        stratify=y[train_idx],
    )
    return {
        "train": train_idx,
        "test": test_idx,
        "meta_train": meta_train_idx,
        "meta_val": meta_val_idx,
        "final_fit": final_fit_idx,
        "final_val": final_val_idx,
    }


def build_sparse_transformer() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def prepare_views(df: pd.DataFrame, splits: dict[str, np.ndarray]) -> dict[str, object]:
    transformer = build_sparse_transformer()
    X = df[FEATURE_COLUMNS].copy()

    meta_train_df = X.iloc[splits["meta_train"]].copy()
    meta_val_df = X.iloc[splits["meta_val"]].copy()
    final_fit_df = X.iloc[splits["final_fit"]].copy()
    final_val_df = X.iloc[splits["final_val"]].copy()
    train_df = X.iloc[splits["train"]].copy()
    test_df = X.iloc[splits["test"]].copy()

    X_meta_train_sparse = transformer.fit_transform(meta_train_df)
    X_meta_val_sparse = transformer.transform(meta_val_df)
    X_final_fit_sparse = transformer.transform(final_fit_df)
    X_final_val_sparse = transformer.transform(final_val_df)
    X_train_sparse = transformer.transform(train_df)
    X_test_sparse = transformer.transform(test_df)

    for frame in [meta_train_df, meta_val_df, final_fit_df, final_val_df, train_df, test_df]:
        frame.loc[:, NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].fillna(meta_train_df[NUMERIC_FEATURES].median())
        frame.loc[:, CATEGORICAL_FEATURES] = frame[CATEGORICAL_FEATURES].fillna("미상").astype(str)

    return {
        "meta_train_raw": meta_train_df,
        "meta_val_raw": meta_val_df,
        "final_fit_raw": final_fit_df,
        "final_val_raw": final_val_df,
        "train_raw": train_df,
        "test_raw": test_df,
        "meta_train_sparse": X_meta_train_sparse,
        "meta_val_sparse": X_meta_val_sparse,
        "final_fit_sparse": X_final_fit_sparse,
        "final_val_sparse": X_final_val_sparse,
        "train_sparse": X_train_sparse,
        "test_sparse": X_test_sparse,
    }


def fit_meta_generation_models(
    views: dict[str, object],
    y_meta_train: np.ndarray,
) -> dict[str, object]:
    cat_indices = [views["meta_train_raw"].columns.get_loc(col) for col in CATEGORICAL_FEATURES]

    models: dict[str, object] = {}

    lgb = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=35,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.3,
        random_state=42,
        verbosity=-1,
    )
    lgb.fit(views["meta_train_sparse"], y_meta_train)
    models["LightGBM"] = lgb

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=900,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.1,
        min_child_weight=2,
        gamma=0.1,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1,
    )
    xgb.fit(views["meta_train_sparse"], y_meta_train)
    models["XGBoost"] = xgb

    cat = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=1600,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=6.0,
        random_seed=42,
        verbose=False,
    )
    cat.fit(
        Pool(views["meta_train_raw"], y_meta_train, cat_features=cat_indices),
        verbose=False,
    )
    models["CatBoost"] = (cat, cat_indices)
    return models


def fit_final_models(
    views: dict[str, object],
    y_final_fit: np.ndarray,
    y_final_val: np.ndarray,
) -> dict[str, object]:
    cat_indices = [views["final_fit_raw"].columns.get_loc(col) for col in CATEGORICAL_FEATURES]

    models: dict[str, object] = {}

    lgb = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=35,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.3,
        random_state=42,
        verbosity=-1,
    )
    lgb.fit(
        views["final_fit_sparse"],
        y_final_fit,
        eval_set=[(views["final_val_sparse"], y_final_val)],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(60, verbose=False)],
    )
    models["LightGBM"] = lgb

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=1500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.1,
        min_child_weight=2,
        gamma=0.1,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1,
        early_stopping_rounds=60,
    )
    xgb.fit(
        views["final_fit_sparse"],
        y_final_fit,
        eval_set=[(views["final_val_sparse"], y_final_val)],
        verbose=False,
    )
    models["XGBoost"] = xgb

    cat = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=6.0,
        random_seed=42,
        verbose=False,
    )
    cat.fit(
        Pool(views["final_fit_raw"], y_final_fit, cat_features=cat_indices),
        eval_set=Pool(views["final_val_raw"], y_final_val, cat_features=cat_indices),
        use_best_model=True,
        verbose=False,
    )
    models["CatBoost"] = (cat, cat_indices)
    return models


def predict_proba(model_obj: object, X_sparse: object, X_raw: pd.DataFrame) -> np.ndarray:
    if isinstance(model_obj, tuple):
        model, cat_indices = model_obj
        return model.predict_proba(Pool(X_raw, cat_features=cat_indices))
    return model_obj.predict_proba(X_sparse)


def build_meta_features(prob_map: dict[str, np.ndarray]) -> np.ndarray:
    return np.hstack([prob_map["LightGBM"], prob_map["XGBoost"], prob_map["CatBoost"]])


def metrics_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    return {
        "model": name,
        "test_accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def plot_metrics(metrics: pd.DataFrame, out_path: Path) -> None:
    ordered = metrics.sort_values(["test_accuracy", "macro_f1"], ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(ordered))
    colors = [COLORS[m] for m in ordered["model"]]

    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.barh(y, ordered["test_accuracy"], color=colors, alpha=0.92)
    ax.scatter(ordered["macro_f1"], y, s=78, color="white", edgecolor="#222222", linewidth=1.0, zorder=3)
    ax.axvline(1 / 3, linestyle="--", color="#BDBDBD", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["model"])
    ax.set_xlabel("Test Accuracy")
    ax.set_title("Stacking Meta Model vs Base Boosting Models")
    ax.grid(axis="x", alpha=0.18)
    ax.set_xlim(0.30, max(0.82, ordered["test_accuracy"].max() + 0.05))
    for idx, value in enumerate(ordered["test_accuracy"]):
        ax.text(value + 0.006, idx, f"{value:.3f}", va="center", fontsize=10, color="#333333")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=DISPLAY_GRADES,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Stacking Meta Model Confusion Matrix")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    y = df[TARGET_COL].map(GRADE_TO_INT).astype(int).to_numpy()
    splits = split_dataset(y)
    views = prepare_views(df, splits)

    y_meta_train = y[splits["meta_train"]]
    y_meta_val = y[splits["meta_val"]]
    y_final_fit = y[splits["final_fit"]]
    y_final_val = y[splits["final_val"]]
    y_test = y[splits["test"]]

    meta_models = fit_meta_generation_models(views, y_meta_train)
    meta_val_prob_map = {
        name: predict_proba(model_obj, views["meta_val_sparse"], views["meta_val_raw"])
        for name, model_obj in meta_models.items()
    }
    meta_X_val = build_meta_features(meta_val_prob_map)
    meta_model = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    meta_model.fit(meta_X_val, y_meta_val)

    final_models = fit_final_models(views, y_final_fit, y_final_val)
    test_prob_map = {
        name: predict_proba(model_obj, views["test_sparse"], views["test_raw"])
        for name, model_obj in final_models.items()
    }

    metrics_rows: list[dict[str, object]] = []
    predictions = pd.DataFrame(
        {
            "구": df.iloc[splits["test"]]["구"].to_numpy(),
            "동": df.iloc[splits["test"]]["동"].to_numpy(),
            "숙소명": df.iloc[splits["test"]]["숙소명"].to_numpy(),
            "업종": df.iloc[splits["test"]]["업종"].to_numpy(),
            "실제등급": df.iloc[splits["test"]][TARGET_COL].to_numpy(),
        }
    )

    for name in ["LightGBM", "XGBoost", "CatBoost"]:
        pred = test_prob_map[name].argmax(axis=1)
        metrics_rows.append(metrics_row(name, y_test, pred))
        predictions[name] = [GRADE_ORDER[int(x)] for x in pred]

    stacking_pred = meta_model.predict(build_meta_features(test_prob_map)).astype(int)
    metrics_rows.append(metrics_row("StackingMeta", y_test, stacking_pred))
    predictions["StackingMeta"] = [GRADE_ORDER[int(x)] for x in stacking_pred]

    coef_names = []
    for model_name in ["LightGBM", "XGBoost", "CatBoost"]:
        coef_names.extend([f"{model_name}_p{i}" for i in range(3)])
    meta_coef = pd.DataFrame(
        {
            "class_index": np.repeat(np.arange(3), len(coef_names)),
            "feature": coef_names * 3,
            "coefficient": meta_model.coef_.reshape(-1),
        }
    )

    metrics = pd.DataFrame(metrics_rows).sort_values(["test_accuracy", "macro_f1"], ascending=[False, False]).reset_index(drop=True)

    metrics_path = TABLE_DIR / "common_grade_stacking_meta_metrics.csv"
    pred_path = TABLE_DIR / "common_grade_stacking_meta_predictions.csv"
    coef_path = TABLE_DIR / "common_grade_stacking_meta_coefficients.csv"
    fig_path = FIG_DIR / "common_grade_stacking_meta_comparison.png"
    cm_path = FIG_DIR / "common_grade_stacking_meta_confusion.png"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
    meta_coef.to_csv(coef_path, index=False, encoding="utf-8-sig")
    plot_metrics(metrics, fig_path)
    plot_confusion(y_test, stacking_pred, cm_path)

    print(metrics.to_string(index=False))
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved predictions: {pred_path}")
    print(f"Saved meta coefficients: {coef_path}")
    print(f"Saved figure: {fig_path}")
    print(f"Saved figure: {cm_path}")


if __name__ == "__main__":
    main()
