from __future__ import annotations

import copy
import itertools
import random
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch_geometric.nn import GCNConv
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")

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
    "시설규모/연면적",
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
    "GNN": "#577590",
    "Boost3 Equal": "#A8DADC",
    "Boost3 Tuned": "#2A9D8F",
    "Boost4 Equal": "#CDB4DB",
    "Boost4 Tuned": "#6D597A",
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    df = df.dropna(subset=[TARGET_COL]).copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("미상").astype(str)
    df[TARGET_COL] = df[TARGET_COL].astype(str)
    return df.reset_index(drop=True)


def build_splits(y: np.ndarray) -> dict[str, np.ndarray]:
    all_idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    train_fit_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.20,
        random_state=42,
        stratify=y[train_idx],
    )
    return {
        "train": train_idx,
        "train_fit": train_fit_idx,
        "val": val_idx,
        "test": test_idx,
    }


def prepare_views(df: pd.DataFrame, splits: dict[str, np.ndarray]) -> dict[str, object]:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )

    X = df[FEATURE_COLUMNS].copy()
    X_train_fit_sparse = preprocessor.fit_transform(X.iloc[splits["train_fit"]])
    X_val_sparse = preprocessor.transform(X.iloc[splits["val"]])
    X_train_sparse = preprocessor.transform(X.iloc[splits["train"]])
    X_test_sparse = preprocessor.transform(X.iloc[splits["test"]])
    X_all_sparse = preprocessor.transform(X)

    dense_scaler = StandardScaler(with_mean=False)
    X_train_fit_dense = dense_scaler.fit_transform(X_train_fit_sparse).astype(np.float32)
    X_val_dense = dense_scaler.transform(X_val_sparse).astype(np.float32)
    X_train_dense = dense_scaler.transform(X_train_sparse).astype(np.float32)
    X_test_dense = dense_scaler.transform(X_test_sparse).astype(np.float32)
    X_all_dense = dense_scaler.transform(X_all_sparse).astype(np.float32)

    train_fit_raw = X.iloc[splits["train_fit"]].copy()
    val_raw = X.iloc[splits["val"]].copy()
    train_raw = X.iloc[splits["train"]].copy()
    test_raw = X.iloc[splits["test"]].copy()

    numeric_fill = train_fit_raw[NUMERIC_FEATURES].median()
    for frame in [train_fit_raw, val_raw, train_raw, test_raw]:
        frame.loc[:, NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].fillna(numeric_fill)
        frame.loc[:, CATEGORICAL_FEATURES] = frame[CATEGORICAL_FEATURES].fillna("미상").astype(str)

    return {
        "train_fit_sparse": X_train_fit_sparse,
        "val_sparse": X_val_sparse,
        "train_sparse": X_train_sparse,
        "test_sparse": X_test_sparse,
        "all_sparse": X_all_sparse,
        "train_fit_dense": X_train_fit_dense,
        "val_dense": X_val_dense,
        "train_dense": X_train_dense,
        "test_dense": X_test_dense,
        "all_dense": X_all_dense,
        "train_fit_raw": train_fit_raw,
        "val_raw": val_raw,
        "train_raw": train_raw,
        "test_raw": test_raw,
    }


def metrics_dict(name: str, train_true: np.ndarray, train_pred: np.ndarray, test_true: np.ndarray, test_pred: np.ndarray, notes: str = "") -> dict[str, object]:
    train_acc = accuracy_score(train_true, train_pred)
    test_acc = accuracy_score(test_true, test_pred)
    return {
        "model": name,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "accuracy_gap": train_acc - test_acc,
        "balanced_accuracy": balanced_accuracy_score(test_true, test_pred),
        "macro_f1": f1_score(test_true, test_pred, average="macro"),
        "weighted_f1": f1_score(test_true, test_pred, average="weighted"),
        "notes": notes,
    }


def fit_lightgbm(views: dict[str, object], y_train_fit: np.ndarray, y_val: np.ndarray) -> tuple[object, dict[str, np.ndarray], str]:
    model = LGBMClassifier(
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
    model.fit(
        views["train_fit_sparse"],
        y_train_fit,
        eval_set=[(views["val_sparse"], y_val)],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(60, verbose=False)],
    )
    probs = {
        "train": model.predict_proba(views["train_sparse"]),
        "val": model.predict_proba(views["val_sparse"]),
        "test": model.predict_proba(views["test_sparse"]),
    }
    return model, probs, f"best_iteration={getattr(model, 'best_iteration_', None)}"


def fit_xgboost(views: dict[str, object], y_train_fit: np.ndarray, y_val: np.ndarray) -> tuple[object, dict[str, np.ndarray], str]:
    model = XGBClassifier(
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
    model.fit(
        views["train_fit_sparse"],
        y_train_fit,
        eval_set=[(views["val_sparse"], y_val)],
        verbose=False,
    )
    probs = {
        "train": model.predict_proba(views["train_sparse"]),
        "val": model.predict_proba(views["val_sparse"]),
        "test": model.predict_proba(views["test_sparse"]),
    }
    return model, probs, f"best_iteration={getattr(model, 'best_iteration', None)}"


def fit_catboost(views: dict[str, object], y_train_fit: np.ndarray, y_val: np.ndarray) -> tuple[object, dict[str, np.ndarray], str]:
    cat_indices = [views["train_fit_raw"].columns.get_loc(col) for col in CATEGORICAL_FEATURES]
    train_pool = Pool(views["train_fit_raw"], y_train_fit, cat_features=cat_indices)
    val_pool = Pool(views["val_raw"], y_val, cat_features=cat_indices)
    train_outer_pool = Pool(views["train_raw"], cat_features=cat_indices)
    test_pool = Pool(views["test_raw"], cat_features=cat_indices)

    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=6.0,
        random_seed=42,
        verbose=False,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)
    probs = {
        "train": model.predict_proba(train_outer_pool),
        "val": model.predict_proba(val_pool),
        "test": model.predict_proba(test_pool),
    }
    return model, probs, f"best_iteration={model.get_best_iteration()}"


class GCNNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.fc = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(p=0.35)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.fc(x)


def build_edge_index(coords: np.ndarray, neighbors: int = 10) -> torch.Tensor:
    knn = NearestNeighbors(n_neighbors=neighbors + 1)
    knn.fit(coords)
    neighbor_idx = knn.kneighbors(coords, return_distance=False)
    edges: list[list[int]] = []
    for src, row in enumerate(neighbor_idx):
        for dst in row[1:]:
            edges.append([src, int(dst)])
            edges.append([int(dst), src])
    return torch.as_tensor(np.array(edges, dtype=np.int64).T, dtype=torch.long)


def fit_gnn(df: pd.DataFrame, splits: dict[str, np.ndarray], views: dict[str, object], y_all: np.ndarray) -> tuple[object, dict[str, np.ndarray], str]:
    device = torch.device("cpu")
    features = torch.as_tensor(views["all_dense"].toarray(), dtype=torch.float32, device=device)
    labels = torch.as_tensor(y_all, dtype=torch.long, device=device)

    coords = df[["위도", "경도"]].copy()
    coords.loc[:, ["위도", "경도"]] = coords[["위도", "경도"]].fillna(coords[["위도", "경도"]].median())
    coord_scaler = StandardScaler()
    coords_scaled = coord_scaler.fit_transform(coords.to_numpy())
    edge_index = build_edge_index(coords_scaled, neighbors=10).to(device)

    train_fit_mask = torch.zeros(len(df), dtype=torch.bool, device=device)
    val_mask = torch.zeros(len(df), dtype=torch.bool, device=device)
    train_mask = torch.zeros(len(df), dtype=torch.bool, device=device)
    test_mask = torch.zeros(len(df), dtype=torch.bool, device=device)
    train_fit_mask[splits["train_fit"]] = True
    val_mask[splits["val"]] = True
    train_mask[splits["train"]] = True
    test_mask[splits["test"]] = True

    model = GCNNet(input_dim=features.shape[1], hidden_dim=64, output_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_val = -1.0
    bad_epochs = 0
    patience = 30

    for _ in range(200):
        model.train()
        optimizer.zero_grad()
        logits = model(features, edge_index)
        loss = criterion(logits[train_fit_mask], labels[train_fit_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(features, edge_index)
            val_pred = logits[val_mask].argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_all[splits["val"]], val_pred)
        if val_acc > best_val + 1e-6:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(features, edge_index)
        probs_all = torch.softmax(logits, dim=1).cpu().numpy()

    probs = {
        "train": probs_all[splits["train"]],
        "val": probs_all[splits["val"]],
        "test": probs_all[splits["test"]],
    }
    return model, probs, "transductive_graph_k10"


def tune_weights(prob_maps: dict[str, dict[str, np.ndarray]], model_names: list[str], y_val: np.ndarray) -> tuple[dict[str, float], float, float]:
    best_weights: dict[str, float] | None = None
    best_macro = -1.0
    best_acc = -1.0
    grid = [0, 1, 2, 3, 4]

    for combo in itertools.product(grid, repeat=len(model_names)):
        if sum(combo) == 0:
            continue
        weights = np.array(combo, dtype=float)
        blended = sum(weights[i] * prob_maps[model_names[i]]["val"] for i in range(len(model_names))) / weights.sum()
        pred = blended.argmax(axis=1)
        macro = f1_score(y_val, pred, average="macro")
        acc = accuracy_score(y_val, pred)
        if macro > best_macro + 1e-9 or (abs(macro - best_macro) <= 1e-9 and acc > best_acc):
            best_macro = macro
            best_acc = acc
            best_weights = {model_names[i]: float(weights[i]) for i in range(len(model_names)) if weights[i] > 0}

    assert best_weights is not None
    return best_weights, best_macro, best_acc


def blend_probabilities(prob_maps: dict[str, dict[str, np.ndarray]], weights: dict[str, float], split: str) -> np.ndarray:
    total = sum(weights.values())
    blended = sum(weight * prob_maps[name][split] for name, weight in weights.items()) / total
    return blended


def equal_weights(model_names: list[str]) -> dict[str, float]:
    return {name: 1.0 for name in model_names}


def plot_metrics(metrics: pd.DataFrame, out_path: Path) -> None:
    ordered = metrics.sort_values("test_accuracy", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    colors = [COLORS[name] for name in ordered["model"]]

    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.barh(y, ordered["test_accuracy"], color=colors, alpha=0.9)
    ax.scatter(ordered["macro_f1"], y, s=85, color="white", edgecolor="#333333", linewidth=1.0, zorder=3, label="Macro F1")
    ax.axvline(1 / 3, color="#BDBDBD", linestyle="--", linewidth=1.3, label="Random baseline (1/3)")
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["model"])
    ax.set_xlabel("Test Accuracy")
    ax.set_title("Base Models vs Ensembles on Common 3-Class Target")
    ax.grid(axis="x", alpha=0.18)
    ax.set_xlim(0.25, max(0.82, ordered["test_accuracy"].max() + 0.05))

    for idx, row in ordered.iterrows():
        ax.text(row["test_accuracy"] + 0.006, idx, f"{row['test_accuracy']:.3f}", va="center", fontsize=10, color="#333333")

    ax.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gap(metrics: pd.DataFrame, out_path: Path) -> None:
    ordered = metrics.sort_values("test_accuracy", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    colors = [COLORS[name] for name in ordered["model"]]

    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    for idx, row in ordered.iterrows():
        ax.plot([row["test_accuracy"], row["train_accuracy"]], [idx, idx], color=colors[idx], linewidth=2.5, alpha=0.85)
        ax.scatter(row["train_accuracy"], idx, color=colors[idx], s=78, marker="o", edgecolor="white", linewidth=1.0)
        ax.scatter(row["test_accuracy"], idx, color=colors[idx], s=86, marker="s", edgecolor="white", linewidth=1.0)
        ax.text(row["train_accuracy"] + 0.005, idx + 0.14, f"gap {row['accuracy_gap']:.3f}", fontsize=9, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(ordered["model"])
    ax.set_xlabel("Accuracy")
    ax.set_title("Train-Test Accuracy Gap for Ensembles")
    ax.grid(axis="x", alpha=0.18)
    ax.set_xlim(0.65, min(1.02, ordered["train_accuracy"].max() + 0.04))
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="#555555", markersize=8, label="Train"),
        plt.Line2D([0], [0], marker="s", linestyle="", color="#555555", markersize=8, label="Test"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_best_confusion(y_test: np.ndarray, pred: np.ndarray, name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        pred,
        display_labels=DISPLAY_GRADES,
        cmap="YlGnBu",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Best Ensemble Confusion Matrix: {name}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed(42)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    y_all = df[TARGET_COL].map(GRADE_TO_INT).astype(int).to_numpy()
    splits = build_splits(y_all)
    views = prepare_views(df, splits)

    y_train_fit = y_all[splits["train_fit"]]
    y_train = y_all[splits["train"]]
    y_val = y_all[splits["val"]]
    y_test = y_all[splits["test"]]

    prob_maps: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    test_predictions = pd.DataFrame(
        {
            "구": df.iloc[splits["test"]]["구"].to_numpy(),
            "동": df.iloc[splits["test"]]["동"].to_numpy(),
            "숙소명": df.iloc[splits["test"]]["숙소명"].to_numpy(),
            "업종": df.iloc[splits["test"]]["업종"].to_numpy(),
            "실제등급": df.iloc[splits["test"]][TARGET_COL].to_numpy(),
        }
    )
    pred_store: dict[str, np.ndarray] = {}

    _, prob_maps["LightGBM"], lgb_notes = fit_lightgbm(views, y_train_fit, y_val)
    lgb_train = prob_maps["LightGBM"]["train"].argmax(axis=1)
    lgb_test = prob_maps["LightGBM"]["test"].argmax(axis=1)
    rows.append(metrics_dict("LightGBM", y_train, lgb_train, y_test, lgb_test, lgb_notes))
    pred_store["LightGBM"] = lgb_test
    test_predictions["LightGBM"] = [GRADE_ORDER[int(x)] for x in lgb_test]

    _, prob_maps["XGBoost"], xgb_notes = fit_xgboost(views, y_train_fit, y_val)
    xgb_train = prob_maps["XGBoost"]["train"].argmax(axis=1)
    xgb_test = prob_maps["XGBoost"]["test"].argmax(axis=1)
    rows.append(metrics_dict("XGBoost", y_train, xgb_train, y_test, xgb_test, xgb_notes))
    pred_store["XGBoost"] = xgb_test
    test_predictions["XGBoost"] = [GRADE_ORDER[int(x)] for x in xgb_test]

    _, prob_maps["CatBoost"], cat_notes = fit_catboost(views, y_train_fit, y_val)
    cat_train = prob_maps["CatBoost"]["train"].argmax(axis=1)
    cat_test = prob_maps["CatBoost"]["test"].argmax(axis=1)
    rows.append(metrics_dict("CatBoost", y_train, cat_train, y_test, cat_test, cat_notes))
    pred_store["CatBoost"] = cat_test
    test_predictions["CatBoost"] = [GRADE_ORDER[int(x)] for x in cat_test]

    _, prob_maps["GNN"], gnn_notes = fit_gnn(df, splits, views, y_all)
    gnn_train = prob_maps["GNN"]["train"].argmax(axis=1)
    gnn_test = prob_maps["GNN"]["test"].argmax(axis=1)
    rows.append(metrics_dict("GNN", y_train, gnn_train, y_test, gnn_test, gnn_notes))
    pred_store["GNN"] = gnn_test
    test_predictions["GNN"] = [GRADE_ORDER[int(x)] for x in gnn_test]

    boost3 = ["LightGBM", "XGBoost", "CatBoost"]
    boost4 = ["LightGBM", "XGBoost", "CatBoost", "GNN"]

    boost3_equal = equal_weights(boost3)
    boost3_equal_train = blend_probabilities(prob_maps, boost3_equal, "train").argmax(axis=1)
    boost3_equal_test = blend_probabilities(prob_maps, boost3_equal, "test").argmax(axis=1)
    rows.append(metrics_dict("Boost3 Equal", y_train, boost3_equal_train, y_test, boost3_equal_test, "equal soft voting"))
    pred_store["Boost3 Equal"] = boost3_equal_test
    test_predictions["Boost3 Equal"] = [GRADE_ORDER[int(x)] for x in boost3_equal_test]

    boost3_tuned, boost3_val_macro, boost3_val_acc = tune_weights(prob_maps, boost3, y_val)
    boost3_tuned_train = blend_probabilities(prob_maps, boost3_tuned, "train").argmax(axis=1)
    boost3_tuned_test = blend_probabilities(prob_maps, boost3_tuned, "test").argmax(axis=1)
    rows.append(
        metrics_dict(
            "Boost3 Tuned",
            y_train,
            boost3_tuned_train,
            y_test,
            boost3_tuned_test,
            f"weights={boost3_tuned}, val_macro_f1={boost3_val_macro:.4f}, val_acc={boost3_val_acc:.4f}",
        )
    )
    pred_store["Boost3 Tuned"] = boost3_tuned_test
    test_predictions["Boost3 Tuned"] = [GRADE_ORDER[int(x)] for x in boost3_tuned_test]

    boost4_equal = equal_weights(boost4)
    boost4_equal_train = blend_probabilities(prob_maps, boost4_equal, "train").argmax(axis=1)
    boost4_equal_test = blend_probabilities(prob_maps, boost4_equal, "test").argmax(axis=1)
    rows.append(metrics_dict("Boost4 Equal", y_train, boost4_equal_train, y_test, boost4_equal_test, "equal soft voting"))
    pred_store["Boost4 Equal"] = boost4_equal_test
    test_predictions["Boost4 Equal"] = [GRADE_ORDER[int(x)] for x in boost4_equal_test]

    boost4_tuned, boost4_val_macro, boost4_val_acc = tune_weights(prob_maps, boost4, y_val)
    boost4_tuned_train = blend_probabilities(prob_maps, boost4_tuned, "train").argmax(axis=1)
    boost4_tuned_test = blend_probabilities(prob_maps, boost4_tuned, "test").argmax(axis=1)
    rows.append(
        metrics_dict(
            "Boost4 Tuned",
            y_train,
            boost4_tuned_train,
            y_test,
            boost4_tuned_test,
            f"weights={boost4_tuned}, val_macro_f1={boost4_val_macro:.4f}, val_acc={boost4_val_acc:.4f}",
        )
    )
    pred_store["Boost4 Tuned"] = boost4_tuned_test
    test_predictions["Boost4 Tuned"] = [GRADE_ORDER[int(x)] for x in boost4_tuned_test]

    metrics = pd.DataFrame(rows).sort_values(["test_accuracy", "macro_f1"], ascending=[False, False]).reset_index(drop=True)
    metrics_path = TABLE_DIR / "common_grade_ensemble_metrics.csv"
    pred_path = TABLE_DIR / "common_grade_ensemble_test_predictions.csv"
    fig_path = FIG_DIR / "common_grade_ensemble_comparison.png"
    gap_path = FIG_DIR / "common_grade_ensemble_gap.png"
    cm_path = FIG_DIR / "common_grade_ensemble_best_confusion.png"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    test_predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
    plot_metrics(metrics, fig_path)
    plot_gap(metrics, gap_path)
    best_name = metrics.iloc[0]["model"]
    plot_best_confusion(y_test, pred_store[best_name], best_name, cm_path)

    print(metrics[["model", "train_accuracy", "test_accuracy", "macro_f1", "accuracy_gap"]].to_string(index=False))
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved predictions: {pred_path}")
    print(f"Saved figure: {fig_path}")
    print(f"Saved figure: {gap_path}")
    print(f"Saved figure: {cm_path}")


if __name__ == "__main__":
    main()
