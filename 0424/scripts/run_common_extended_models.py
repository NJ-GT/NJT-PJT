from __future__ import annotations

import copy
import json
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
from ngboost import NGBClassifier
from ngboost.distns import k_categorical
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "0424" / "분석" / "tables" / "분석변수_최종테이블0423_AHP3등급비교.csv"
OUTPUT_DIR = ROOT / "0424" / "분석"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"

TARGET_COL = "공통_3등급"
GRADE_ORDER = ["안전", "보통", "위험"]
GRADE_TO_INT = {grade: idx for idx, grade in enumerate(GRADE_ORDER)}
INT_TO_GRADE = {idx: grade for grade, idx in GRADE_TO_INT.items()}
DISPLAY_GRADES = ["Safe", "Moderate", "Risk"]

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
INFO_COLUMNS = ["구", "동", "숙소명", "업종", TARGET_COL]

MODEL_COLORS = {
    "XGBoost": "#E07A5F",
    "LightGBM": "#81B29A",
    "CatBoost": "#F2CC8F",
    "TabNet": "#3D405B",
    "NGBoost": "#6D597A",
    "GaussianProcess": "#277DA1",
    "IsolationForest": "#90BE6D",
    "Autoencoder": "#F94144",
    "GNN": "#577590",
}
MODEL_FAMILY = {
    "XGBoost": "Boosting",
    "LightGBM": "Boosting",
    "CatBoost": "Boosting",
    "TabNet": "Neural",
    "NGBoost": "Probabilistic",
    "GaussianProcess": "Kernel",
    "IsolationForest": "Anomaly",
    "Autoencoder": "Anomaly",
    "GNN": "Graph",
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


def prepare_feature_views(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
) -> dict[str, object]:
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
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                ),
            ),
        ]
    )
    sparse_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )

    X_df = df[FEATURE_COLUMNS].copy()
    X_train_fit_sparse = sparse_preprocessor.fit_transform(X_df.iloc[splits["train_fit"]])
    X_val_sparse = sparse_preprocessor.transform(X_df.iloc[splits["val"]])
    X_train_sparse = sparse_preprocessor.transform(X_df.iloc[splits["train"]])
    X_test_sparse = sparse_preprocessor.transform(X_df.iloc[splits["test"]])
    X_all_sparse = sparse_preprocessor.transform(X_df)

    svd_components = int(min(64, max(16, X_train_fit_sparse.shape[1] - 1)))
    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    dense_scaler = StandardScaler()

    X_train_fit_dense = dense_scaler.fit_transform(svd.fit_transform(X_train_fit_sparse))
    X_val_dense = dense_scaler.transform(svd.transform(X_val_sparse))
    X_train_dense = dense_scaler.transform(svd.transform(X_train_sparse))
    X_test_dense = dense_scaler.transform(svd.transform(X_test_sparse))
    X_all_dense = dense_scaler.transform(svd.transform(X_all_sparse))

    gp_components = int(min(24, X_train_fit_dense.shape[1]))
    X_train_fit_gp = X_train_fit_dense[:, :gp_components]
    X_val_gp = X_val_dense[:, :gp_components]
    X_train_gp = X_train_dense[:, :gp_components]
    X_test_gp = X_test_dense[:, :gp_components]

    train_fit_raw = X_df.iloc[splits["train_fit"]].copy()
    val_raw = X_df.iloc[splits["val"]].copy()
    train_raw = X_df.iloc[splits["train"]].copy()
    test_raw = X_df.iloc[splits["test"]].copy()

    numeric_fill = train_fit_raw[NUMERIC_FEATURES].median()
    for frame in [train_fit_raw, val_raw, train_raw, test_raw]:
        frame.loc[:, NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].fillna(numeric_fill)
        frame.loc[:, CATEGORICAL_FEATURES] = frame[CATEGORICAL_FEATURES].fillna("미상").astype(str)

    return {
        "sparse_preprocessor": sparse_preprocessor,
        "svd_components": svd_components,
        "gp_components": gp_components,
        "train_fit_sparse": X_train_fit_sparse,
        "val_sparse": X_val_sparse,
        "train_sparse": X_train_sparse,
        "test_sparse": X_test_sparse,
        "all_sparse": X_all_sparse,
        "train_fit_dense": X_train_fit_dense.astype(np.float32),
        "val_dense": X_val_dense.astype(np.float32),
        "train_dense": X_train_dense.astype(np.float32),
        "test_dense": X_test_dense.astype(np.float32),
        "all_dense": X_all_dense.astype(np.float32),
        "train_fit_gp": X_train_fit_gp.astype(np.float32),
        "val_gp": X_val_gp.astype(np.float32),
        "train_gp": X_train_gp.astype(np.float32),
        "test_gp": X_test_gp.astype(np.float32),
        "train_fit_raw": train_fit_raw,
        "val_raw": val_raw,
        "train_raw": train_raw,
        "test_raw": test_raw,
    }


def build_test_prediction_frame(df: pd.DataFrame, test_idx: np.ndarray) -> pd.DataFrame:
    result = df.iloc[test_idx][INFO_COLUMNS].copy()
    result = result.rename(columns={TARGET_COL: "실제등급"})
    return result.reset_index(drop=True)


def record_result(
    rows: list[dict[str, object]],
    predictions: pd.DataFrame,
    model_name: str,
    train_pred: np.ndarray,
    train_true: np.ndarray,
    test_pred: np.ndarray,
    test_true: np.ndarray,
    representation: str,
    notes: str = "",
) -> None:
    train_pred = np.asarray(train_pred, dtype=int)
    train_true = np.asarray(train_true, dtype=int)
    test_pred = np.asarray(test_pred, dtype=int)
    test_true = np.asarray(test_true, dtype=int)
    rows.append(
        {
            "model": model_name,
            "family": MODEL_FAMILY[model_name],
            "representation": representation,
            "train_accuracy": accuracy_score(train_true, train_pred),
            "test_accuracy": accuracy_score(test_true, test_pred),
            "accuracy_gap": accuracy_score(train_true, train_pred)
            - accuracy_score(test_true, test_pred),
            "balanced_accuracy": balanced_accuracy_score(test_true, test_pred),
            "macro_f1": f1_score(test_true, test_pred, average="macro"),
            "weighted_f1": f1_score(test_true, test_pred, average="weighted"),
            "notes": notes,
        }
    )
    predictions[model_name] = [INT_TO_GRADE[int(x)] for x in test_pred]


def map_scores_to_three_classes(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, list[float]]:
    thresholds = np.quantile(train_scores, [1 / 3, 2 / 3]).tolist()

    def to_bins(values: np.ndarray) -> np.ndarray:
        return np.digitize(values, thresholds, right=False).astype(int)

    direct_train = to_bins(train_scores)
    reversed_train = 2 - direct_train
    direct_f1 = f1_score(y_train, direct_train, average="macro")
    reversed_f1 = f1_score(y_train, reversed_train, average="macro")
    orientation = "high_score_to_high_risk"
    if reversed_f1 > direct_f1:
        orientation = "high_score_to_low_risk"
        direct_train = reversed_train
        test_pred = 2 - to_bins(test_scores)
    else:
        test_pred = to_bins(test_scores)
    return direct_train, test_pred.astype(int), orientation, thresholds


def run_xgboost(
    views: dict[str, object],
    y_train_fit: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(GRADE_ORDER),
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
    train_pred = model.predict(views["train_sparse"])
    test_pred = model.predict(views["test_sparse"])
    notes = f"best_iteration={getattr(model, 'best_iteration', None)}"
    return train_pred, test_pred, notes


def run_lightgbm(
    views: dict[str, object],
    y_train_fit: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(GRADE_ORDER),
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
    train_pred = model.predict(views["train_sparse"])
    test_pred = model.predict(views["test_sparse"])
    notes = f"best_iteration={getattr(model, 'best_iteration_', None)}"
    return train_pred, test_pred, notes


def run_catboost(
    views: dict[str, object],
    y_train_fit: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
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
    train_pred = model.predict(train_outer_pool).reshape(-1).astype(int)
    test_pred = model.predict(test_pool).reshape(-1).astype(int)
    notes = f"best_iteration={model.get_best_iteration()}"
    return train_pred, test_pred, notes


def run_tabnet(
    views: dict[str, object],
    y_train_fit: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    model = TabNetClassifier(
        seed=42,
        n_d=24,
        n_a=24,
        n_steps=4,
        gamma=1.3,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 30, "gamma": 0.8},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="sparsemax",
        verbose=0,
    )
    model.fit(
        X_train=views["train_fit_dense"],
        y_train=y_train_fit,
        eval_set=[(views["val_dense"], y_val)],
        eval_name=["val"],
        eval_metric=["accuracy"],
        max_epochs=200,
        patience=25,
        batch_size=512,
        virtual_batch_size=128,
        num_workers=0,
        weights=0,
        drop_last=False,
    )
    train_pred = model.predict(views["train_dense"]).astype(int)
    test_pred = model.predict(views["test_dense"]).astype(int)
    notes = f"best_epoch={getattr(model, 'best_epoch', None)}"
    return train_pred, test_pred, notes


def run_ngboost(
    views: dict[str, object],
    y_train_fit: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    base = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42)
    model = NGBClassifier(
        Dist=k_categorical(len(GRADE_ORDER)),
        Base=base,
        n_estimators=220,
        learning_rate=0.03,
        verbose=False,
        verbose_eval=0,
        random_state=42,
    )
    model.fit(views["train_fit_dense"], y_train_fit)
    train_pred = model.predict(views["train_dense"]).astype(int)
    test_pred = model.predict(views["test_dense"]).astype(int)
    notes = "multiclass_ngboost"
    return train_pred, test_pred, notes


def run_gaussian_process(
    views: dict[str, object],
    y_train_fit: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    train_sample_size = min(900, len(y_train_fit))
    sampler = StratifiedShuffleSplit(n_splits=1, train_size=train_sample_size, random_state=42)
    sampled_idx, _ = next(sampler.split(views["train_fit_gp"], y_train_fit))
    X_train_sample = views["train_fit_gp"][sampled_idx]
    y_train_sample = y_train_fit[sampled_idx]
    kernel = ConstantKernel(1.0, (0.5, 2.0)) * RBF(length_scale=1.0)
    model = GaussianProcessClassifier(
        kernel=kernel,
        random_state=42,
        multi_class="one_vs_rest",
        n_restarts_optimizer=0,
        max_iter_predict=80,
    )
    model.fit(X_train_sample, y_train_sample)
    train_pred = model.predict(views["train_gp"]).astype(int)
    test_pred = model.predict(views["test_gp"]).astype(int)
    notes = f"fit_rows={train_sample_size}, gp_dims={views['gp_components']}"
    return train_pred, test_pred, notes


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 24),
            nn.ReLU(),
            nn.Linear(24, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 24),
            nn.ReLU(),
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def compute_reconstruction_error(model: nn.Module, data: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(data, dtype=torch.float32)
        recon = model(tensor)
        errors = torch.mean((tensor - recon) ** 2, dim=1)
    return errors.cpu().numpy()


def run_autoencoder(
    views: dict[str, object],
    y_train_fit: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    device = torch.device("cpu")
    model = AutoEncoder(views["train_fit_dense"].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_tensor = torch.as_tensor(views["train_fit_dense"], dtype=torch.float32)
    val_tensor = torch.as_tensor(views["val_dense"], dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=256, shuffle=True)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience = 15
    bad_epochs = 0
    max_epochs = 120

    for _ in range(max_epochs):
        model.train()
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_recon = model(val_tensor.to(device))
            val_loss = criterion(val_recon, val_tensor.to(device)).item()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_scores = compute_reconstruction_error(model, views["train_dense"])
    test_scores = compute_reconstruction_error(model, views["test_dense"])
    train_pred, test_pred, orientation, thresholds = map_scores_to_three_classes(train_scores, test_scores, y_train)
    notes = f"{orientation}, thresholds={[round(x, 5) for x in thresholds]}"
    return train_pred, test_pred, notes


def run_isolation_forest(
    views: dict[str, object],
    y_train_fit: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    model = IsolationForest(
        n_estimators=700,
        contamination="auto",
        random_state=42,
        n_jobs=1,
    )
    model.fit(views["train_fit_dense"])
    train_scores = -model.score_samples(views["train_dense"])
    test_scores = -model.score_samples(views["test_dense"])
    train_pred, test_pred, orientation, thresholds = map_scores_to_three_classes(train_scores, test_scores, y_train)
    notes = f"{orientation}, thresholds={[round(x, 5) for x in thresholds]}"
    return train_pred, test_pred, notes


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
    nn_idx = knn.kneighbors(coords, return_distance=False)
    edges: list[list[int]] = []
    for src, row in enumerate(nn_idx):
        for dst in row[1:]:
            edges.append([src, int(dst)])
            edges.append([int(dst), src])
    edge_array = np.array(edges, dtype=np.int64).T
    return torch.as_tensor(edge_array, dtype=torch.long)


def run_gnn(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    views: dict[str, object],
    y_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    device = torch.device("cpu")
    features = torch.as_tensor(views["all_dense"], dtype=torch.float32, device=device)
    labels = torch.as_tensor(y_all, dtype=torch.long, device=device)
    coords = df[["위도", "경도"]].copy()
    coords.loc[:, ["위도", "경도"]] = coords[["위도", "경도"]].fillna(coords[["위도", "경도"]].median())
    coord_scaler = StandardScaler()
    coords_scaled = coord_scaler.fit_transform(coords.to_numpy())
    edge_index = build_edge_index(coords_scaled, neighbors=10).to(device)

    train_mask = torch.zeros(len(df), dtype=torch.bool, device=device)
    val_mask = torch.zeros(len(df), dtype=torch.bool, device=device)
    test_mask = torch.zeros(len(df), dtype=torch.bool, device=device)
    train_mask[splits["train_fit"]] = True
    val_mask[splits["val"]] = True
    test_mask[splits["test"]] = True

    model = GCNNet(features.shape[1], hidden_dim=64, output_dim=len(GRADE_ORDER)).to(device)
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
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(features, edge_index)
            val_pred = logits[val_mask].argmax(dim=1)
            val_acc = accuracy_score(labels[val_mask].cpu().numpy(), val_pred.cpu().numpy())
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
        train_pred = logits[splits["train"]].argmax(dim=1).cpu().numpy()
        test_pred = logits[splits["test"]].argmax(dim=1).cpu().numpy()
    notes = "k_neighbors=10, transductive_graph"
    return train_pred, test_pred, notes


def plot_performance(metrics: pd.DataFrame, out_path: Path) -> None:
    sorted_df = metrics.sort_values(["test_accuracy", "macro_f1"], ascending=[True, True]).reset_index(drop=True)
    y_pos = np.arange(len(sorted_df))
    colors = [MODEL_COLORS[name] for name in sorted_df["model"]]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.barh(y_pos, sorted_df["test_accuracy"], color=colors, alpha=0.92, edgecolor="none")
    ax.scatter(
        sorted_df["macro_f1"],
        y_pos,
        s=95,
        color="white",
        edgecolor="#222222",
        linewidth=1.0,
        label="Macro F1",
        zorder=3,
    )
    ax.axvline(1 / 3, color="#BDBDBD", linestyle="--", linewidth=1.3, label="Random baseline (1/3)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df["model"])
    ax.set_xlabel("Test Accuracy")
    ax.set_title("Extended Model Comparison on Common 3-Class Target")
    ax.set_xlim(0, max(0.82, sorted_df["test_accuracy"].max() + 0.05))
    ax.grid(axis="x", alpha=0.18)

    for idx, value in enumerate(sorted_df["test_accuracy"]):
        ax.text(value + 0.006, idx, f"{value:.3f}", va="center", fontsize=10, color="#333333")

    family_handles = []
    seen_family: set[str] = set()
    for model in sorted_df["model"]:
        family = MODEL_FAMILY[model]
        if family in seen_family:
            continue
        seen_family.add(family)
        family_handles.append(
            plt.Line2D([0], [0], marker="s", linestyle="", color=MODEL_COLORS[model], markersize=10, label=family)
        )
    metric_handle = plt.Line2D(
        [0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="#222222", markersize=8, label="Macro F1"
    )
    baseline_handle = plt.Line2D([0], [0], color="#BDBDBD", linestyle="--", linewidth=1.3, label="Random baseline (1/3)")
    ax.legend(handles=family_handles + [metric_handle, baseline_handle], loc="lower right", frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gap(metrics: pd.DataFrame, out_path: Path) -> None:
    sorted_df = metrics.sort_values("test_accuracy", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(sorted_df))
    colors = [MODEL_COLORS[name] for name in sorted_df["model"]]

    fig, ax = plt.subplots(figsize=(14, 8))
    for idx, row in sorted_df.iterrows():
        ax.plot(
            [row["test_accuracy"], row["train_accuracy"]],
            [idx, idx],
            color=colors[idx],
            linewidth=2.4,
            alpha=0.85,
        )
        ax.scatter(row["train_accuracy"], idx, color=colors[idx], s=80, marker="o", edgecolor="white", linewidth=1.0)
        ax.scatter(row["test_accuracy"], idx, color=colors[idx], s=90, marker="s", edgecolor="white", linewidth=1.0)
        ax.text(
            row["train_accuracy"] + 0.006,
            idx + 0.15,
            f"gap {row['accuracy_gap']:.3f}",
            fontsize=9,
            color="#333333",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df["model"])
    ax.set_xlabel("Accuracy")
    ax.set_title("Train-Test Accuracy Gap")
    ax.grid(axis="x", alpha=0.18)
    ax.set_xlim(0.25, min(1.02, sorted_df["train_accuracy"].max() + 0.04))

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="#555555", markersize=8, label="Train"),
        plt.Line2D([0], [0], marker="s", linestyle="", color="#555555", markersize=8, label="Test"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_best_confusion(
    y_test: np.ndarray,
    pred: np.ndarray,
    model_name: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        pred,
        display_labels=DISPLAY_GRADES,
        cmap="YlOrBr",
        colorbar=False,
        ax=ax,
    )
    disp.ax_.set_title(f"Best Model Confusion Matrix: {model_name}")
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
    views = prepare_feature_views(df, splits)

    y_train_fit = y_all[splits["train_fit"]]
    y_train = y_all[splits["train"]]
    y_val = y_all[splits["val"]]
    y_test = y_all[splits["test"]]

    metrics_rows: list[dict[str, object]] = []
    predictions = build_test_prediction_frame(df, splits["test"])
    test_pred_store: dict[str, np.ndarray] = {}

    xgb_train_pred, xgb_test_pred, xgb_notes = run_xgboost(views, y_train_fit, y_train, y_val, y_test)
    record_result(metrics_rows, predictions, "XGBoost", xgb_train_pred, y_train, xgb_test_pred, y_test, "sparse_onehot", xgb_notes)
    test_pred_store["XGBoost"] = xgb_test_pred

    lgb_train_pred, lgb_test_pred, lgb_notes = run_lightgbm(views, y_train_fit, y_train, y_val, y_test)
    record_result(metrics_rows, predictions, "LightGBM", lgb_train_pred, y_train, lgb_test_pred, y_test, "sparse_onehot", lgb_notes)
    test_pred_store["LightGBM"] = lgb_test_pred

    cat_train_pred, cat_test_pred, cat_notes = run_catboost(views, y_train_fit, y_train, y_val, y_test)
    record_result(metrics_rows, predictions, "CatBoost", cat_train_pred, y_train, cat_test_pred, y_test, "raw_categorical", cat_notes)
    test_pred_store["CatBoost"] = cat_test_pred

    tab_train_pred, tab_test_pred, tab_notes = run_tabnet(views, y_train_fit, y_train, y_val, y_test)
    record_result(metrics_rows, predictions, "TabNet", tab_train_pred, y_train, tab_test_pred, y_test, "svd_dense", tab_notes)
    test_pred_store["TabNet"] = tab_test_pred

    ngb_train_pred, ngb_test_pred, ngb_notes = run_ngboost(views, y_train_fit, y_train, y_val, y_test)
    record_result(metrics_rows, predictions, "NGBoost", ngb_train_pred, y_train, ngb_test_pred, y_test, "svd_dense", ngb_notes)
    test_pred_store["NGBoost"] = ngb_test_pred

    gp_train_pred, gp_test_pred, gp_notes = run_gaussian_process(views, y_train_fit, y_train, y_val, y_test)
    record_result(metrics_rows, predictions, "GaussianProcess", gp_train_pred, y_train, gp_test_pred, y_test, "svd24_dense", gp_notes)
    test_pred_store["GaussianProcess"] = gp_test_pred

    iso_train_pred, iso_test_pred, iso_notes = run_isolation_forest(views, y_train_fit, y_train, y_val, y_test)
    record_result(metrics_rows, predictions, "IsolationForest", iso_train_pred, y_train, iso_test_pred, y_test, "svd_dense", iso_notes)
    test_pred_store["IsolationForest"] = iso_test_pred

    ae_train_pred, ae_test_pred, ae_notes = run_autoencoder(views, y_train_fit, y_train, y_val, y_test)
    record_result(metrics_rows, predictions, "Autoencoder", ae_train_pred, y_train, ae_test_pred, y_test, "svd_dense", ae_notes)
    test_pred_store["Autoencoder"] = ae_test_pred

    gnn_train_pred, gnn_test_pred, gnn_notes = run_gnn(df, splits, views, y_all)
    record_result(metrics_rows, predictions, "GNN", gnn_train_pred, y_train, gnn_test_pred, y_test, "graph+svd_dense", gnn_notes)
    test_pred_store["GNN"] = gnn_test_pred

    metrics = pd.DataFrame(metrics_rows)
    metrics = metrics.sort_values(["test_accuracy", "macro_f1"], ascending=[False, False]).reset_index(drop=True)

    metrics_path = TABLE_DIR / "common_grade_extended_model_metrics.csv"
    predictions_path = TABLE_DIR / "common_grade_extended_model_test_predictions.csv"
    meta_path = TABLE_DIR / "common_grade_extended_model_metadata.json"
    performance_png = FIG_DIR / "common_grade_extended_model_comparison.png"
    gap_png = FIG_DIR / "common_grade_extended_model_gap.png"
    confusion_png = FIG_DIR / "common_grade_extended_model_best_confusion.png"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    metadata = {
        "target": TARGET_COL,
        "rows_total": int(len(df)),
        "rows_train_outer": int(len(splits["train"])),
        "rows_train_fit": int(len(splits["train_fit"])),
        "rows_val": int(len(splits["val"])),
        "rows_test": int(len(splits["test"])),
        "svd_components": int(views["svd_components"]),
        "gp_components": int(views["gp_components"]),
        "gnn_neighbors": 10,
        "grade_distribution_total": df[TARGET_COL].value_counts().to_dict(),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_performance(metrics, performance_png)
    plot_gap(metrics, gap_png)

    best_model = metrics.iloc[0]["model"]
    plot_best_confusion(y_test, test_pred_store[best_model], best_model, confusion_png)

    print(metrics[["model", "test_accuracy", "macro_f1", "accuracy_gap"]].to_string(index=False))
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved figure: {performance_png}")
    print(f"Saved figure: {gap_png}")
    print(f"Saved figure: {confusion_png}")


if __name__ == "__main__":
    main()
