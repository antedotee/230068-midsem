from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification, make_moons
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 30
ROOT = Path(__file__).resolve().parents[1]
PARTB_DIR = ROOT / "partB"
DATA_DIR = PARTB_DIR / "data"
RESULTS_DIR = PARTB_DIR / "results"


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _write_dataset_csv(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_1", "feature_2", "label"])
        for row, label in zip(X, y):
            writer.writerow([float(row[0]), float(row[1]), int(label)])


def _read_dataset_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    X: list[list[float]] = []
    y: list[int] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            X.append([float(row["feature_1"]), float(row["feature_2"])])
            y.append(int(row["label"]))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def create_datasets(seed: int = SEED) -> None:
    ensure_directories()
    X_main, y_main = make_moons(n_samples=360, noise=0.27, random_state=seed)
    X_failure, y_failure = make_classification(
        n_samples=320,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=2.6,
        flip_y=0.0,
        random_state=seed,
    )
    _write_dataset_csv(DATA_DIR / "main_moons_dataset.csv", X_main, y_main)
    _write_dataset_csv(DATA_DIR / "failure_linear_dataset.csv", X_failure, y_failure)


def load_main_dataset() -> tuple[np.ndarray, np.ndarray]:
    return _read_dataset_csv(DATA_DIR / "main_moons_dataset.csv")


def load_failure_dataset() -> tuple[np.ndarray, np.ndarray]:
    return _read_dataset_csv(DATA_DIR / "failure_linear_dataset.csv")


def _to_signed(y: np.ndarray) -> np.ndarray:
    return np.where(y > 0, 1.0, -1.0)


@dataclass
class ModelMetrics:
    name: str
    train_accuracy: float
    test_accuracy: float
    train_error: float
    test_error: float
    mean_pairwise_alignment: float


class SimpleDRMClassifier:
    """A simplified DRM-style ensemble trained in an explicit feature map."""

    def __init__(
        self,
        n_learners: int = 7,
        mu: float = 0.06,
        lr: float = 0.03,
        epochs: int = 750,
        l2: float = 1e-3,
        gamma: float = 1.3,
        n_components: int = 280,
        use_rff: bool = True,
        random_state: int = SEED,
    ) -> None:
        self.n_learners = n_learners
        self.mu = mu
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.gamma = gamma
        self.n_components = n_components
        self.use_rff = use_rff
        self.random_state = random_state

    def _fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        if self.use_rff:
            self.mapper_ = RBFSampler(
                gamma=self.gamma,
                n_components=self.n_components,
                random_state=self.random_state,
            )
            return self.mapper_.fit_transform(X_scaled)
        self.mapper_ = None
        return X_scaled

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_.transform(X)
        if self.mapper_ is None:
            return X_scaled
        return self.mapper_.transform(X_scaled)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleDRMClassifier":
        Z = self._fit_transform(X)
        y_signed = _to_signed(y)
        n_samples, n_features = Z.shape
        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(0.0, 0.08, size=(self.n_learners, n_features))
        self.bias_ = np.zeros(self.n_learners, dtype=float)
        self.loss_history_: list[float] = []

        for epoch in range(self.epochs):
            scores = Z @ self.weights_.T + self.bias_
            margins = y_signed[:, None] * scores
            slack = np.maximum(0.0, 1.0 - margins)
            grad_scores = (-2.0 / n_samples) * y_signed[:, None] * slack
            grad_w = grad_scores.T @ Z + 2.0 * self.l2 * self.weights_
            grad_b = grad_scores.sum(axis=0)

            # Stable diversity surrogate: discourage aligned learners by
            # penalizing pairwise squared inner products after unit-ball projection.
            if self.n_learners > 1 and self.mu > 0:
                norms = np.linalg.norm(self.weights_, axis=1, keepdims=True)
                normalized = self.weights_ / np.clip(norms, 1e-8, None)
                gram = normalized @ normalized.T
                np.fill_diagonal(gram, 0.0)
                grad_w += 2.0 * self.mu * (gram @ normalized)

            self.weights_ -= self.lr * grad_w
            self.bias_ -= self.lr * grad_b

            norms = np.linalg.norm(self.weights_, axis=1, keepdims=True)
            self.weights_ = self.weights_ / np.maximum(1.0, norms)

            if epoch % 20 == 0 or epoch == self.epochs - 1:
                objective = float((slack**2).mean() + self.l2 * (self.weights_**2).sum())
                if self.n_learners > 1:
                    normalized = self.weights_ / np.clip(
                        np.linalg.norm(self.weights_, axis=1, keepdims=True), 1e-8, None
                    )
                    gram = normalized @ normalized.T
                    diversity_term = float(
                        self.mu * np.triu(gram**2, k=1).sum()
                    )
                    objective += diversity_term
                self.loss_history_.append(objective)

        return self

    def component_scores(self, X: np.ndarray) -> np.ndarray:
        Z = self._transform(X)
        return Z @ self.weights_.T + self.bias_

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.component_scores(X).mean(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    def test_error(self, X: np.ndarray, y: np.ndarray) -> float:
        return 1.0 - self.score(X, y)

    def mean_pairwise_alignment(self) -> float:
        if self.n_learners < 2:
            return 1.0
        norms = np.linalg.norm(self.weights_, axis=1, keepdims=True)
        normalized = self.weights_ / np.clip(norms, 1e-8, None)
        gram = normalized @ normalized.T
        upper = gram[np.triu_indices(self.n_learners, k=1)]
        return float(np.mean(upper))


def split_dataset(
    X: np.ndarray, y: np.ndarray, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        X, y, test_size=0.35, stratify=y, random_state=seed
    )


def fit_rbf_svm(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=2.0, gamma=1.3, random_state=SEED),
    )
    model.fit(X_train, y_train)
    return model


def collect_metrics(name: str, model: Any, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    alignment = getattr(model, "mean_pairwise_alignment", lambda: math.nan)()
    return ModelMetrics(
        name=name,
        train_accuracy=float(train_acc),
        test_accuracy=float(test_acc),
        train_error=float(1.0 - train_acc),
        test_error=float(1.0 - test_acc),
        mean_pairwise_alignment=float(alignment),
    )


def run_main_experiment(seed: int = SEED) -> dict[str, Any]:
    set_seed(seed)
    X, y = load_main_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y, seed)

    drm = SimpleDRMClassifier(
        n_learners=7,
        mu=0.06,
        lr=0.03,
        epochs=750,
        l2=8e-4,
        gamma=1.3,
        n_components=280,
        use_rff=True,
        random_state=seed,
    ).fit(X_train, y_train)

    no_div = SimpleDRMClassifier(
        n_learners=7,
        mu=0.0,
        lr=0.03,
        epochs=750,
        l2=8e-4,
        gamma=1.3,
        n_components=280,
        use_rff=True,
        random_state=seed,
    ).fit(X_train, y_train)

    svm = fit_rbf_svm(X_train, y_train)

    metrics = [
        collect_metrics("Simplified DRM", drm, X_train, X_test, y_train, y_test),
        collect_metrics("No-diversity ensemble", no_div, X_train, X_test, y_train, y_test),
        collect_metrics("RBF SVM baseline", svm, X_train, X_test, y_train, y_test),
    ]

    return {
        "dataset": (X, y),
        "split": (X_train, X_test, y_train, y_test),
        "models": {"drm": drm, "no_div": no_div, "svm": svm},
        "metrics": metrics,
    }


def run_ablation_experiment(seed: int = SEED) -> dict[str, Any]:
    set_seed(seed)
    X, y = load_main_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y, seed)

    full_model = SimpleDRMClassifier(
        n_learners=7,
        mu=0.06,
        lr=0.03,
        epochs=750,
        l2=8e-4,
        gamma=1.3,
        n_components=280,
        use_rff=True,
        random_state=seed,
    ).fit(X_train, y_train)

    ablation_div = SimpleDRMClassifier(
        n_learners=7,
        mu=0.0,
        lr=0.03,
        epochs=750,
        l2=8e-4,
        gamma=1.3,
        n_components=280,
        use_rff=True,
        random_state=seed,
    ).fit(X_train, y_train)

    ablation_feature = SimpleDRMClassifier(
        n_learners=7,
        mu=0.06,
        lr=0.04,
        epochs=750,
        l2=8e-4,
        gamma=1.3,
        n_components=280,
        use_rff=False,
        random_state=seed,
    ).fit(X_train, y_train)

    metrics = [
        collect_metrics("Full method", full_model, X_train, X_test, y_train, y_test),
        collect_metrics("Ablation 1: mu = 0", ablation_div, X_train, X_test, y_train, y_test),
        collect_metrics(
            "Ablation 2: no RFF map",
            ablation_feature,
            X_train,
            X_test,
            y_train,
            y_test,
        ),
    ]

    return {
        "dataset": (X, y),
        "split": (X_train, X_test, y_train, y_test),
        "models": {
            "full": full_model,
            "ablation_div": ablation_div,
            "ablation_feature": ablation_feature,
        },
        "metrics": metrics,
    }


def run_failure_experiment(seed: int = SEED) -> dict[str, Any]:
    set_seed(seed)
    X, y = load_failure_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y, seed)
    mu_values = [0.0, 0.02, 0.05, 0.10, 0.18, 0.30]
    records: list[dict[str, float]] = []

    for mu in mu_values:
        model = SimpleDRMClassifier(
            n_learners=7,
            mu=mu,
            lr=0.03,
            epochs=750,
            l2=8e-4,
            gamma=0.9,
            n_components=160,
            use_rff=False,
            random_state=seed,
        ).fit(X_train, y_train)
        records.append(
            {
                "mu": float(mu),
                "train_error": float(model.test_error(X_train, y_train)),
                "test_error": float(model.test_error(X_test, y_test)),
                "alignment": float(model.mean_pairwise_alignment()),
            }
        )

    best_linear = SimpleDRMClassifier(
        n_learners=7,
        mu=0.0,
        lr=0.03,
        epochs=750,
        l2=8e-4,
        gamma=0.9,
        n_components=160,
        use_rff=False,
        random_state=seed,
    ).fit(X_train, y_train)

    return {
        "dataset": (X, y),
        "split": (X_train, X_test, y_train, y_test),
        "records": records,
        "reference_model": best_linear,
    }


def metrics_table(metrics: list[ModelMetrics]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for item in metrics:
        rows.append(
            {
                "model": item.name,
                "train_accuracy": round(item.train_accuracy, 4),
                "test_accuracy": round(item.test_accuracy, 4),
                "train_error": round(item.train_error, 4),
                "test_error": round(item.test_error, 4),
                "mean_pairwise_alignment": round(item.mean_pairwise_alignment, 4)
                if not math.isnan(item.mean_pairwise_alignment)
                else "NA",
            }
        )
    return rows


def _mesh_grid(X: np.ndarray, padding: float = 0.5, points: int = 220) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, points), np.linspace(y_min, y_max, points))
    return xx, yy


def plot_model_boundary(ax: plt.Axes, model: Any, X: np.ndarray, y: np.ndarray, title: str) -> None:
    xx, yy = _mesh_grid(X)
    grid = np.c_[xx.ravel(), yy.ravel()]
    if hasattr(model, "decision_function"):
        zz = model.decision_function(grid)
    else:
        zz = model.predict(grid)
    zz = np.asarray(zz).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=20, cmap="coolwarm", alpha=0.35)
    ax.contour(xx, yy, zz, levels=[0], colors="black", linewidths=1.2)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=18, edgecolors="black", linewidth=0.2)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


def save_main_plots(result: dict[str, Any]) -> dict[str, Path]:
    ensure_directories()
    metrics = result["metrics"]
    metric_path = RESULTS_DIR / "q2_metric_comparison.png"
    boundary_path = RESULTS_DIR / "q2_decision_boundaries.png"

    fig, ax = plt.subplots(figsize=(8, 4.8))
    names = [m.name for m in metrics]
    errors = [m.test_error for m in metrics]
    bars = ax.bar(names, errors, color=["#2563eb", "#f59e0b", "#10b981"])
    ax.set_ylabel("Test error (1 - accuracy)")
    ax.set_title("Question 2: test error comparison")
    ax.set_ylim(0, max(errors) + 0.08)
    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, err + 0.01, f"{err:.3f}", ha="center")
    fig.autofmt_xdate(rotation=10)
    fig.tight_layout()
    fig.savefig(metric_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    X, y = result["dataset"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    plot_model_boundary(axes[0], result["models"]["drm"], X, y, "Simplified DRM")
    plot_model_boundary(axes[1], result["models"]["no_div"], X, y, "No-diversity ensemble")
    plot_model_boundary(axes[2], result["models"]["svm"], X, y, "RBF SVM baseline")
    fig.suptitle("Question 2: decision boundary comparison", fontsize=12)
    fig.tight_layout()
    fig.savefig(boundary_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {"metric_plot": metric_path, "boundary_plot": boundary_path}


def save_ablation_plot(result: dict[str, Any]) -> Path:
    ensure_directories()
    path = RESULTS_DIR / "q3_ablation_comparison.png"
    metrics = result["metrics"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    names = [m.name for m in metrics]
    errors = [m.test_error for m in metrics]
    bars = ax.bar(names, errors, color=["#2563eb", "#ef4444", "#8b5cf6"])
    ax.set_ylabel("Test error (1 - accuracy)")
    ax.set_title("Question 3.1: ablation comparison")
    ax.set_ylim(0, max(errors) + 0.08)
    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, err + 0.01, f"{err:.3f}", ha="center")
    fig.autofmt_xdate(rotation=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_failure_plot(result: dict[str, Any]) -> Path:
    ensure_directories()
    path = RESULTS_DIR / "q3_failure_mode.png"
    fig, ax = plt.subplots(figsize=(8, 4.8))
    mu_values = [item["mu"] for item in result["records"]]
    train_errors = [item["train_error"] for item in result["records"]]
    test_errors = [item["test_error"] for item in result["records"]]
    ax.plot(mu_values, train_errors, marker="o", label="Train error")
    ax.plot(mu_values, test_errors, marker="s", label="Test error")
    ax.set_xlabel("Diversity weight mu")
    ax.set_ylabel("Error (1 - accuracy)")
    ax.set_title("Question 3.2: failure mode under excessive diversity")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def format_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    widths = {h: len(h) for h in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))

    def fmt_row(row: dict[str, Any]) -> str:
        return " | ".join(str(row[h]).ljust(widths[h]) for h in headers)

    separator = "-+-".join("-" * widths[h] for h in headers)
    lines = [fmt_row({h: h for h in headers}), separator]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)
