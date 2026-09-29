from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer


PRIMARY_SHAPES = (
    (6, 1, "sqrt", None),
    (6, 1, "sqrt", "balanced"),
    (12, 5, 0.5, None),
    (None, 1, "sqrt", None),
    (None, 5, 0.5, None),
    (None, 20, 1.0, "balanced"),
)


def primary_candidates(views: tuple[str, ...]) -> list[dict]:
    records = []
    for view in views:
        for family in ("RF", "ExtraTrees"):
            for index, (depth, leaf, fraction, weight) in enumerate(PRIMARY_SHAPES):
                records.append(
                    {
                        "family": family,
                        "view": view,
                        "index": index,
                        "n_estimators": 256,
                        "max_depth": depth,
                        "min_samples_leaf": leaf,
                        "max_features": fraction,
                        "class_weight": weight,
                        "key": f"{family}/{view}/{index}",
                    }
                )
            if family == "RF":
                records.append(
                    {
                        "family": family,
                        "view": view,
                        "index": 6,
                        "n_estimators": 1000,
                        "max_depth": 6,
                        "min_samples_leaf": 1,
                        "max_features": "sqrt",
                        "class_weight": "balanced",
                        "key": f"{family}/{view}/6",
                    }
                )
    return records


def controlled_candidates(view: str) -> list[dict]:
    return [
        {
            "family": family,
            "view": view,
            "index": 0,
            "n_estimators": 32,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "max_features": 0.5,
            "class_weight": None,
            "key": f"{family}/{view}/0",
        }
        for family in ("RF", "ExtraTrees")
    ]


class ForestModel:
    def __init__(self, specification: dict, seed: int, threads: int = 1):
        if specification["family"] not in {"RF", "ExtraTrees"}:
            raise ValueError("ForestModel supports RF and ExtraTrees")
        self.specification = dict(specification)
        self.seed = int(seed)
        self.threads = int(threads)

    def fit(self, features: np.ndarray, labels) -> "ForestModel":
        labels = np.asarray(labels, dtype=int)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Training requires both binary outcomes")
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        values = self.imputer.fit_transform(features)
        parameters = {
            key: value
            for key, value in self.specification.items()
            if key not in {"family", "view", "index", "key"}
        }
        estimator_class = (
            RandomForestClassifier
            if self.specification["family"] == "RF"
            else ExtraTreesClassifier
        )
        self.estimator = estimator_class(
            **parameters, random_state=self.seed, n_jobs=self.threads
        ).fit(values, labels)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        prediction = self.estimator.predict_proba(self.imputer.transform(features))[:, 1]
        prediction = np.asarray(prediction, dtype=float)
        if not np.isfinite(prediction).all() or (prediction < 0).any() or (prediction > 1).any():
            raise ValueError("Estimator returned invalid probabilities")
        return prediction
