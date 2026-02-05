from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

@dataclass
class ExploitModelResult:
    auc: float
    top_decile_precision: float
    model_info: Dict[str, Any]

def train_exploit_predictor(df: pd.DataFrame, random_state: int = 7) -> Tuple[Any, ExploitModelResult]:
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X = df.copy()
    y = X["exploited_observed"].astype(int).values
    X = X.drop(columns=["exploited_observed"])

    num_cols = [c for c in ["cvss_base_score","dep_update_delay_days","scanner_confirmed_present","telemetry_exploit_signal","sbom_present"] if c in X.columns]
    text_col = "description"

    pre = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), text_col),
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=random_state)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))

    k = max(1, int(0.10 * len(proba)))
    idx = np.argsort(-proba)[:k]
    top_prec = float(y_test[idx].mean())

    res = ExploitModelResult(
        auc=auc,
        top_decile_precision=top_prec,
        model_info={"n_train": int(len(y_train)), "n_test": int(len(y_test)), "num_features": num_cols, "text_features": "tfidf(1-2grams,5000)"},
    )
    return pipe, res
