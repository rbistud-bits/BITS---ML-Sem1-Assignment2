import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# XGBoost (required by assignment)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


def ensure_model_dir():
    os.makedirs("model", exist_ok=True)


def apply_missing_value_strategy(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "Drop rows with any missing values":
        return df.dropna(axis=0).copy()

    df2 = df.copy()
    use_mean = strategy == "Impute: Numeric=mean, Categorical=mode"

    for col in df2.columns:
        if df2[col].isna().any():
            if pd.api.types.is_numeric_dtype(df2[col]):
                fill_val = df2[col].mean() if use_mean else df2[col].median()
                df2[col] = df2[col].fillna(fill_val)
            else:
                mode = df2[col].mode(dropna=True)
                df2[col] = df2[col].fillna(mode.iloc[0] if len(mode) else "missing")
    return df2


def prepare_X_y(df: pd.DataFrame, target_col: str):
    """
    - One-hot encode categorical features
    - Always LabelEncode target to 0..K-1 (robust for numeric labels too)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y_raw = df[target_col]
    if y_raw.isna().any():
        raise ValueError("Target contains missing values. Please impute/drop missing target rows.")

    X_raw = df.drop(columns=[target_col])
    X = pd.get_dummies(X_raw, drop_first=False)

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw.astype(str)), name=target_col)
    class_labels = [str(c) for c in le.classes_]

    return X, y, class_labels


def get_proba_or_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)

    if hasattr(model, "decision_function"):
        scores = np.array(model.decision_function(X_test))
        if scores.ndim == 1:
            s = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            return np.vstack([1 - s, s]).T
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-9)

    return None


def compute_auc_general(y_true, y_proba, n_classes: int):
    if y_proba is None:
        return np.nan
    try:
        if n_classes == 2:
            return roc_auc_score(y_true, y_proba[:, 1])
        return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        return np.nan


def make_lr_pipeline(enable_scaling: bool, C: float):
    steps = []
    if enable_scaling:
        steps.append(("scaler", StandardScaler(with_mean=False)))
    steps.append(("clf", LogisticRegression(C=C, max_iter=5000, solver="lbfgs")))
    return Pipeline(steps)


def make_knn_pipeline(enable_scaling: bool, k: int):
    steps = []
    if enable_scaling:
        steps.append(("scaler", StandardScaler(with_mean=False)))
    steps.append(("clf", KNeighborsClassifier(n_neighbors=k)))
    return Pipeline(steps)


def build_models(
    n_classes: int,
    random_state: int,
    use_scaling: bool,
    lr_c: float,
    dt_max_depth: int,
    dt_min_samples_split: int,
    knn_k: int,
    rf_n_estimators: int,
    rf_max_depth: int,
    xgb_n_estimators: int,
    xgb_max_depth: int,
    xgb_learning_rate: float,
    xgb_subsample: float,
    xgb_colsample_bytree: float,
):
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("XGBoost is required by the assignment. Install with: pip install xgboost")

    models = {
        "Logistic Regression": make_lr_pipeline(enable_scaling=use_scaling, C=lr_c),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=dt_max_depth, min_samples_split=dt_min_samples_split, random_state=random_state
        ),
        "KNN": make_knn_pipeline(enable_scaling=use_scaling, k=knn_k),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=random_state, n_jobs=-1
        ),
    }

    if n_classes == 2:
        models["XGBoost"] = XGBClassifier(
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            subsample=xgb_subsample,
            colsample_bytree=xgb_colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        models["XGBoost"] = XGBClassifier(
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            subsample=xgb_subsample,
            colsample_bytree=xgb_colsample_bytree,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )

    return models


def evaluate_all_models(X_train, X_test, y_train, y_test, models, class_labels):
    n_classes = len(class_labels)
    rows = []
    results = {}
    reports = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = get_proba_or_scores(model, X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        auc_val = compute_auc_general(y_test, y_proba, n_classes=n_classes)

        cm = confusion_matrix(y_test, y_pred)

        rep_dict = classification_report(
            y_test,
            y_pred,
            labels=list(range(n_classes)),
            target_names=class_labels,
            output_dict=True,
            zero_division=0,
        )
        rep_df = pd.DataFrame(rep_dict).T

        row = {
            "Model": name,
            "Accuracy": acc,
            "AUC (OvR macro)": auc_val,
            "Precision (macro)": prec,
            "Recall (macro)": rec,
            "F1 (macro)": f1m,
            "MCC": mcc,
        }
        rows.append(row)

        results[name] = {
            "metrics": row,
            "cm": cm,
            "report_df": rep_df,
            "y_proba": y_proba,
        }
        reports[name] = rep_df

    metrics_df = pd.DataFrame(rows).sort_values(by="MCC", ascending=False)
    return metrics_df, results, reports
