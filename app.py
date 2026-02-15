import io
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc as sk_auc,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# XGBoost (required)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="ML Classification Benchmark", layout="wide")
st.title("ML Classification Benchmark")


# -------------------------
# Session State
# -------------------------
if "auto_ran_default" not in st.session_state:
    st.session_state.auto_ran_default = False

# store results separately for each tab
if "state_default" not in st.session_state:
    st.session_state.state_default = {"ready": False, "payload": None}

if "state_upload" not in st.session_state:
    st.session_state.state_upload = {"ready": False, "payload": None}


# -------------------------
# Helpers
# -------------------------
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
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])

    # one-hot encode features
    X = pd.get_dummies(X_raw, drop_first=False)

    # encode target
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.copy()
        if pd.api.types.is_float_dtype(y) and np.all(np.isclose(y, np.round(y))):
            y = y.round().astype(int)
        uniques = list(pd.unique(y))
        mapping = {val: idx for idx, val in enumerate(sorted(uniques))}
        y = y.map(mapping).astype(int)
        class_labels = [str(v) for v in sorted(uniques)]
    else:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), name=target_col)
        class_labels = list(le.classes_)

    if y.isna().any():
        raise ValueError("Target contains missing values after preprocessing.")

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
    models = {
        "Logistic Regression": make_lr_pipeline(enable_scaling=use_scaling, C=lr_c),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=dt_max_depth,
            min_samples_split=dt_min_samples_split,
            random_state=random_state,
        ),
        "KNN": make_knn_pipeline(enable_scaling=use_scaling, k=knn_k),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if XGBOOST_AVAILABLE:
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

        results[name] = {"metrics": row, "cm": cm, "report_df": rep_df, "y_proba": y_proba}
        reports[name] = rep_df

    metrics_df = pd.DataFrame(rows).sort_values(by="MCC", ascending=False)
    return metrics_df, results, reports


def safe_stratify(y, test_size_val: float):
    counts = Counter(y)
    min_count = min(counts.values())
    n_classes = len(counts)
    if min_count < 2:
        return None
    n = len(y)
    n_test = int(np.ceil(test_size_val * n))
    if n_test < n_classes:
        return None
    return y


@st.cache_data(show_spinner=False)
def cached_train_eval(df_clean: pd.DataFrame, target_col: str, params: dict):
    X, y, class_labels = prepare_X_y(df_clean, target_col=target_col)
    n_classes = len(class_labels)

    stratify_val = None
    if params["use_stratify"] and n_classes > 1:
        stratify_val = safe_stratify(y, params["test_size"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=stratify_val,
    )

    models = build_models(
        n_classes=n_classes,
        random_state=params["random_state"],
        use_scaling=params["use_scaling"],
        lr_c=params["lr_c"],
        dt_max_depth=params["dt_max_depth"],
        dt_min_samples_split=params["dt_min_samples_split"],
        knn_k=params["knn_k"],
        rf_n_estimators=params["rf_n_estimators"],
        rf_max_depth=params["rf_max_depth"],
        xgb_n_estimators=params["xgb_n_estimators"],
        xgb_max_depth=params["xgb_max_depth"],
        xgb_learning_rate=params["xgb_learning_rate"],
        xgb_subsample=params["xgb_subsample"],
        xgb_colsample_bytree=params["xgb_colsample_bytree"],
    )

    metrics_df, results, reports = evaluate_all_models(
        X_train, X_test, y_train, y_test, models, class_labels
    )
    return X, y, class_labels, X_test, y_test, metrics_df, results, reports


def plot_roc(model_name: str, y_test, y_proba, class_labels):
    if y_proba is None:
        st.info("ROC curve not available (model has no probability/score output).")
        return

    n_classes = len(class_labels)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.set_aspect("equal", adjustable="box")

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title(f"ROC Curve — {model_name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
    else:
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_proba[:, i])
            roc_auc = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{class_labels[i]} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title(f"ROC Curve (OvR) — {model_name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("Split & Preprocessing")
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
use_stratify = st.sidebar.checkbox("Use stratified split (recommended)", value=True)
use_scaling = st.sidebar.checkbox("Enable feature scaling (LR & KNN)", value=True)

missing_strategy = st.sidebar.selectbox(
    "Missing value strategy",
    [
        "Impute: Numeric=median, Categorical=mode",
        "Impute: Numeric=mean, Categorical=mode",
        "Drop rows with any missing values",
    ],
)

st.sidebar.header("Hyperparameters")
lr_c = st.sidebar.slider("Logistic Regression: C", 0.01, 10.0, 1.0, 0.01)
dt_max_depth = st.sidebar.slider("Decision Tree: max_depth", 1, 30, 7, 1)
dt_min_samples_split = st.sidebar.slider("Decision Tree: min_samples_split", 2, 30, 2, 1)
knn_k = st.sidebar.slider("KNN: n_neighbors (k)", 1, 25, 5, 1)
rf_n_estimators = st.sidebar.slider("Random Forest: n_estimators", 50, 800, 250, 50)
rf_max_depth = st.sidebar.slider("Random Forest: max_depth", 1, 40, 12, 1)
xgb_n_estimators = st.sidebar.slider("XGBoost: n_estimators", 50, 800, 250, 50)
xgb_max_depth = st.sidebar.slider("XGBoost: max_depth", 1, 15, 4, 1)
xgb_learning_rate = st.sidebar.slider("XGBoost: learning_rate", 0.01, 0.5, 0.1, 0.01)
xgb_subsample = st.sidebar.slider("XGBoost: subsample", 0.5, 1.0, 0.9, 0.05)
xgb_colsample_bytree = st.sidebar.slider("XGBoost: colsample_bytree", 0.5, 1.0, 0.9, 0.05)

params = {
    "test_size": test_size,
    "random_state": random_state,
    "use_stratify": use_stratify,
    "use_scaling": use_scaling,
    "lr_c": lr_c,
    "dt_max_depth": dt_max_depth,
    "dt_min_samples_split": dt_min_samples_split,
    "knn_k": knn_k,
    "rf_n_estimators": rf_n_estimators,
    "rf_max_depth": rf_max_depth,
    "xgb_n_estimators": xgb_n_estimators,
    "xgb_max_depth": xgb_max_depth,
    "xgb_learning_rate": xgb_learning_rate,
    "xgb_subsample": xgb_subsample,
    "xgb_colsample_bytree": xgb_colsample_bytree,
}

if not XGBOOST_AVAILABLE:
    st.sidebar.warning("XGBoost not installed. Add `xgboost` in requirements.txt.")


# -------------------------
# Tabs
# -------------------------
tab_default, tab_upload = st.tabs(["Default Dataset (Breast Cancer)", "Upload Dataset (CSV)"])


def render_results(payload, source_key: str):
    X, y, class_labels, X_test, y_test, metrics_df, results, reports, target_col = payload

    st.subheader("Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{X.shape[0]}")
    c2.metric("Features (after encoding)", f"{X.shape[1]}")
    c3.metric("Classes", f"{len(class_labels)}")
    c4.metric("Target", target_col)

    st.subheader("Metrics Table")
    st.dataframe(
        metrics_df.style.format(
            {
                "Accuracy": "{:.4f}",
                "AUC (OvR macro)": "{:.4f}",
                "Precision (macro)": "{:.4f}",
                "Recall (macro)": "{:.4f}",
                "F1 (macro)": "{:.4f}",
                "MCC": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Download Metrics Table (CSV)",
        data=metrics_df.to_csv(index=False).encode("utf-8"),
        file_name="model_metrics.csv",
        mime="text/csv",
        key=f"dl_metrics_{source_key}",
    )

    st.subheader("Model Details")
    model_name = st.selectbox("Select a model", options=list(results.keys()), key=f"model_{source_key}")
    out = results[model_name]
    m = out["metrics"]

    left, right = st.columns([1.1, 1])

    with left:
        st.write("### Key Metrics")
        mc = st.columns(2)
        mc[0].metric("Accuracy", f"{m['Accuracy']:.4f}")
        mc[0].metric("Precision", f"{m['Precision (macro)']:.4f}")
        mc[0].metric("Recall", f"{m['Recall (macro)']:.4f}")
        mc[1].metric("AUC", f"{m['AUC (OvR macro)']:.4f}" if not np.isnan(m["AUC (OvR macro)"]) else "NA")
        mc[1].metric("F1", f"{m['F1 (macro)']:.4f}")
        mc[1].metric("MCC", f"{m['MCC']:.4f}")

        st.write("**Confusion Matrix**")
        cm = out["cm"]
        cm_df = pd.DataFrame(
            cm,
            index=[f"True {c}" for c in class_labels],
            columns=[f"Pred {c}" for c in class_labels],
        )
        st.dataframe(cm_df, use_container_width=True)

    with right:
        st.write("### ROC Curve")
        plot_roc(model_name, y_test, out.get("y_proba"), class_labels)

    st.write("**Classification Report**")
    rep_df = out["report_df"]
    st.dataframe(rep_df, use_container_width=True)

    st.download_button(
        f"⬇️ Download {model_name} Report (CSV)",
        data=rep_df.to_csv(index=True).encode("utf-8"),
        file_name=f"classification_report_{model_name.replace(' ', '_').lower()}.csv",
        mime="text/csv",
        key=f"dl_report_{source_key}_{model_name}",
    )

    # zip all reports
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for mn, df_report in reports.items():
            fname = f"classification_report_{mn.replace(' ', '_').lower()}.csv"
            zf.writestr(fname, df_report.to_csv(index=True))
    zip_buffer.seek(0)

    st.download_button(
        "⬇️ Download ALL Reports (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="classification_reports.zip",
        mime="application/zip",
        key=f"dl_zip_{source_key}",
    )


def run_pipeline(df: pd.DataFrame, default_target: str, auto_run: bool, state_key: str):
    # Preview
    st.subheader("Preview")
    with st.expander("Preview data", expanded=True):
        max_prev = max(10, min(200, len(df)))
        default_prev = min(50, max_prev)
        step_prev = 10 if max_prev >= 20 else 1
        n_prev = st.slider("Rows to preview", 10, max_prev, default_prev, step_prev, key=f"prev_{state_key}")
        st.dataframe(df.head(n_prev), use_container_width=True)

    st.divider()

    # Target + Train under preview
    col_target, col_train = st.columns([1, 1])

    with col_target:
        st.subheader("Target Variable")
        auto_target = default_target if default_target in df.columns else df.columns[-1]
        default_index = list(df.columns).index(auto_target)
        target_col = st.selectbox(
            f"Auto-detected = {auto_target}",
            options=list(df.columns),
            index=default_index,
            key=f"target_{state_key}",
        )

    with col_train:
        st.subheader("Train & Evaluate")
        st.write("")
        st.write("")
        run_btn = st.button("Train & Evaluate Models", key=f"run_{state_key}")

    # default auto run once
    if auto_run and (not st.session_state.auto_ran_default):
        run_btn = True
        st.session_state.auto_ran_default = True

    df_clean = apply_missing_value_strategy(df, missing_strategy)

    if run_btn:
        with st.spinner("Training (cached)..."):
            X, y, class_labels, X_test, y_test, metrics_df, results, reports = cached_train_eval(
                df_clean=df_clean,
                target_col=target_col,
                params=params,
            )

        payload = (X, y, class_labels, X_test, y_test, metrics_df, results, reports, target_col)
        st.session_state[state_key]["ready"] = True
        st.session_state[state_key]["payload"] = payload
        st.success("Training complete ✅")

    # Always render from state so changing model works
    if st.session_state[state_key]["ready"]:
        render_results(st.session_state[state_key]["payload"], source_key=state_key)
    else:
        st.info("Click **Train & Evaluate Models** to generate results.")


# -------------------------
# Default tab
# -------------------------
with tab_default:
    st.subheader("Default Dataset: Breast Cancer Wisconsin")
    st.caption("Auto-trains once on load (cached). Target = diagnosis.")

    data = load_breast_cancer(as_frame=True)
    df_default = data.frame.copy()
    df_default["diagnosis"] = df_default["target"].map({0: "malignant", 1: "benign"})
    df_default = df_default.drop(columns=["target"])

    run_pipeline(df_default, default_target="diagnosis", auto_run=True, state_key="state_default")


# -------------------------
# Upload tab
# -------------------------
with tab_upload:
    st.subheader("Upload a Classification Dataset (CSV)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="upload_csv")

    if uploaded is None:
        st.info("Upload a CSV to start.")
    else:
        df_upload = pd.read_csv(uploaded)
        run_pipeline(df_upload, default_target=df_upload.columns[-1], auto_run=False, state_key="state_upload")
