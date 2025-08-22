# app.py
# Streamlit web app for second-hand car price prediction using scikit-learn
# Supports: ['Make', 'Model', 'Year', 'Mileage', 'Engine', 'NumCylinders', 'pricesold', 'yearsold', 'PriceRange']
# - Trains a regression model to predict `pricesold`
# - Optionally trains a classifier to predict `PriceRange` if present
# - Lets users upload a CSV to train/evaluate and then do interactive predictions

import os
import io
import time
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score, f1_score, classification_report


st.set_page_config(page_title="Second-Hand Car Price Predictor", page_icon="🚗", layout="wide")
st.title("🚗 Second-Hand Car Price Predictor")
st.caption("Python • scikit-learn • Streamlit")

# ---------------------------
# Utility helpers
# ---------------------------
REQUIRED_COLUMNS = ["Make", "Model", "Year", "Mileage", "Engine", "NumCylinders"]
TARGET_REG = "pricesold"
TARGET_CLS = "PriceRange"  # optional
AUX_COLS = ["yearsold"]    # optional, can be derived from Year
MODEL_REG_PATH = "model_reg.pkl"
MODEL_CLS_PATH = "model_cls.pkl"
ENCODING_ORDER = {
    # Reasonable ordering for ranges if present
    TARGET_CLS: ["Low", "Mid", "High", "Premium"]
}


def derive_yearsold(df: pd.DataFrame) -> pd.DataFrame:
    if "yearsold" not in df.columns:
        current_year = pd.Timestamp.today().year
        if "Year" in df.columns:
            df["yearsold"] = (current_year - pd.to_numeric(df["Year"], errors="coerce")).clip(lower=0)
        else:
            df["yearsold"] = np.nan
    return df


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Trim column names and unify capitalization
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def split_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    y_reg = df[TARGET_REG] if TARGET_REG in df.columns else None
    y_cls = df[TARGET_CLS] if TARGET_CLS in df.columns else None
    X = df[[c for c in REQUIRED_COLUMNS + AUX_COLS if c in df.columns]].copy()
    return X, y_reg, y_cls


def build_regression_pipeline(categorical: List[str], numeric: List[str]) -> Pipeline:
    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_tf, categorical),
            ("num", num_tf, numeric),
        ]
    )
    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    return Pipeline(steps=[("pre", pre), ("rf", model)])


def build_classifier_pipeline(categorical: List[str], numeric: List[str]) -> Pipeline:
    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_tf, categorical),
            ("num", num_tf, numeric),
        ]
    )
    model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    return Pipeline(steps=[("pre", pre), ("rf", model)])


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return pd.DataFrame({
        "MAE": [mae],
        "RMSE": [rmse],
        "R^2": [r2]
    })


def evaluate_classification(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return pd.DataFrame({
        "Accuracy": [acc],
        "F1 (weighted)": [f1]
    })


# ---------------------------
# Sidebar: training/evaluation
# ---------------------------
st.sidebar.header("📦 Data & Training")
st.sidebar.write("Upload your CSV with columns: \n**Make, Model, Year, Mileage, Engine, NumCylinders, pricesold, yearsold (optional), PriceRange (optional)**")
uploaded = st.sidebar.file_uploader("Upload training CSV", type=["csv"]) 

col_train_a, col_train_b = st.sidebar.columns(2)
with col_train_a:
    do_train = st.button("Train / Retrain Models", use_container_width=True)
with col_train_b:
    do_clear = st.button("Clear Saved Models", use_container_width=True)

if do_clear:
    if os.path.exists(MODEL_REG_PATH):
        os.remove(MODEL_REG_PATH)
    if os.path.exists(MODEL_CLS_PATH):
        os.remove(MODEL_CLS_PATH)
    st.sidebar.success("Cleared saved models.")

# Persistent state
if "trained" not in st.session_state:
    st.session_state.trained = False

# ---------------------------
# Load & preview data
# ---------------------------
train_df = None
if uploaded is not None:
    try:
        train_df = pd.read_csv(uploaded)
        train_df = sanitize_columns(train_df)
        train_df = derive_yearsold(train_df)
        st.subheader("📄 Data Preview")
        st.dataframe(train_df.head(20), use_container_width=True)

        # Quick schema check
        missing_core = [c for c in REQUIRED_COLUMNS if c not in train_df.columns]
        if missing_core:
            st.warning(f"Missing required columns: {missing_core}. You can still proceed, but predictions may be degraded.")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

# ---------------------------
# Training logic
# ---------------------------
if do_train:
    if train_df is None:
        st.error("Please upload a CSV to train models.")
    else:
        with st.spinner("Training models..."):
            X, y_reg, y_cls = split_features_targets(train_df)

            # Identify column types
            categorical = [c for c in X.columns if X[c].dtype == 'object']
            numeric = [c for c in X.columns if c not in categorical]

            trained_any = False

            # Regression model (pricesold)
            if y_reg is not None:
                reg_pipe = build_regression_pipeline(categorical, numeric)
                reg_pipe.fit(X, y_reg)
                with open(MODEL_REG_PATH, "wb") as f:
                    pickle.dump(reg_pipe, f)
                st.success("Trained regression model for `pricesold` and saved to model_reg.pkl")
                trained_any = True

                # In-sample evaluation (for quick sanity check)
                yhat = reg_pipe.predict(X)
                st.markdown("**Regression (pricesold) — In-sample Metrics**")
                st.dataframe(evaluate_regression(y_reg, yhat))

            else:
                st.info("Column `pricesold` not found — skipping regression training.")

            # Classification model (PriceRange) — optional
            if y_cls is not None:
                # If y_cls isn't ordinal, we still treat it as categorical labels
                cls_pipe = build_classifier_pipeline(categorical, numeric)
                cls_pipe.fit(X, y_cls)
                with open(MODEL_CLS_PATH, "wb") as f:
                    pickle.dump(cls_pipe, f)
                st.success("Trained classifier for `PriceRange` and saved to model_cls.pkl")
                trained_any = True

                yhat_c = cls_pipe.predict(X)
                st.markdown("**Classification (PriceRange) — In-sample Metrics**")
                st.dataframe(evaluate_classification(y_cls, yhat_c))

            else:
                st.info("Column `PriceRange` not found — skipping classification training.")

            st.session_state.trained = trained_any

# ---------------------------
# Load saved models if present
# ---------------------------
reg_model = None
cls_model = None
if os.path.exists(MODEL_REG_PATH):
    try:
        with open(MODEL_REG_PATH, "rb") as f:
            reg_model = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load regression model: {e}")
if os.path.exists(MODEL_CLS_PATH):
    try:
        with open(MODEL_CLS_PATH, "rb") as f:
            cls_model = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load classifier model: {e}")

# ---------------------------
# Interactive prediction form
# ---------------------------
st.subheader("🧮 Predict")
st.write("Use the form to enter car details and get predictions. Models must be trained or loaded.")

if (reg_model is None) and (cls_model is None):
    st.info("Train models by uploading a CSV and clicking **Train / Retrain Models** in the sidebar.")

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        make = st.text_input("Make", "Toyota")
        model_name = st.text_input("Model", "Corolla")
        engine = st.text_input("Engine (e.g., 1.8L Petrol)", "1.2L Petrol")
    with c2:
        year = st.number_input("Year", min_value=1980, max_value=2050, value=2018, step=1)
        mileage = st.number_input("Mileage (km)", min_value=0, value=50000, step=1000)
        num_cyl = st.number_input("NumCylinders", min_value=2, max_value=16, value=4, step=1)
    with c3:
        yearsold_manual = st.number_input("yearsold (optional)", min_value=0, value=0, step=1, help="Leave 0 to auto-compute from Year")

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build a one-row DataFrame matching training feature names best-effort
    features = {
        "Make": [make],
        "Model": [model_name],
        "Year": [year],
        "Mileage": [mileage],
        "Engine": [engine],
        "NumCylinders": [num_cyl],
    }
    df_in = pd.DataFrame(features)
    df_in = derive_yearsold(df_in)
    if yearsold_manual and yearsold_manual > 0:
        df_in["yearsold"] = yearsold_manual

    pred_cols = [c for c in REQUIRED_COLUMNS + AUX_COLS if c in df_in.columns]

    outputs = {}
    if reg_model is not None:
        try:
            price_pred = reg_model.predict(df_in[pred_cols])[0]
            outputs["Predicted pricesold"] = round(float(price_pred), 2)
        except Exception as e:
            st.error(f"Regression prediction failed: {e}")

    if cls_model is not None:
        try:
            pr_pred = cls_model.predict(df_in[pred_cols])[0]
            outputs["Predicted PriceRange"] = str(pr_pred)
        except Exception as e:
            st.error(f"Classification prediction failed: {e}")

    if outputs:
        st.success("Prediction ready!")
        st.json(outputs)
    else:
        st.warning("No models loaded to make predictions.")

# ---------------------------
# Tips & How-to
# ---------------------------
with st.expander(ℹ️ Usage Tips"):
    st.markdown(
        """
        **Data Requirements**
        - Minimum required columns: `Make, Model, Year, Mileage, Engine, NumCylinders`.
        - For price regression, include the target column `pricesold` in your training CSV.
        - For range classification, include the target column `PriceRange` (e.g., `Low/Mid/High/Premium`).
        - `yearsold` is optional; if omitted it will be derived from `Year`.

        **Workflow**
        1. Upload your CSV on the left.
        2. Click **Train / Retrain Models**.
        3. Use the **Predict** form to get results.

        **Notes**
        - This app trains in-sample for quick feedback. For production, split into train/validation sets.
        - Models are saved to `model_reg.pkl` and `model_cls.pkl` in the working directory.
        - You can clear saved models from the sidebar at any time.
        """
    )

# ---------------------------
# Optional: Simple holdout evaluation (if user uploaded a dataset)
# ---------------------------
if train_df is not None and (reg_model is not None or cls_model is not None):
    st.subheader("🧪 Quick Evaluation on Uploaded Data (holdout split)")
    test_size = st.slider("Test size (for holdout split)", 0.1, 0.5, 0.2, 0.05)

    from sklearn.model_selection import train_test_split
    X_all, y_reg_all, y_cls_all = split_features_targets(train_df)

    # Keep rows where at least one target exists
    mask = pd.Series(True, index=train_df.index)
    if y_reg_all is not None:
        mask &= y_reg_all.notna()
    if y_cls_all is not None:
        mask &= y_cls_all.notna()

    X_all = X_all[mask]
    y_reg_all = y_reg_all[mask] if y_reg_all is not None else None
    y_cls_all = y_cls_all[mask] if y_cls_all is not None else None

    if len(X_all) >= 10:
        X_tr, X_te, y_reg_tr, y_reg_te = train_test_split(
            X_all, y_reg_all, test_size=test_size, random_state=42
        )
        # Retrain fresh for fair holdout metrics
        categorical = [c for c in X_all.columns if X_all[c].dtype == 'object']
        numeric = [c for c in X_all.columns if c not in categorical]

        if y_reg_all is not None:
            reg_hold = build_regression_pipeline(categorical, numeric)
            reg_hold.fit(X_tr, y_reg_tr)
            y_pred_te = reg_hold.predict(X_te)
            st.markdown("**Holdout Regression Metrics**")
            st.dataframe(evaluate_regression(y_reg_te, y_pred_te))

        if y_cls_all is not None:
            X_trc, X_tec, y_trc, y_tec = train_test_split(
                X_all, y_cls_all, test_size=test_size, random_state=42
            )
            cls_hold = build_classifier_pipeline(categorical, numeric)
            cls_hold.fit(X_trc, y_trc)
            y_pred_tec = cls_hold.predict(X_tec)
            st.markdown("**Holdout Classification Metrics**")
            st.dataframe(evaluate_classification(y_tec, y_pred_tec))
    else:
        st.info("Need at least 10 labeled rows for a quick holdout evaluation.")

st.caption("© 2025 Second-Hand Car Price Predictor • Built with Streamlit & scikit-learn")
