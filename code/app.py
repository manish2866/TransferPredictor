import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from Visualizations import (
    draw_scatter_plots,
    draw_box_plots,
    draw_joint_plots,
    draw_bar_plots,
)

# ------------------------------
# Helpers
# ------------------------------

TARGET_CANDIDATES = ["value_euro", "market_value_eur", "market_value", "Value", "value"]

def _infer_target_column(df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    # Fallback: choose the most "value-like" numeric column by name
    candidates = [c for c in df.columns if c.lower().startswith("value") and pd.api.types.is_numeric_dtype(df[c])]
    if candidates:
        return candidates[0]
    # Absolute fallback: last numeric column
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols[-1] if num_cols else df.columns[-1]


def _safe_mean(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        # return zeros with same length
        return pd.Series(np.zeros(len(df)), index=df.index)
    return df[existing].mean(axis=1)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregate feature groups if source columns exist."""
    df = df.copy()

    # Normalize common column naming variations
    if "wages_euro" in df.columns and "wage_euro" not in df.columns:
        df.rename(columns={"wages_euro": "wage_euro"}, inplace=True)

    if "positions" in df.columns and "primary_position" not in df.columns:
        df["primary_position"] = (
            df["positions"]
            .fillna("")
            .astype(str)
            .apply(lambda s: s.split(",")[0].strip() if s else "")
        )

    # Buckets used in many FIFA-like datasets
    attacking = ["crossing", "finishing", "heading_accuracy", "short_passing", "volleys"]
    skill = ["dribbling", "curve", "freekick_accuracy", "long_passing", "ball_control"]
    movement = ["acceleration", "sprint_speed", "agility", "reactions", "balance"]
    power = ["shot_power", "jumping", "stamina", "strength", "long_shots"]
    mentality = ["aggression", "interceptions", "positioning", "vision", "penalties", "composure"]
    defending = ["marking", "standing_tackle", "sliding_tackle"]

    df["attacking_stats"] = _safe_mean(df, attacking).round(2)
    df["skill_stats"] = _safe_mean(df, skill).round(2)
    df["movement_stats"] = _safe_mean(df, movement).round(2)
    df["power_stats"] = _safe_mean(df, power).round(2)
    df["mentality_stats"] = _safe_mean(df, mentality).round(2)
    df["defending_stats"] = _safe_mean(df, defending).round(2)

    # Individual performance proxy
    if {"overall_rating", "potential"}.issubset(df.columns):
        df["ind_performance"] = ((df["overall_rating"] + df["potential"]) / 2).round(2)
    else:
        df["ind_performance"] = _safe_mean(df, ["overall_rating", "potential"]).round(2)

    skills = ["attacking_stats","skill_stats","movement_stats","power_stats","mentality_stats","defending_stats"]
    df["mean_skill_ind"] = _safe_mean(df, skills).round(2)

    return df


def upload_file() -> pd.DataFrame | None:
    st.subheader("Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)  # try without encoding
        st.write("Preview:", df.head())
        return df
    return None


def train_model(df: pd.DataFrame, target_col: str) -> Tuple[XGBRegressor, List[str], StandardScaler]:
    """Train a simple XGBRegressor on numeric features."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
    X = df[numeric_cols].copy()
    y = df[target_col].copy()

    # Fill missing numeric values
    X = X.fillna(X.median(numeric_only=True))
    y = y.fillna(y.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.06,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=4,
        objective="reg:squarederror",
    )
    model.fit(X_train_scaled, y_train)

    # Metrics
    pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    st.info(f"Model performance — R²: {r2:.3f} | MAE: {mae:,.2f}")

    return model, numeric_cols, scaler


def user_input_form(feature_names: List[str]) -> pd.DataFrame:
    st.subheader("Enter Player Features for Prediction")
    values = {}
    for name in feature_names:
        # Provide sane defaults
        if "age" in name.lower():
            values[name] = st.number_input(name, min_value=0, max_value=60, value=25)
        elif any(k in name.lower() for k in ["rating", "reputation", "weak_foot", "skill_moves", "stats"]):
            values[name] = st.number_input(name, min_value=0, max_value=100, value=50)
        else:
            # generic numeric input
            values[name] = st.number_input(name, value=float(0.0))
    return pd.DataFrame([values])


def main():
    st.title("Player Value Modeling & Visualization")
    st.caption("Upload data, explore visualizations, engineer features, and train a model to predict market value.")

    df = upload_file()
    if df is None:
        st.stop()

    # Feature engineering
    df = feature_engineering(df)

    # Choose target
    target_col = st.selectbox("Select target column (to predict)", options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])], index=max(0, [c in df.columns for c in TARGET_CANDIDATES].index(True)) if any(c in df.columns for c in TARGET_CANDIDATES) else 0)
    st.write(f"Using **{target_col}** as target.")

    # Visualizations
    with st.expander("Visualizations", expanded=False):
        vis_type = st.selectbox("Select a visualization type", ["Scatter Plots", "Box Plots", "Joint Plots", "Bar Plots"])

        if vis_type == "Scatter Plots":
            x_feature = st.selectbox("X-axis feature", options=[c for c in df.columns if c != target_col])
            draw_scatter_plots(df, x_feature, target_col)

        elif vis_type == "Box Plots":
            categorical_candidates = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            if not categorical_candidates:
                st.warning("No categorical columns found for box plot.")
            else:
                cat = st.selectbox("Category", options=categorical_candidates)
                draw_box_plots(df, cat, target_col)

        elif vis_type == "Joint Plots":
            x_feature = st.selectbox("X-axis feature", options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_col])
            draw_joint_plots(df, x_feature, target_col)

        elif vis_type == "Bar Plots":
            feat = st.selectbox("Numeric feature", options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_col])
            draw_bar_plots(df, feat)

    # Train model
    st.header("Model Training")
    model, features, scaler = train_model(df, target_col)

    # Prediction
    st.header("Predict")
    ui = user_input_form(features)
    if st.button("Predict value"):
        pred = model.predict(scaler.transform(ui[features]))
        st.success(f"Predicted {target_col}: {float(pred[0]):,.2f}")


if __name__ == "__main__":
    main()
