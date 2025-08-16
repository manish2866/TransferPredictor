
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# NOTE:
# - Avoid Streamlit global pyplot setting (was removed). 
# - Always render with st.pyplot(fig) using explicit figures,
#   so no deprecation toggles are needed.

def _ensure_numeric(df: pd.DataFrame, cols):
    """Coerce columns to numeric where possible (non-destructively)."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def draw_scatter_plots(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    title: str | None = None,
    height: int = 500,
):
    """Scatter plot with optional hue."""
    if x not in df.columns or y not in df.columns:
        st.info("Select valid columns for X and Y.")
        return
    
    work = _ensure_numeric(df, [x, y])
    work = work.dropna(subset=[x, y])
    if work.empty:
        st.warning("No data to plot after cleaning.")
        return

    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    sns.scatterplot(data=work, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(title or f"{y} vs {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

def draw_box_plots(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str | None = None,
):
    """Box plot of y across categories x."""
    if x not in df.columns or y not in df.columns:
        st.info("Select valid columns for X (category) and Y (value).")
        return

    work = _ensure_numeric(df, [y]).dropna(subset=[x, y])
    if work.empty:
        st.warning("No data to plot after cleaning.")
        return

    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    sns.boxplot(data=work, x=x, y=y, ax=ax)
    ax.set_title(title or f"{y} by {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig, clear_figure=True)

def draw_joint_plots(
    df: pd.DataFrame,
    x: str,
    y: str,
    kind: str = "scatter",
    title: str | None = None,
):
    """Joint plot (scatter/hex/kde) between two numeric variables."""
    if x not in df.columns or y not in df.columns:
        st.info("Select valid columns for X and Y.")
        return

    work = _ensure_numeric(df, [x, y]).dropna(subset=[x, y])
    if work.empty:
        st.warning("No data to plot after cleaning.")
        return

    # Use seaborn jointplot which creates its own figure
    g = sns.jointplot(data=work, x=x, y=y, kind=kind, height=5)
    if title:
        # jointplot returns a JointGrid (with fig attribute on newer seaborn)
        try:
            g.figure.suptitle(title)
            g.figure.tight_layout()
            g.figure.subplots_adjust(top=0.92)
            st.pyplot(g.figure, clear_figure=True)
            return
        except Exception:
            pass
    # Fallback
    try:
        st.pyplot(g.figure, clear_figure=True)
    except Exception:
        # Older seaborn: pull the current figure
        fig = plt.gcf()
        st.pyplot(fig, clear_figure=True)

def draw_bar_plots(
    df: pd.DataFrame,
    category: str,
    value: str,
    agg: str = "mean",
    top_n: int = 20,
    title: str | None = None,
):
    """Bar plot of aggregated values per category."""
    if category not in df.columns or value not in df.columns:
        st.info("Select valid columns for category and value.")
        return

    work = _ensure_numeric(df, [value]).dropna(subset=[category, value])
    if work.empty:
        st.warning("No data to plot after cleaning.")
        return

    if agg not in {"mean", "sum", "median"}:
        agg = "mean"

    grouped = getattr(work.groupby(category)[value], agg)().reset_index()
    grouped = grouped.sort_values(by=value, ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    sns.barplot(data=grouped, x=category, y=value, ax=ax)
    ax.set_title(title or f"{value} ({agg}) by {category}")
    ax.set_xlabel(category)
    ax.set_ylabel(f"{value} ({agg})")
    ax.tick_params(axis="x", rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig, clear_figure=True)
