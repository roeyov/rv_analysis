"""
mcmc.selector_app — Streamlit-based best-row chooser comparison tool.

Exposes ``get_best_row()`` which is also used by ``mcmc.batch``.

Usage (Streamlit app):
    streamlit run -m MCMC.selector_app
"""

import os
import glob
from pathlib import Path

import pandas as pd
import numpy as np


# -------------------------------------------------------
# Selector helpers (no Streamlit dependency)
# -------------------------------------------------------

def get_best_row(results_df, filter_expr, field_to_check, take_min):
    """
    Select the best row from *results_df* via a pandas-query *filter_expr*,
    optimising *field_to_check* for min or max.
    """
    if results_df is None or results_df.empty:
        print("Results DataFrame is empty.")
        return None

    df = results_df.copy()

    # 1. Null-hyp reference value
    null_field = None
    if field_to_check in df.columns and "candidate_method" in df.columns:
        null_mask = df["candidate_method"].str.contains("null_hyp_jitter", na=False)
        if null_mask.any():
            null_field = df.loc[null_mask, field_to_check].iloc[0]

    # 2. Apply user filter
    if filter_expr is not None and filter_expr.strip():
        try:
            clean_expr = " ".join(filter_expr.splitlines())
            df = df.query(clean_expr, engine="python")
        except Exception as e:
            print(f"Failed to apply filter expression '{filter_expr}': {e}")
            return None

    if df.empty:
        print(f"No rows remain after filters: {filter_expr!r}")
        return None

    # 3. Field-based extra filtering
    field = field_to_check
    if field not in df.columns:
        print(f"Field '{field}' not in DataFrame columns.")
        return None

    if field == "bic" and null_field is not None:
        df = df[df[field] < null_field]

    df = df[df[field].notna()]
    if df.empty:
        print(f"No valid values in '{field}' after NaN removal.")
        return None

    # 4. Select best row
    idx = df[field].idxmin() if take_min else df[field].idxmax()
    return df.loc[idx]


# -------------------------------------------------------
# Data processing helpers
# -------------------------------------------------------

def collect_best_rows(sol_dir, filter_expr, field_to_check, take_min):
    """Iterate over sol_dir/{star}/lmfit_summary.csv and pick best rows."""
    star_names = os.listdir(sol_dir)
    rows = []

    for star_name in star_names:
        lmfit_path = os.path.join(sol_dir, star_name, "lmfit_summary.csv")
        if not os.path.exists(lmfit_path):
            continue
        try:
            df = pd.read_csv(lmfit_path)
        except Exception as e:
            print(f"Failed reading {lmfit_path}: {e}")
            continue

        best_row = get_best_row(df, filter_expr, field_to_check, take_min)
        if best_row is None:
            continue

        best_row = best_row.copy()
        best_row["star_name"] = star_name

        # Attach null-hyp reference value
        null_val = np.nan
        if "candidate_method" in df.columns and field_to_check in df.columns:
            null_mask = df["candidate_method"].str.contains("null_hyp", na=False)
            if null_mask.any():
                null_df = df[null_mask]
                best_has_jitter = "jitter" in str(best_row.get("candidate_method", ""))
                if best_has_jitter:
                    sub = null_df[null_df["candidate_method"].str.contains("jitter", na=False)]
                    null_val = sub[field_to_check].iloc[0] if not sub.empty else null_df[field_to_check].iloc[0]
                else:
                    sub = null_df[~null_df["candidate_method"].str.contains("jitter", na=False)]
                    null_val = sub[field_to_check].iloc[0] if not sub.empty else null_df[field_to_check].iloc[0]

        best_row[f"null_{field_to_check}"] = null_val
        rows.append(best_row)

    if not rows:
        return pd.DataFrame()
    result_df = pd.DataFrame(rows)
    return result_df.set_index("star_name").sort_index()


# -------------------------------------------------------
# Streamlit UI helpers
# -------------------------------------------------------

def _embed_html_file(html_path, height=1000):
    """Embed a local HTML file in Streamlit."""
    import streamlit as st
    import streamlit.components.v1 as components

    if not os.path.exists(html_path):
        st.warning(f"File not found: {html_path}")
        return
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    except UnicodeDecodeError:
        with open(html_path, "r", encoding="latin-1") as f:
            html_content = f.read()
    components.html(html_content, height=height, scrolling=False)


def _build_chooser_from_sidebar(prefix, title):
    import streamlit as st

    st.sidebar.markdown(f"### {title}")
    filter_expr = st.sidebar.text_area(
        "Row filter expression (on lmfit_summary columns)",
        value="", key=f"{prefix}_filter", height=300,
        help="Use a pandas.query-style boolean expression.",
    )
    field_to_check = st.sidebar.text_input(
        "Field to optimize on", value="redchi", key=f"{prefix}_field",
    )
    min_or_max = st.sidebar.radio(
        "Optimization direction", options=["min", "max"], index=0,
        key=f"{prefix}_minmax", horizontal=True,
    )
    return filter_expr, field_to_check, (min_or_max == "min")


def _make_graph_header(label, row, field_to_check):
    base = f"### {label} graphs"
    if row is None or not field_to_check:
        return base
    val = row.get(field_to_check, np.nan)
    null_val = row.get(f"null_{field_to_check}", np.nan)
    ls_iter_fap = row.get("LS_iter_fap", np.nan)
    prob_bicc = row.get("prob_bicc", np.nan)

    def fmt(v):
        try:
            if pd.isna(v):
                return "NA"
        except Exception:
            pass
        try:
            return f"{float(v):.3g}"
        except Exception:
            return str(v)

    return (f"{base} ({field_to_check}={fmt(val)}, null={fmt(null_val)})\n"
            f" ls_iter_fap={fmt(ls_iter_fap)}, prob_bicc={fmt(prob_bicc)}")


# -------------------------------------------------------
# Streamlit app entry point
# -------------------------------------------------------

def main():
    import streamlit as st

    st.set_page_config(page_title="Best-row chooser comparison", layout="wide")
    st.title("Best-row chooser comparison")

    for key in ("chooser1_df", "chooser2_df"):
        if key not in st.session_state:
            st.session_state[key] = None
    if "show_compare" not in st.session_state:
        st.session_state["show_compare"] = False

    st.sidebar.header("Paths")
    sol_dir = st.sidebar.text_input("Solution directory", value="/path/to/sol_dir")

    c1_filter, c1_field, c1_take_min = _build_chooser_from_sidebar("c1", "Chooser 1 config")
    c2_filter, c2_field, c2_take_min = _build_chooser_from_sidebar("c2", "Chooser 2 config")

    st.sidebar.markdown("---")
    choose_btn = st.sidebar.button("Choose solutions")
    compare_btn = st.sidebar.button("Compare solutions")

    # Example filter snippets
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filter expression examples:**")
    for snippet in [
        'candidate_method.str.contains("jitter")',
        '~candidate_method.str.contains("jitter")',
        "bin_flag > 0",
        "mass_flag_peri == 0",
        "bin_flag_Ftest == 1",
        "(Period_value < 0.9) | (Period_value > 1.1)",
    ]:
        st.sidebar.code(snippet, language="python")

    if choose_btn:
        if not os.path.isdir(sol_dir):
            st.error(f"Solution directory does not exist: {sol_dir}")
        else:
            with st.spinner("Collecting best rows for chooser 1..."):
                st.session_state["chooser1_df"] = collect_best_rows(
                    sol_dir, c1_filter, c1_field, c1_take_min)
            with st.spinner("Collecting best rows for chooser 2..."):
                st.session_state["chooser2_df"] = collect_best_rows(
                    sol_dir, c2_filter, c2_field, c2_take_min)
            st.success("Best rows collected.")

    if compare_btn:
        st.session_state["show_compare"] = True

    chooser1_df = st.session_state["chooser1_df"]
    chooser2_df = st.session_state["chooser2_df"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Chooser 1 results")
        if chooser1_df is not None and not chooser1_df.empty:
            st.dataframe(chooser1_df, use_container_width=True)
        else:
            st.info("No results yet for chooser 1.")
    with col2:
        st.subheader("Chooser 2 results")
        if chooser2_df is not None and not chooser2_df.empty:
            st.dataframe(chooser2_df, use_container_width=True)
        else:
            st.info("No results yet for chooser 2.")

    if not st.session_state["show_compare"]:
        return
    st.markdown("---")
    st.header("Compare solutions")

    if chooser1_df is None or chooser2_df is None:
        st.warning("Please run 'Choose solutions' first.")
        return
    if chooser1_df.empty and chooser2_df.empty:
        st.warning("Both choosers have empty results.")
        return

    stars1 = set(chooser1_df.index) if chooser1_df is not None else set()
    stars2 = set(chooser2_df.index) if chooser2_df is not None else set()
    union_stars = sorted(stars1 | stars2)

    status_rows = []
    for s in union_stars:
        in1, in2 = s in stars1, s in stars2
        p1 = chooser1_df.loc[s, "Period_value"] if in1 else np.nan
        p2 = chooser2_df.loc[s, "Period_value"] if in2 else np.nan
        delta_logP = (float(abs(np.log10(p1) - np.log10(p2)))
                      if (in1 and in2 and p1 > 0 and p2 > 0) else np.nan)
        status_rows.append({
            "star_name": s,
            "chooser1_has_solution": in1,
            "chooser2_has_solution": in2,
            "status": "both" if in1 and in2 else ("chooser1_only" if in1 else "chooser2_only"),
            "Period1": p1, "Period2": p2, "delta_logP": delta_logP,
        })

    status_df = pd.DataFrame(status_rows).set_index("star_name")

    def _highlight(row):
        if row["status"] == "both":
            return [""] * len(row)
        return ["background-color: red"] * len(row)

    st.subheader("Stars with solutions")
    styler = (status_df.style
              .apply(_highlight, axis=1)
              .background_gradient(subset=["delta_logP"], cmap="RdYlGn_r"))
    st.dataframe(styler, use_container_width=True)

    if "star_index" not in st.session_state:
        st.session_state.star_index = 0
    if st.session_state.star_index >= len(union_stars):
        st.session_state.star_index = 0

    nav1, nav2, nav3 = st.columns([1, 3, 1])
    with nav1:
        if st.button("⬆ Previous"):
            st.session_state.star_index = (st.session_state.star_index - 1) % len(union_stars)
    with nav3:
        if st.button("⬇ Next"):
            st.session_state.star_index = (st.session_state.star_index + 1) % len(union_stars)
    with nav2:
        selected_star = st.selectbox("Pick a star", union_stars,
                                     index=st.session_state.star_index)
        st.session_state.star_index = union_stars.index(selected_star)

    if selected_star:
        st.markdown(f"### Selected star: `{selected_star}`")
        c1_row = chooser1_df.loc[selected_star] if selected_star in chooser1_df.index else None
        c2_row = chooser2_df.loc[selected_star] if selected_star in chooser2_df.index else None

        colA, colB = st.columns(2)
        with colA:
            st.markdown("#### Chooser 1 solution")
            if c1_row is not None:
                st.write(c1_row.to_frame().T)
            else:
                st.info("No solution from chooser 1.")
        with colB:
            st.markdown("#### Chooser 2 solution")
            if c2_row is not None:
                st.write(c2_row.to_frame().T)
            else:
                st.info("No solution from chooser 2.")

        st.markdown("---")
        st.subheader("Associated graphs")
        gcols = st.columns(2)

        def _show_graphs(row, label):
            if row is None:
                st.info(f"No {label} solution.")
                return
            if "solution_id" not in row.index:
                st.error(f"{label} row has no solution_id.")
                return
            sid = int(row["solution_id"])
            for suffix in ("time_residuals", "phase_residuals"):
                path = os.path.join(sol_dir, selected_star, "lmfit_solutions",
                                    f"{selected_star}_sid-{sid}_{suffix}.html")
                st.markdown(f"**{label} – {suffix.replace('_', ' ')}**")
                _embed_html_file(path)

        with gcols[0]:
            st.markdown(_make_graph_header("Chooser 1", c1_row, c1_field))
            _show_graphs(c1_row, "Chooser 1")
        with gcols[1]:
            st.markdown(_make_graph_header("Chooser 2", c2_row, c2_field))
            _show_graphs(c2_row, "Chooser 2")


if __name__ == "__main__":
    main()
