import os
import glob
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components


# -------------------------------------------------------
# Selector helpers
# -------------------------------------------------------

def get_best_row(results_df, filter_expr, field_to_check, take_min):
    """
    Select the best row from results_df according to a free-form filter expression,
    field_to_check, and optimization direction (min / max).
    - filter_expr: a pandas.query() boolean expression on the columns.
    """

    if results_df is None or results_df.empty:
        print("Results DataFrame is empty.")
        return None

    df = results_df.copy()

    # ---------------------------------------------------------
    # 1. Get null_hyp value for bic/prob_bic logic
    # ---------------------------------------------------------
    null_field = None
    if field_to_check in df.columns and "candidate_method" in df.columns:
        null_mask = df["candidate_method"].str.contains("null_hyp_jitter", na=False)
        if null_mask.any():
            null_field = df.loc[null_mask, field_to_check].iloc[0]

    # ---------------------------------------------------------
    # 2. Apply user-defined filter expression (if provided)
    # ---------------------------------------------------------
    if filter_expr is not None and filter_expr.strip():
        try:
            # Make query() happy by flattening multi-line input
            clean_expr = " ".join(filter_expr.splitlines())
            df = df.query(clean_expr, engine="python")
        except Exception as e:
            print(f"Failed to apply filter expression '{filter_expr}': {e}")
            return None

    # ---------------------------------------------------------
    # 3. Check if any rows remain
    # ---------------------------------------------------------
    if df.empty:
        print(f"No rows remain after applying filters: {filter_expr!r}")
        return None

    # ---------------------------------------------------------
    # 4. Field-based extra filtering (bic / prob_bic logic)
    # ---------------------------------------------------------
    field = field_to_check

    if field not in df.columns:
        print(f"Field '{field}' not found in DataFrame columns.")
        return None

    if field == "bic" and null_field is not None:
        df = df[df[field] < null_field]

    # Remove NaNs from the optimization field
    df = df[df[field].notna()]

    if df.empty:
        print(f"No valid values in '{field}' after NaN removal.")
        return None

    # ---------------------------------------------------------
    # 5. Select min or max row based on field_to_check
    # ---------------------------------------------------------
    if take_min:
        idx = df[field].idxmin()
    else:
        idx = df[field].idxmax()

    return df.loc[idx]


def build_chooser_from_sidebar(prefix, title):
    st.sidebar.markdown(f"### {title}")

    filter_expr = st.sidebar.text_area(
        "Row filter expression (on lmfit_summary columns)",
        value="",
        key=f"{prefix}_filter",
        height=300,
        help=(
            "Use a pandas.query-style boolean expression, e.g.\n"
            "  candidate_method.str.contains(\"jitter\") & (bin_flag > 0)\n"
            "  mass_flag_peri == 0 & (Period_value < 0.9 | Period_value > 1.1)"
        ),
    )

    field_to_check = st.sidebar.text_input(
        "Field to optimize on",
        value="redchi",
        key=f"{prefix}_field",
    )

    min_or_max = st.sidebar.radio(
        "Optimization direction",
        options=["min", "max"],
        index=0,
        key=f"{prefix}_minmax",
        horizontal=True,
    )
    take_min = (min_or_max == "min")

    return filter_expr, field_to_check, take_min


# -------------------------------------------------------
# Data processing helpers
# -------------------------------------------------------

def collect_best_rows(sol_dir, filter_expr, field_to_check, take_min):
    """
    Iterate over rv_dir/{star_name}_CCF_RVs.csv,
    for each star read sol_dir/{star_name}/lmfit_summary.csv,
    and pick best row using get_best_row with the given parameters.
    Also attach the matching null-hyp value for field_to_check.
    """
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

        # -------------------------------------------------
        # Find corresponding null_hyp value (with/without jitter)
        # -------------------------------------------------
        null_val = np.nan
        if "candidate_method" in df.columns and field_to_check in df.columns:
            null_mask = df["candidate_method"].str.contains("null_hyp", na=False)
            if null_mask.any():
                null_df = df[null_mask]

                # Decide which null to prefer based on jitter in best_row
                best_has_jitter = False
                if "candidate_method" in best_row.index:
                    cm = str(best_row["candidate_method"])
                    best_has_jitter = "jitter" in cm

                if best_has_jitter:
                    # Prefer null_hyp with jitter, fallback to any null_hyp
                    sub = null_df[null_df["candidate_method"].str.contains("jitter", na=False)]
                    if not sub.empty:
                        null_val = sub[field_to_check].iloc[0]
                    else:
                        null_val = null_df[field_to_check].iloc[0]
                else:
                    # Prefer null_hyp without jitter, fallback to any null_hyp
                    sub = null_df[~null_df["candidate_method"].str.contains("jitter", na=False)]
                    if not sub.empty:
                        null_val = sub[field_to_check].iloc[0]
                    else:
                        null_val = null_df[field_to_check].iloc[0]

        best_row[f"null_{field_to_check}"] = null_val

        rows.append(best_row)

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    result_df = result_df.set_index("star_name").sort_index()
    return result_df


def embed_html_file(html_path, height=1000, key=None):
    """
    Embed a local HTML file (e.g. Plotly/Bokeh export) in Streamlit
    without internal scrollbars. Increase height so the whole graph fits.
    """
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



def make_graph_header(label, row, field_to_check):
    base = f"### {label} graphs"
    if row is None or not field_to_check:
        return base

    val = row.get(field_to_check, np.nan)
    null_val = row.get(f"null_{field_to_check}", np.nan)
    ls_iter_fap = row.get("LS_iter_fap", np.nan)
    prob_bicc = row.get(f"prob_bicc", np.nan)
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

    return f"{base} ({field_to_check}={fmt(val)}, null={fmt(null_val)})\n ls_iter_fap={fmt(ls_iter_fap)}, prob_bicc={fmt(prob_bicc)}"


# -------------------------------------------------------
# Streamlit app
# -------------------------------------------------------

def main():
    st.set_page_config(page_title="Best-row chooser comparison", layout="wide")

    st.title("Best-row chooser comparison")

    # --- Session state init ---
    if "chooser1_df" not in st.session_state:
        st.session_state["chooser1_df"] = None
    if "chooser2_df" not in st.session_state:
        st.session_state["chooser2_df"] = None
    if "show_compare" not in st.session_state:
        st.session_state["show_compare"] = False

    # --- Sidebar inputs ---
    st.sidebar.header("Paths")

    sol_dir = st.sidebar.text_input(
        "Solution directory (sol_dir)",
        value="/path/to/sol_dir",
    )

    # Per-chooser filter + optimization settings
    chooser1_filter, chooser1_field, chooser1_take_min = build_chooser_from_sidebar(
        "c1", "Chooser 1 config"
    )
    chooser2_filter, chooser2_field, chooser2_take_min = build_chooser_from_sidebar(
        "c2", "Chooser 2 config"
    )

    st.sidebar.markdown("---")
    choose_btn = st.sidebar.button("Choose solutions")
    compare_btn = st.sidebar.button("Compare solutions")

    # --- Example filter expressions (equivalent to old BestRowConfig flags) ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filter expression examples** (copy & edit):")

    st.sidebar.code(
        "# Only solutions that used jitter\n"
        "candidate_method.str.contains(\"jitter\")",
        language="python",
    )

    st.sidebar.code(
        "# Only solutions without jitter\n"
        "~candidate_method.str.contains(\"jitter\")",
        language="python",
    )

    st.sidebar.code(
        "# Require bin_flag > 0\n"
        "bin_flag > 0",
        language="python",
    )

    st.sidebar.code(
        "# Require Roche constraint satisfied\n"
        "mass_flag_peri == 0",
        language="python",
    )

    st.sidebar.code(
        "# Require F-test passed\n"
        "bin_flag_Ftest == 1",
        language="python",
    )

    st.sidebar.code(
        "# Remove ~1-day aliases\n"
        "(Period_value < 0.9) | (Period_value > 1.1)",
        language="python",
    )

    st.sidebar.code(
        "# Remove ~0.5-day aliases only for PDC solutions\n"
        "~(((Period_value > 0.45) & (Period_value < 0.55)) "
        "& candidate_method.str.contains(\"PDC\"))",
        language="python",
    )

    st.sidebar.code(
        " candidate_method.str.contains('jitter')\n"
        "& (mass_flag_peri == 0)\n"
        "& (bin_flag > 0)\n"
        "& (prob_bic > 0.99))",
        language="python",
    )

    # --- Choose solutions ---
    if choose_btn:
        if not os.path.isdir(sol_dir):
            st.error(f"Solution directory does not exist: {sol_dir}")
        else:
            with st.spinner("Collecting best rows for chooser 1..."):
                df1 = collect_best_rows(
                    sol_dir, chooser1_filter, chooser1_field, chooser1_take_min
                )
                st.session_state["chooser1_df"] = df1

            with st.spinner("Collecting best rows for chooser 2..."):
                df2 = collect_best_rows(
                    sol_dir, chooser2_filter, chooser2_field, chooser2_take_min
                )
                st.session_state["chooser2_df"] = df2

            st.success("Best rows collected for both choosers.")

    # --- Compare solutions toggle ---
    if compare_btn:
        st.session_state["show_compare"] = True

    chooser1_df = st.session_state["chooser1_df"]
    chooser2_df = st.session_state["chooser2_df"]

    # --- Show per-chooser tables ---
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

    # --- Compare view ---
    if st.session_state["show_compare"]:
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

        # Build status table with period & ΔlogP info
        status_rows = []
        for s in union_stars:
            in1 = s in stars1
            in2 = s in stars2

            if in1 and in2:
                status = "both"
            elif in1:
                status = "chooser1_only"
            else:
                status = "chooser2_only"

            # Periods from each chooser (if exist)
            p1 = chooser1_df.loc[s, "Period_value"] if in1 else np.nan
            p2 = chooser2_df.loc[s, "Period_value"] if in2 else np.nan

            # ΔlogP = |log10(P1) - log10(P2)| when both present & > 0
            if in1 and in2 and p1 > 0 and p2 > 0:
                delta_logP = float(abs(np.log10(p1) - np.log10(p2)))
            else:
                delta_logP = np.nan

            status_rows.append(
                {
                    "star_name": s,
                    "chooser1_has_solution": in1,
                    "chooser2_has_solution": in2,
                    "status": status,
                    "Period1": p1,
                    "Period2": p2,
                    "delta_logP": delta_logP,
                }
            )

        status_df = pd.DataFrame(status_rows).set_index("star_name")

        def highlight_row(row):
            # Red background for stars where only one chooser found a solution
            if row["status"] == "both":
                return [""] * len(row)
            else:
                return ["background-color: red"] * len(row)

        st.subheader(
            "Stars with solutions "
            "(red row = only one chooser found a solution, "
            "delta_logP green = similar periods, red = very different)"
        )

        styler = (
            status_df.style
            .apply(highlight_row, axis=1)
            .background_gradient(
                subset=["delta_logP"],
                cmap="RdYlGn_r",  # small diff → green, large diff → red
            )
        )

        st.dataframe(styler, use_container_width=True)

        # --- Star navigation state ---
        if "star_index" not in st.session_state:
            st.session_state.star_index = 0

        # Ensure index is valid
        if st.session_state.star_index >= len(union_stars):
            st.session_state.star_index = 0

        current_star = union_stars[st.session_state.star_index]

        # Layout: Previous ← , selector, Next →
        nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])

        with nav_col1:
            if st.button("⬆ Previous"):
                st.session_state.star_index = (st.session_state.star_index - 1) % len(union_stars)

        with nav_col3:
            if st.button("⬇ Next"):
                st.session_state.star_index = (st.session_state.star_index + 1) % len(union_stars)

        # Dropdown selector (also updates index)
        with nav_col2:
            selected_star = st.selectbox(
                "Pick a star to inspect",
                options=union_stars,
                index=st.session_state.star_index,
            )
            # If user chose from dropdown → sync index
            st.session_state.star_index = union_stars.index(selected_star)

        if selected_star:
            st.markdown(f"### Selected star: `{selected_star}`")

            c1_row = chooser1_df.loc[selected_star] if selected_star in chooser1_df.index else None
            c2_row = chooser2_df.loc[selected_star] if selected_star in chooser2_df.index else None

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("#### Chooser 1 solution")
                if c1_row is not None:
                    st.write(c1_row.to_frame().T)
                else:
                    st.info("No solution from chooser 1 for this star.")

            with col_b:
                st.markdown("#### Chooser 2 solution")
                if c2_row is not None:
                    st.write(c2_row.to_frame().T)
                else:
                    st.info("No solution from chooser 2 for this star.")

            # Plot HTML graphs for each available solution
            st.markdown("---")
            st.subheader("Associated graphs")

            graph_cols = st.columns(2)

            # Helper to show graphs for one row
            def show_graphs_for_row(row, label):
                if row is None:
                    st.info(f"No {label} solution to show.")
                    return

                if "solution_id" not in row.index:
                    st.error(f"{label} row has no 'solution_id' column.")
                    return

                solution_id = int(row["solution_id"])

                time_path = os.path.join(
                    sol_dir,
                    selected_star,
                    "lmfit_solutions",
                    f"{selected_star}_sid-{solution_id}_time_residuals.html"
                )
                phase_path = os.path.join(
                    sol_dir,
                    selected_star,
                    "lmfit_solutions",
                    f"{selected_star}_sid-{solution_id}_phase_residuals.html"
                )

                st.markdown(f"**{label} – time series with residuals**")
                embed_html_file(time_path)

                st.markdown(f"**{label} – phase-folded with residuals**")
                embed_html_file(phase_path)

            with graph_cols[0]:
                st.markdown(make_graph_header("Chooser 1", c1_row, chooser1_field))
                show_graphs_for_row(c1_row, "Chooser 1")

            with graph_cols[1]:
                st.markdown(make_graph_header("Chooser 2", c2_row, chooser2_field))
                show_graphs_for_row(c2_row, "Chooser 2")


if __name__ == "__main__":
    main()
