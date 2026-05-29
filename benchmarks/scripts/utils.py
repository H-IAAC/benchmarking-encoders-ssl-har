from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import graphviz
import plotly.graph_objects as go
import plotly.express as px


def prepare_dataframe(
    df: pd.DataFrame,
    query_str: str,
    sort_columns: Optional[list] = None,
    query_vars: Optional[dict] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Filter and sort a DataFrame based on a query string and sort columns

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter and sort.
    query_str : str
        The query string to filter the DataFrame (in the format of a pandas
        query).
    sort_columns : Optional[list], optional
        The list of columns to sort the DataFrame by, after filtering, by
        default None.
    query_vars : Optional[dict], optional
        A dictionary of variables to pass to the query string, by default None.
    verbose : bool, optional
        Whether to print the query string, by default False.

    Returns
    -------
    pd.DataFrame
        The filtered and sorted DataFrame.
    """
    if not query_vars:
        query_vars = dict()

    # Pass the variables using local_dict in the query function
    result_df = df.query(query_str, local_dict=query_vars)

    if sort_columns:
        result_df = result_df.sort_values(by=sort_columns).reset_index(
            drop=True
        )

    if verbose:
        print(
            f" -> Querying dataframe ({len(df)} rows) with query: '{query_str}'"
        )
        print(f" -> In-use Variables: {query_vars}")
        print(f" <- Resulting dataframe has {len(result_df)} rows.")
        if sort_columns:
            print(f" <- Sorted by columns: {sort_columns}")

    return result_df


def prepare_experiment_df(
    df: pd.DataFrame,
    select_backbones: Optional[List[str]] = None,
    select_tsk_pretext: Optional[List[str]] = None,
    select_d_pretext: Optional[List[str]] = None,
    select_head_pred: Optional[List[str]] = None,
    select_ft_strategy: Optional[List[str]] = None,
    select_tsk_target: Optional[List[str]] = None,
    select_d_target: Optional[List[str]] = None,
    select_frac_dtarget: Optional[List[float]] = None,
    select_metric_target: Optional[List[str]] = None,
    sort_columns: Optional[List[str]] = None,
    verbose: bool = False,
):
    """Filter a DataFrame based on the provided parameters

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    select_backbones : Optional[List[str]], optional
        The list of backbones to filter by, by default None.
    select_tsk_pretext : Optional[List[str]], optional
        The list of task pretexts to filter by, by default None.
    select_d_pretext : Optional[List[str]], optional
        The list of dataset pretexts to filter by, by default None.
    select_head_pred : Optional[List[str]], optional
        The list of head predictors to filter by, by default None.
    select_ft_strategy : Optional[List[str]], optional
        The list of finetuning strategies to filter by, by default None.
    select_tsk_target : Optional[List[str]], optional
        The list of target tasks to filter by, by default None.
    select_d_target : Optional[List[str]], optional
        The list of target datasets to filter by, by default None.
    select_frac_dtarget : Optional[List[float]], optional
        The list of target dataset fractions to filter by, by default None.
    select_metric_target : Optional[List[str]], optional
        The list of target metrics to filter by, by default None.
    sort_columns : Optional[List[str]], optional
        The list of columns to sort the DataFrame by, by default None.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame.
    """
    query_str = ""
    query_vars = dict()

    if select_backbones:
        query_str += f"backbone in @select_backbones and "
        query_vars["select_backbones"] = select_backbones
    if select_tsk_pretext:
        query_str += "tsk_pretext in @select_tsk_pretext and "
        query_vars["select_tsk_pretext"] = select_tsk_pretext
    if select_d_pretext:
        query_str += "d_pretext in @select_d_pretext and "
        query_vars["select_d_pretext"] = select_d_pretext
    if select_head_pred:
        query_str += "head_pred in @select_head_pred and "
        query_vars["select_head_pred"] = select_head_pred
    if select_ft_strategy:
        query_str += "ft_strategy in @select_ft_strategy and "
        query_vars["select_ft_strategy"] = select_ft_strategy
    if select_tsk_target:
        query_str += "tsk_target in @select_tsk_target and "
        query_vars["select_tsk_target"] = select_tsk_target
    if select_d_target:
        query_str += "d_target in @select_d_target and "
        query_vars["select_d_target"] = select_d_target
    if select_frac_dtarget:
        query_str += "frac_dtarget in @select_frac_dtarget and "
        query_vars["select_frac_dtarget"] = select_frac_dtarget
    if select_metric_target:
        query_str += "metric_target in @select_metric_target and "
        query_vars["select_metric_target"] = select_metric_target

    if len(query_str) == 0:
        print("No filters applied.")
        return df

    # Remove the trailing " and " from the query string
    query_str = query_str[:-5]
    res = prepare_dataframe(
        df,
        query_str,
        query_vars=query_vars,
        sort_columns=sort_columns,
        verbose=verbose,
    )
    return res


# Function to resize and center the LaTeX table
def resize_latex_table(latex_output: str):
    """
    Adds a \\resizebox command to the LaTeX table to resize it to the column
    width and centers it.
    """
    row_index = latex_output.find("\\begin{tabular}")
    latex_output = (
        latex_output[:row_index]
        + "\\centering\n\\resizebox{\\columnwidth}{!}{\n"
        + latex_output[row_index:]
    )
    row_index = latex_output.find("\\end{tabular}")
    latex_output = (
        latex_output[: row_index + 14] + "}\n" + latex_output[row_index + 14 :]
    )
    return latex_output


def add_summary_row(
    df: pd.DataFrame, variant_columns: list, agg_func: str | list = "mean"
):
    if isinstance(agg_func, str):
        agg_func = [agg_func]  # Ensure that agg_func is always a list

    # Prepare empty rows for summary (matching number of aggregation functions and columns)
    lines = np.full((len(agg_func), len(df.columns)), np.nan, dtype=object)

    for variant_col in variant_columns:
        if variant_col not in df.columns:
            raise ValueError(f"Column {variant_col} not found in DataFrame.")

        col_ind = df.columns.get_loc(variant_col)

        for row_ind, agg_f in enumerate(agg_func):
            variant_res = df[variant_col]
            lines[row_ind][col_ind] = variant_res.agg(agg_f)

    # Convert np.nan rows to a DataFrame to avoid issues with types and missing values
    summary_df = pd.DataFrame(lines, columns=df.columns)

    # Optionally, fill NaN values with a placeholder (e.g., 0, empty string, or None)
    summary_df = summary_df.fillna(
        ""
    )  # or .fillna(0) or .fillna('None') based on your requirement

    # Append the summary rows to the original DataFrame without dropping NaNs
    df = pd.concat([df, summary_df], ignore_index=True)

    return df


# Function to add a \midrule before the summary row ("Média") in the LaTeX string
def add_midrule_before_summary(latex_output: str):
    """
    Adds a \\midrule before the summary row ("Média") in the LaTeX table.
    """
    # Find the index of the line containing "Média"
    summary_row_index = latex_output.find("bottomrule")

    # Find the start of the line containing "Média"
    line_start_index = latex_output.rfind("\n", 0, summary_row_index)
    line_start_index = latex_output.rfind("\n", 0, line_start_index)

    # Insert \midrule before the line containing "Média"
    latex_output = (
        latex_output[:line_start_index]
        + "\n\\midrule"
        + latex_output[line_start_index:]
    )

    return latex_output


# Generic function to calculate grouped results, with customizable variant
# column and multiple variants
def calculate_scenario_variants_results(
    df: pd.DataFrame,
    group_columns: List[str],
    variant_column: str,
    variants: List[str],
    metric_column: str = "metric",
    scenario_name: str = "Scenario",
    variants_name: str = "Variants",
) -> pd.DataFrame:
    """Generate a DataFrame with grouped results for each group.

    Parameters
    ----------
    result_df : pd.DataFrame
        The DataFrame containing the results.
    group_columns : List[str]
        The columns to group by.
    variant_column : str
        The column containing the variants
    variants : List[str]
        The variants to calculate the results for.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the grouped results.
    """
    dfs = []
    for _, sub_df in df.groupby(group_columns):
        line = sub_df.iloc[0].to_dict()
        line = {(scenario_name, k): v for k, v in line.items()}

        results = []
        for variant in variants:
            r = sub_df[sub_df[variant_column] == variant][metric_column]
            r = np.nan if r.empty else r.mean()
            r = float(r)
            results.append(r)
            line[(variants_name, variant)] = float(r)

        # Only calculate differences if there are exactly two variants
        if len(variants) == 2:
            line[(variants_name, "Diff")] = float(results[0]) - float(
                results[1]
            )
        else:
            line[(variants_name, "Mean")] = np.mean(
                [float(r) for r in results if not np.isnan(float(r))]
            )

        dfs.append(line)

    return pd.DataFrame(dfs)


# Modified LaTeX output generation function to include table resizing
def generate_latex_output(
    results_df, caption, label, resize_table=True, add_midrule=True
):
    """
    Generate LaTeX output from the results dataframe.
    """
    cols = results_df.columns
    new_cols = []
    for col in cols:
        if "_" in col[1]:
            parts = col[1].split("_")
            new_cols.append((col[0], "$" + parts[0] + "_{" + parts[1] + "}$"))
        else:
            new_cols.append(col)

    results_df.columns = pd.MultiIndex.from_tuples(new_cols)

    latex_output = results_df.to_latex(
        multicolumn=True,
        multirow=True,
        index=False,
        escape=False,
        bold_rows=True,
        float_format="%.2f",
        caption=caption,
        label=label,
        position="!hpbt",
        na_rep="",
    )

    # Resize and center the LaTeX table
    if resize_table:
        latex_output = resize_latex_table(latex_output)

    # Optionally add \midrule before summary row
    if add_midrule:
        latex_output = add_midrule_before_summary(latex_output)

    return latex_output


def read_aggregated_results(path: Path) -> pd.DataFrame:
    """Read the aggregated results from the CSV file.

    Parameters
    ----------
    path : Path
        The path to the CSV file containing the aggregated results.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the aggregated results. It will have the
        following columns:
        - backbone (str)
        - tsk_pretext (str)
        - d_pretext (str)
        - head_pred (str)
        - ft_strategy (str)
        - tsk_target (str)
        - d_target (str)
        - frac_dtarget (float)
        - metric_target (str)
        - metric (float)
    """
    df = pd.read_csv(path)
    values = []
    for row_id, row in df.iterrows():
        i = str(row["model_id"])

        desc = dict()
        if "+" in i:
            desc["backbone"] = i.split("+")[1]
            desc["tsk_pretext"] = i.split("+")[0]
        else:
            desc["backbone"] = i
            desc["tsk_pretext"] = "Supervised"

        desc["d_pretext"] = row["dataset"]
        desc["head_pred"] = "MLP"

        if "(Freeze)" in desc["backbone"]:
            # Remove " (Freeze)" from the backbone"
            desc["backbone"] = desc["backbone"].replace(" (Freeze)", "")
            desc["ft_strategy"] = "Freeze"
        else:
            desc["ft_strategy"] = "Full Finetune"

        desc["tsk_target"] = row["pipeline_id"]
        desc["d_target"] = row["dataset"]
        desc["frac_dtarget"] = float(row["percentage"])
        desc["metric_target"] = row["used_metric"]
        desc["metric"] = float(row["metric"])  # f'{row["metric"] * 100:.2f}'

        values.append(desc)

    df = pd.DataFrame(values)
    df["frac_dtarget"] = df["frac_dtarget"].astype(float)
    df["metric"] = df["metric"].astype(float)
    return df


def calculate_variant_wilcoxon(
    result_df: pd.DataFrame,
    scenario_columns: List[str],
    threshold: float = 0.05,
    column_name: str = "metric",
    double_sided: bool = False,
    apply_bonferroni: bool = False,
) -> pd.DataFrame:
    """
    Perform Wilcoxon signed-rank tests between experimental variants with optional
    Bonferroni correction as described in Napierala (2012).
    
    The Bonferroni correction divides the significance threshold (α) by the number of
    comparisons to maintain family-wise error rate.
    Reference: NAPIERALA, Matthew A. What is the Bonferroni correction?. Aaos now, p. 40-41, 2012.

    
    Parameters
    ----------
    result_df : pd.DataFrame
        Input dataframe with experimental results
    scenario_columns : List[str]
        Columns that define the experimental variants
    threshold : float, optional
        Initial significance threshold (α), by default 0.05
    column_name : str, optional
        Name of column containing metric values, by default "metric"
    double_sided : bool, optional
        Whether to do all pairwise comparisons, by default False
    apply_bonferroni : bool, optional
        Whether to apply Bonferroni correction, by default False
    
    Returns
    -------
    pd.DataFrame
        Results with raw p-values, adjusted threshold, and significance
    """

    # Group the data by the scenario columns
    all_variants = list(result_df.groupby(scenario_columns).groups.items())

    values = []
    raw_p_values = []
    for i in range(len(all_variants)):
        v1_name, v1_indices = all_variants[i]
        v1_df = result_df.loc[v1_indices]
        result_v1 = np.array(v1_df[column_name].values)

        if double_sided:
            start_range = 0
        else:
            start_range = i + 1

        for j in range(start_range, len(all_variants)):
            if i == j:
                continue
            v2_name, v2_indices = all_variants[j]
            v2_df = result_df.loc[v2_indices]
            result_v2 = np.array(v2_df[column_name].values)

            # Only proceed if both variants have the same number of results
            if len(result_v1) != len(result_v2):
                print(
                    f"WARNING: number of samples for variants {v1_name} and {v2_name} do not match."
                )
                continue

            # Perform the Wilcoxon test between the two result sets
            w_test = wilcoxon(
                result_v1, result_v2, alternative="two-sided", method="auto"
            )
            raw_p_values.append(w_test.pvalue)
            # print(f'v1_name: {v1_name}, type v1_name: {type(v1_name)}, type v2_name: {type(v2_name)},  v2_name: {v2_name}, p-value: {w_test.pvalue:.4f}')
            # Append the results to the list
            values.append(
                {
                    # "Variant 1": " + ".join(v1_name) if not isinstance(v1_name, str) else v1_name,  # type: ignore
                    # "Variant 2": " + ".join(v2_name) if not isinstance(v1_name, str) else v2_name,  # type: ignore
                    "Variant 1": " + ".join(map(str, v1_name)) if not isinstance(v1_name, str) else v1_name,
                    "Variant 2": " + ".join(map(str, v2_name)) if not isinstance(v2_name, str) else v2_name,
                    "Variant 1 Mean": np.mean(result_v1),
                    "Variant 2 Mean": np.mean(result_v2),
                    "Variant 1 stdev": np.std(result_v1),
                    "Variant 2 stdev": np.std(result_v2),
                    "P-Value": w_test.pvalue,
                    "Significant": (
                        True if w_test.pvalue < threshold else False
                    ),
                }
            )

    # Create DataFrame
    results_df = pd.DataFrame(values)
    
    # Apply Bonferroni correction per Napierala (2012)
    if apply_bonferroni and len(raw_p_values) > 0:
        n_comparisons = len(raw_p_values)
        
        # Core Bonferroni adjustment: α/m
        adjusted_threshold = threshold / n_comparisons
        print(f"Bonferroni correction applied: α/{n_comparisons} = {adjusted_threshold:.6f}")

        
        # For reporting, show both adjusted p-values and threshold
        results_df["Adjusted P-Value"] = results_df["P-Value"] * n_comparisons
        
        # Determine significance using adjusted threshold
        results_df["Significant"] = results_df["P-Value"] <= adjusted_threshold
        
    else:
        results_df["Adjusted P-Value"] = 'not applicable'
        results_df["Significant"] = results_df["P-Value"] <= threshold

    # Reorder columns
    col_order = [
        "Variant 1", "Variant 2",
        "Variant 1 Mean", "Variant 2 Mean", 
        "Variant 1 stdev", "Variant 2 stdev",
        "P-Value", "Adjusted P-Value",
        "Significant"
    ]
    
    results_df = results_df[[c for c in col_order if c in results_df.columns]]

    return results_df.sort_values(by="P-Value").reset_index(drop=True)


def create_precedence_graph(
    wilcoxon_df: pd.DataFrame, threshold: float = 0.05, show_stdev=False,apply_bonferroni=False
) -> graphviz.Digraph:
    """
    Create a precedence graph based on Wilcoxon test results, where an edge points to the worse variant
    if the test fails (p-value below the threshold), nodes include the mean values, and nodes without
    incoming edges are colored green.

    If a wilcoxon test with Bonferroni correction was used, it will automatically find it

    Parameters
    ----------
    wilcoxon_df : pd.DataFrame
        DataFrame containing Wilcoxon test results with columns ['Variant 1', 'Variant 2', 'Variant 1 Mean', 'Variant 2 Mean', 'P-Value'].
    threshold : float
        P-value threshold below which a directed edge will be added (default is 0.05).

    Returns
    -------
    graphviz.Digraph
        A directed graph in Graphviz format showing the precedence between variants.
    """
    if len(wilcoxon_df) == 0:
        print("Wilcoxon dataframe is empty! Do you have more than one variant?")

    # Initialize the Graphviz directed graph
    dot = graphviz.Digraph(
        comment="Precedence Graph", graph_attr={"rankdir": "TB"}
    )

    # Track nodes that have incoming edges and outgoing edges
    incoming_edges = set()
    outgoing_edges = {}

    # Create a set to track the variants we've added as nodes
    added_variants = set()

    # Iterate through the Wilcoxon test results
    for _, row in wilcoxon_df.iterrows():
        # Determine which p-value to use
        if apply_bonferroni:
            if _ == 0:
                print("Using adjusted p-value")
            p_value = row["Adjusted P-Value"]
        else:
            if _ == 0:
                print("Using raw p-value")
            p_value = row["P-Value"]
        # p_value = row["Adjusted P-Value"] if "Adjusted P-Value" in row else row["Raw P-Value"]

        variant1 = row["Variant 1"]
        variant2 = row["Variant 2"]
        mean1 = row["Variant 1 Mean"]
        mean2 = row["Variant 2 Mean"]
        stdev1 = row["Variant 1 stdev"]
        stdev2 = row["Variant 2 stdev"]

        # Add nodes with mean values if not already added
        if variant1 not in added_variants:
            if show_stdev:
                dot.node(
                    variant1,
                    f"{variant1}\nMean Acc.\n {mean1:.2f} ± {stdev1:.2f}",
                )
            else:
                dot.node(variant1, f"{variant1}\nMean: {mean1:.2f}")
            added_variants.add(variant1)
        if variant2 not in added_variants:
            if show_stdev:
                dot.node(
                    variant2,
                    f"{variant2}\nMean Acc.\n {mean2:.2f} ± {stdev2:.2f}",
                )
            else:
                dot.node(variant2, f"{variant2}\nMean: {mean2:.2f}")
            added_variants.add(variant2)

        # Check if the p-value is below the threshold
        if p_value < threshold:
            if mean1 > mean2:
                # Variant 1 is better, point to Variant 2 with p-value
                dot.edge(variant1, variant2, label=f"p={p_value:.3f}")
                incoming_edges.add(variant2)
                outgoing_edges[variant1] = outgoing_edges.get(variant1, 0) + 1
            else:
                # Variant 2 is better, point to Variant 1 with p-value
                dot.edge(variant2, variant1, label=f"p={p_value:.3f}")
                incoming_edges.add(variant1)
                outgoing_edges[variant2] = outgoing_edges.get(variant2, 0) + 1

    # Set nodes without incoming edges to green color
    for variant in added_variants:
        if variant not in incoming_edges:
            dot.node(variant, style="filled", fillcolor="#B6FFA1")  # "#00CC96")

    # Set node ranks based on the number of outgoing edges
    for variant, out_count in outgoing_edges.items():
        dot.attr("node", rank=f"{out_count}")

    return dot


def plot_performance_matrix(
    df,
    row_name: str,
    col_name: str,
    metric_name: str = "metric",
    agg_func: str = "mean",
    sort_by: str = "mean",
    colorscale: str = "Viridis",
    text_size: int = 12,
):
    pivot_df = df.pivot_table(
        index=row_name, columns=col_name, values=metric_name, aggfunc=agg_func
    )
    pivot_df["max"] = pivot_df.max(axis=1)
    pivot_df["mean"] = pivot_df.mean(axis=1)
    pivot_df["min"] = pivot_df.min(axis=1)
    pivot_df = pivot_df.sort_values(by=sort_by, ascending=False)
    pivot_df = pivot_df * 100

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_df.values,
            x=list(pivot_df.columns),
            y=list(pivot_df.index),
            colorscale=colorscale,
            showscale=True,
            text=pivot_df.values,
            texttemplate="%{text:.2f}",
            hoverinfo="text",
            xgap=1,
            ygap=1,
        )
    )

    separator_position = pivot_df.shape[1] - 3
    fig.add_vline(
        x=separator_position - 0.5,
        line_width=2,
        line_dash="dash",
        line_color="black",
    )

    if isinstance(col_name, list):
        col_name = ", ".join(col_name)
    if isinstance(row_name, list):
        row_name = ", ".join(row_name)

    fig.update_layout(
        xaxis_title=col_name,
        yaxis_title=row_name,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            side="top",
            tickangle=-90,
            tickfont=dict(family="Times New Roman", size=text_size),
        ),
        yaxis=dict(
            tickfont=dict(family="Times New Roman", size=text_size),
        ),
        font=dict(family="Times New Roman", size=text_size),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # fig.update_traces(
    #     hoverongaps=False,
    #     zmin=pivot_df.values.min(),
    #     zmax=pivot_df.values.max(),
    #     xgap=1,
    #     ygap=1,
    # )

    return fig


def plot_comparison_bar(
    df,
    bar_column,  # Column to be used for unique bar categories (was tsk_target)
    bar_group_column,  # Column to group bars by (was percentages)
    bar_color_column,  # Column to define bar colors (was backbones)
    bar_pattern_column=None,  # Column to define bar patterns (was strategies)
    bar_group_column_text=None,  # Order of the bar groups
    colors=None,  # List of colors corresponding to the unique values in bar_column
    patterns=None,  # List of patterns corresponding to the unique values in bar_pattern_column
    result_column="result",  # Column for the y-axis result values
    metric_name="Metric",  # Name for the y-axis title
    bar_width=0.1,  # Width of the bars
    fig_width=1200,  # Width of the figure
    fig_height=600,  # Height of the figure
    title=None,  # Title of the plot
    title_y_pos=None,  # Vertical position of the title
    filename=None,  # Optional: File name to save the plot
    xaxis_title="Group",  # Optional: Name for the x-axis title
    yaxis_title=None,  # Optional: Name for the y-axis title,
    error_bars=True,  # Optional: Show error bars
    y_range=None,  # Optional: Minimum value for the y-axis
):
    # Default colors if not provided by the user
    if colors is None:
        colors = px.colors.qualitative.Plotly

    # Default patterns if not provided by the user
    if patterns is None:
        patterns = ["", "/", "x", "+"]

    if yaxis_title is None:
        yaxis_title = metric_name

    if title_y_pos is None:
        if title is None:
            title_y_pos = 0.8
        else:
            title_y_pos = 0.95

    fig = go.Figure()

    # Retrieve unique values from the respective columns
    bar_values = sorted(df[bar_column].unique())
    bar_groups = sorted(df[bar_group_column].unique())
    bar_colors = sorted(df[bar_color_column].unique())
    if bar_pattern_column:
        bar_patterns = sorted(df[bar_pattern_column].unique())
    else:
        bar_patterns = [""]

    # Loop through the combinations to create the bars
    for group_idx, bar_group in enumerate(bar_groups):
        for value_idx, bar_value in enumerate(bar_values):
            for color_idx, bar_color in enumerate(bar_colors):
                for pattern_idx, bar_pattern in enumerate(bar_patterns):

                    # Filter the dataframe based on the combination of values
                    sub_df = df[
                        (df[bar_group_column] == bar_group)
                        & (df[bar_column] == bar_value)
                        & (df[bar_color_column] == bar_color)
                    ]
                    if bar_pattern_column:
                        sub_df = sub_df[
                            sub_df[bar_pattern_column] == bar_pattern
                        ]

                    if sub_df.empty:
                        continue

                    if bar_group_column_text:
                        x = sub_df[bar_group_column_text].iloc[0]
                    else:
                        x = str(bar_group)

                    # Extract values for plotting
                    y = (
                        sub_df[result_column].astype(float).mean()
                    )  # Mean result
                    y_std = (
                        sub_df[result_column].astype(float).std()
                    )  # Standard deviation
                    color = colors[
                        value_idx % len(colors)
                    ]  # Color based on bar_value
                    pattern = patterns[
                        pattern_idx % len(patterns)
                    ]  # Pattern based on bar_pattern
                    name = f"{bar_value} {bar_color}"  # Bar label
                    if bar_pattern_column:
                        name += f" ({bar_pattern})"

                    if pd.isna(y):
                        continue

                    if error_bars:
                        error_y = dict(
                            type="data",
                            array=[y_std],
                            visible=True,
                            color="black",
                        )
                    else:
                        error_y = None

                    # Assign unique offsetgroup for each combination
                    unique_offsetgroup = (
                        f"{value_idx}-{color_idx}-{pattern_idx}"
                    )

                    fig.add_trace(
                        go.Bar(
                            x=[x],  # Categorical x value
                            y=[y],  # Mean result
                            name=name,
                            marker_color=color,
                            marker_pattern_shape=pattern,
                            error_y=error_y,
                            marker_pattern_fillmode="replace",
                            marker_pattern_solidity=0.75,
                            width=bar_width,
                            offsetgroup=unique_offsetgroup,  # Ensure bars are uniquely offset
                            showlegend=False,
                        )
                    )

                    # Add legend entry for each unique combination on the first group only
                    if group_idx == 0:
                        fig.add_trace(
                            go.Bar(
                                x=[None],
                                y=[None],
                                name=name,
                                marker_color=color,
                                marker_pattern_shape=pattern,
                                marker_pattern_fillmode="replace",
                                marker_pattern_solidity=0.75,
                                width=bar_width,
                                offsetgroup=unique_offsetgroup,
                                showlegend=True,
                            )
                        )

    # Update layout with axis titles and legend
    fig.update_layout(
        barmode="group",  # Group bars side by side
        xaxis_title=xaxis_title,  # Generic name for x-axis
        font=dict(family="Times New Roman", size=14),
        legend=dict(
            font=dict(family="Times New Roman", size=14),
            title="Models",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        bargap=0.2,  # Space between bar groups
        width=fig_width,  # Plot width
        height=fig_height,  # Adjust height to fit the legend better
        margin=dict(l=10, r=10, t=120, b=80),  # Tighten margins
        title=title,  # Set the plot title
        title_x=0.5,  # Center the title
        title_y=title_y_pos,  # Adjust title position
        yaxis=dict(title=yaxis_title, range=y_range),
    )

    # Save the figure if a filename is provided
    if filename:
        fig.write_image(filename)
        print(f"Plot saved to {filename}")

    return fig


def plot_comparison_line(
    df,
    line_column,  # Column to be used for unique line categories
    line_group_column,  # Column to group lines by (was percentages)
    line_color_column,  # Column to define line colors (was backbones)
    line_style_column=None,  # Column to define line styles
    line_group_column_text=None,  # Order of the line groups
    colors=None,  # List of colors corresponding to the unique values in line_column
    line_styles=None,  # List of line styles corresponding to the unique values in line_style_column
    result_column="result",  # Column for the y-axis result values
    metric_name="Metric",  # Name for the y-axis title
    fig_width=1200,  # Width of the figure
    fig_height=600,  # Height of the figure
    title=None,  # Title of the plot
    title_y_pos=None,  # Vertical position of the title
    filename=None,  # Optional: File name to save the plot
    xaxis_title="Group",  # Optional: Name for the x-axis title
    yaxis_title=None,  # Optional: Name for the y-axis title,
    # error_bars=True,  # Optional: Show error bars
    y_range=None,  # Optional: Minimum value for the y-axis
):
    # Default colors if not provided by the user
    if colors is None:
        colors = px.colors.qualitative.Plotly

    # Default line styles if not provided by the user
    if line_styles is None:
        line_styles = ["solid", "dash", "dot", "dashdot"]

    if yaxis_title is None:
        yaxis_title = metric_name

    if title_y_pos is None:
        if title is None:
            title_y_pos = 0.8
        else:
            title_y_pos = 0.95

    fig = go.Figure()

    # Retrieve unique values from the respective columns
    line_values = sorted(df[line_column].unique())
    line_groups = sorted(df[line_group_column].unique())
    line_colors = sorted(df[line_color_column].unique())
    if line_style_column:
        line_styles_values = sorted(df[line_style_column].unique())
    else:
        line_styles_values = [""]

    # Loop through the combinations to create the lines
    for value_idx, line_value in enumerate(line_values):
        for color_idx, line_color in enumerate(line_colors):
            for style_idx, line_style_value in enumerate(line_styles_values):

                # Filter the dataframe based on the combination of values
                sub_df = df[
                    (df[line_column] == line_value)
                    & (df[line_color_column] == line_color)
                ]
                if line_style_column:
                    sub_df = sub_df[
                        sub_df[line_style_column] == line_style_value
                    ]

                if sub_df.empty:
                    continue

                if line_group_column_text:
                    x = sub_df[line_group_column_text].tolist()
                else:
                    x = sub_df[line_group_column].tolist()

                # Extract values for plotting
                y = sub_df[result_column].astype(float).tolist()
                y_std = (
                    sub_df[result_column].astype(float).std()
                )  # Standard deviation
                color = colors[
                    value_idx % len(colors)
                ]  # Color based on line_value
                line_style = line_styles[
                    style_idx % len(line_styles)
                ]  # Line style
                name = f"{line_value} {line_color}"  # Line label
                if line_style_column:
                    name += f" ({line_style_value})"

                if pd.isna(y).all():
                    continue

                # if error_bars:
                #     error_y = dict(
                #         type="data",
                #         array=[y_std] * len(y),
                #         visible=True,
                #         color="black",
                #     )
                # else:
                #     error_y = None

                # Add the line trace
                fig.add_trace(
                    go.Scatter(
                        x=x,  # Categorical or ordered x value
                        y=y,  # List of result values
                        name=name,
                        mode="lines",  # Ensure both lines and markers
                        line=dict(
                            color=color, dash=line_style
                        ),  # Line style and color
                        # error_y=error_y,  # Error bars
                    )
                )

    # Update layout with axis titles and legend
    fig.update_layout(
        xaxis_title=xaxis_title,  # X-axis title
        font=dict(family="Times New Roman", size=14),
        legend=dict(
            font=dict(family="Times New Roman", size=14),
            title="Models",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        width=fig_width,  # Plot width
        height=fig_height,  # Adjust height to fit the legend better
        margin=dict(l=10, r=10, t=120, b=80),  # Tighten margins
        title=title,  # Set the plot title
        title_x=0.5,  # Center the title
        title_y=title_y_pos,  # Adjust title position
        yaxis=dict(title=yaxis_title, range=y_range),
    )

    # Save the figure if a filename is provided
    if filename:
        fig.write_image(filename)
        print(f"Plot saved to {filename}")

    return fig


# Plots a bar chart with information from variants and scenarios.
# variants_vars contains the set of variables that defines the variants.
#   It is a list of tuples (variable, trait) in which variable indicates one of the variable that defines the variants and a trait that will be used to decorate the bar. Valid trait values are: "color", "pattern" and  ""
#   Example: variants_vars = [("tsk_pretext","color"), ("backbone","pattern")]
# scenarios_vars contains the list of variables that define how the results will be organized in scenarios groups.
#   Example: group_scenarios_vars = ["tsk_target", "d_target"]
def plot_bar_chart(
    df,
    variants_vars,  # list of tubples with variables/trait pairs that define the variants.
    # traits are "color", "pattern", "".
    group_scenarios_vars,  # list of variables that define the scenarios to group results (bar groups)
    result_column="result",  # Column for the y-axis result values
    yaxis_title="Metric",  # Name for the y-axis title
    bar_width=0.1,  # Width of the bars
    fig_width=1200,  # Width of the figure
    fig_height=600,  # Height of the figure
    title=None,  # Title of the plot
    title_y_pos=None,  # Vertical position of the title
    filename=None,  # Optional: File name to save the plot
    xaxis_title="Bar group category name",  # Optional: Name for the x-axis title
    error_bars=True,  # Optional: Show error bars
    y_range=None,  # Optional: Minimum value for the y-axis
    bar_colors=None,  # Optional: list of colors to be used when coloring the bars
    bar_patterns=None,  # Optional: list of patterns to be used when drawing the bars. Default = ["", "/", "x", "+"]
    variants_sort_fn=None,
    verbose=False,
):
    # Variants vars indices that define bars colors ant patterns
    bar_pattern_indices = []
    bar_color_indices = []
    for i, (var, trait) in enumerate(variants_vars):
        if trait == "color":
            bar_color_indices.append(i)
        if trait == "pattern":
            bar_pattern_indices.append(i)

    # Initialize the bar_color dictionary
    bar_color_idx = 0
    if bar_colors is None:
        bar_colors = px.colors.qualitative.Plotly
    bar_color_dict = {}

    def get_bar_color(vars):
        nonlocal bar_color_idx
        variant_color_id = tuple([vars[i] for i in bar_color_indices])
        if not variant_color_id in bar_color_dict:
            bar_color_dict[variant_color_id] = bar_colors[
                bar_color_idx % len(bar_colors)
            ]
            bar_color_idx += 1
        return bar_color_dict[variant_color_id]

    # Initialize the bar_pattern dictionary
    bar_pattern_idx = 0
    if bar_patterns is None:
        bar_patterns = ["", "/", "x", "+"]
    bar_pattern_dict = {}

    def get_bar_pattern(vars):
        nonlocal bar_pattern_idx
        variant_pattern_id = tuple([vars[i] for i in bar_pattern_indices])
        if not variant_pattern_id in bar_pattern_dict:
            bar_pattern_dict[variant_pattern_id] = bar_patterns[
                bar_pattern_idx % len(bar_patterns)
            ]
            bar_pattern_idx += 1
        return bar_pattern_dict[variant_pattern_id]

    if title_y_pos is None:
        if title is None:
            title_y_pos = 0.8
        else:
            title_y_pos = 0.95

    fig = go.Figure()

    # Loop through the combinations to create the bars
    legend_set = set()
    variants_vars_only = [v for v, t in variants_vars]
    for scenario_ID, scenario_df in sorted(df.groupby(group_scenarios_vars)):
        scenario_ID_fmt = "/".join(scenario_ID)

        if verbose:
            print("Scenario: ", scenario_ID_fmt)

        for variant_ID, variant_scenario_df in sorted(
            scenario_df.groupby(variants_vars_only), key=variants_sort_fn
        ):
            # Convert to a single value if the variant_ID is a string
            if isinstance(variant_ID, str):
                variant_ID = [variant_ID]
                
            variant_ID_fmt = "+".join([str(v) for v in variant_ID])
            
            if variant_scenario_df.empty:
                print(
                    f'WARNING: no results for variant "{variant_ID_fmt}" in scenario "{scenario_ID_fmt}"'
                )
                continue
            # Extract values for plotting
            y = (
                variant_scenario_df[result_column].astype(float).mean()
            )  # Mean result
            y_std = (
                variant_scenario_df[result_column].astype(float).std()
            )  # Standard deviation
            if error_bars:
                error_y = dict(
                    type="data",
                    array=[y_std],
                    visible=True,
                    color="black",
                )
            else:
                error_y = None

            color = get_bar_color(variant_ID)
            pattern = get_bar_pattern(variant_ID)

            if verbose:
                print(
                    f" - Variant: {variant_ID_fmt:20s} : {y:.2f} +- {y_std:.2f} : {color} : {pattern} : {variant_ID}"
                )
            bar_label = variant_ID_fmt

            unique_offsetgroup = variant_ID_fmt

            fig.add_trace(
                go.Bar(
                    x=[scenario_ID_fmt],  # Categorical x value
                    y=[y],  # Mean result
                    name=bar_label,
                    marker_color=color,
                    marker_pattern_shape=pattern,
                    error_y=error_y,
                    marker_pattern_fillmode="replace",
                    marker_pattern_solidity=0.75,
                    width=bar_width,
                    offsetgroup=unique_offsetgroup,  # Ensure bars are uniquely offset
                    showlegend=False,
                )
            )

            # Add legend entry for each unique combination on the first group only
            if not variant_ID in legend_set:
                legend_set.add(variant_ID)
                fig.add_trace(
                    go.Bar(
                        x=[None],
                        y=[None],
                        name=bar_label,
                        marker_color=color,
                        marker_pattern_shape=pattern,
                        marker_pattern_fillmode="replace",
                        marker_pattern_solidity=0.75,
                        width=bar_width,
                        offsetgroup=unique_offsetgroup,
                        showlegend=True,
                    )
                )

    # Update layout with axis titles and legend
    fig.update_layout(
        barmode="group",  # Group bars side by side
        xaxis_title=xaxis_title,  # Generic name for x-axis
        font=dict(family="Times New Roman", size=14),
        legend=dict(
            font=dict(family="Times New Roman", size=14),
            title="Variants",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        bargap=0.2,  # Space between bar groups
        width=fig_width,  # Plot width
        height=fig_height,  # Adjust height to fit the legend better
        margin=dict(l=10, r=10, t=120, b=80),  # Tighten margins
        title=title,  # Set the plot title
        title_x=0.5,  # Center the title
        title_y=title_y_pos,  # Adjust title position
        yaxis=dict(title=yaxis_title, range=y_range),
    )

    # Save the figure if a filename is provided
    if filename:
        fig.write_image(filename)
        print(f"Plot saved to {filename}")

    return fig