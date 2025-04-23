import dataclasses
import pandas as pd
import plotly.graph_objects as go
import logging

from typing import List, Dict
from databricks.data_monitoring.anomalydetection import (
    freshness_utils as utils,
)
from databricks.data_monitoring.anomalydetection.freshness_info import (
    TableFreshnessInfo,
    ResultStatus,
)

_logger = logging.getLogger(__name__)

SCENARIO_COLOR_MAP = {
    # Color scheme for different number of updates in the past
    "past_update_0_commits": "#EDF0F3",  # light grey
    "past_update_1_commit": "#44C970",  # green for 1 commit in the hour
    "past_update_2_6_commits": "#277C43",  # dark green for 2-6 commits
    "past_update_7+_commits": "#054B1C",  # darkest green for 7+ commits
    # Color scheme for the prediction
    "stale_window": "rgba(250, 203, 102, 0.5)",  # yellow with 50% opacity
    "predicted_window": "rgba(138, 202, 255, 0.5)",  # blue with 50% opacity
    # Color scheme for the background and reference lines
    "now": "grey",  # dark grey
    "background": "white",  # white
    "day_divider": "#e8e8e8",  # grey
}


@dataclasses.dataclass
class PlotData:
    """
    Data class for storing data needed to plot a table's freshness status with
    recent update history as reference.
    """

    name: str
    history_pdf: pd.DataFrame
    freshness_info: TableFreshnessInfo


def _decide_bar_color(
    timestamp_hr: pd.Timestamp,
    update_counts: int,
) -> str:
    """
    Determines the color of the hour bar base on the number of updates in the hour.

    :param timestamp_hr: timestamp of the hour bar
    :param update_counts: number of updates happened in this hour
    :return: A color string for the bar that can be recognized by plotly
    """
    count = update_counts[timestamp_hr]
    if count == 0:
        return SCENARIO_COLOR_MAP["past_update_0_commits"]
    elif count == 1:
        return SCENARIO_COLOR_MAP["past_update_1_commit"]
    elif 2 <= count <= 6:
        return SCENARIO_COLOR_MAP["past_update_2_6_commits"]
    else:
        return SCENARIO_COLOR_MAP["past_update_7+_commits"]


def _add_prediction_window_highlight(
    fig: go.Figure, freshness_info: TableFreshnessInfo, y_base: float, bar_height: float
):
    """Shade a region on the plot to highlight the predicted window for freshness status."""
    predicted_next_data_update_hr = pd.Timestamp(freshness_info.predicted_next_data_update).floor(
        "H"
    )
    predicted_upper_bound_next_data_update_hr = pd.Timestamp(
        freshness_info.predicted_upper_bound_next_data_update
    ).ceil("H")
    evaluated_at_hr = pd.Timestamp(freshness_info.evaluated_at).ceil("H")
    if freshness_info.commit_freshness_status == ResultStatus.FRESH:
        fig.add_trace(
            go.Scatter(
                x=[
                    predicted_next_data_update_hr,
                    predicted_upper_bound_next_data_update_hr,
                    predicted_upper_bound_next_data_update_hr,
                    predicted_next_data_update_hr,
                ],
                y=[
                    y_base - 0.1,
                    y_base - 0.1,
                    y_base + bar_height + 0.1,
                    y_base + bar_height + 0.1,
                ],
                fill="toself",
                fillcolor=SCENARIO_COLOR_MAP["predicted_window"],
                line=dict(width=0),  # No border
                mode="lines",
                hoverinfo="text",  # Ensure hover displays only text
                text=f"<b>Predicted Window</b><br>Start: {freshness_info.predicted_next_data_update}<br>End: {freshness_info.predicted_upper_bound_next_data_update}",
                hoveron="fills",
                showlegend=False,
            )
        )
    elif freshness_info.commit_freshness_status == ResultStatus.STALE:
        fig.add_trace(
            go.Scatter(
                x=[
                    predicted_next_data_update_hr,
                    evaluated_at_hr,
                    evaluated_at_hr,
                    predicted_next_data_update_hr,
                ],
                y=[
                    y_base - 0.1,
                    y_base - 0.1,
                    y_base + bar_height + 0.1,
                    y_base + bar_height + 0.1,
                ],
                fill="toself",
                fillcolor=SCENARIO_COLOR_MAP["stale_window"],
                line=dict(width=0),  # No border
                mode="lines",
                hoverinfo="text",  # Ensure hover displays only text
                text=f"<b>Stale Window</b><br>Start: {freshness_info.predicted_next_data_update}<br>End: {freshness_info.evaluated_at}",
                hoveron="fills",
                showlegend=False,
            )
        )
    else:
        pass

    return fig


def _convert_commit_timestamps_to_pandas_df(commit_timestamps: List[int]) -> pd.DataFrame:
    """
    Converts a list of commit timestamps to a pandas DataFrame.

    :param commit_timestamps: List of commit timestamps.
    :return: A pandas DataFrame with a single column 'timestamp'.
    """
    history_pdf = pd.DataFrame(pd.to_datetime(commit_timestamps, unit="s"), columns=["timestamp"])
    return history_pdf


def _get_plot_data(freshness_summary: Dict[str, TableFreshnessInfo]) -> List[PlotData]:
    """Given a freshness summary dictionary, returns a list of PlotData objects for visualization."""
    plot_data_list = []
    sorted_freshness_summary = utils.sort_freshness_summary(freshness_summary)
    for table_name, freshness_info in sorted_freshness_summary.items():
        if freshness_info.commit_timestamps is None or len(freshness_info.commit_timestamps) == 0:
            _logger.info(
                f"Failed to retrieve commit timestamps for {table_name} when creating visualization."
            )
            continue
        history_pdf = _convert_commit_timestamps_to_pandas_df(freshness_info.commit_timestamps)
        plot_data_list.append(
            PlotData(name=table_name, history_pdf=history_pdf, freshness_info=freshness_info)
        )
    return plot_data_list


def _get_figure_for_plot_data(
    plot_data_list: List[PlotData],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> go.Figure:
    """Creates a plotly figure for the given list of plot data between start and end time."""
    bar_height = 0.6  # Height of each hour bar, making bars shorter
    bar_width = 1 * 60 * 60 * 1000  # Width of each hour bar, in milliseconds
    row_gap = 0.5  # Gap between rows for separation
    title_gap = 0.5  # Extra space above the content for the title
    row_height = bar_height + row_gap  # Total height for each row (bar + gap)
    total_height_y = row_height * len(plot_data_list) + title_gap  # Total y-axis height
    plot_height_px = 100 + 50 * len(plot_data_list)  # Calculate total plot height in pixels

    # Shared hour bar ranges and day divider lines for all plots
    hourly_range = pd.date_range(start=start_time, end=end_time, freq="H")
    daily_range = pd.date_range(start=start_time, end=end_time, freq="D")

    fig = go.Figure()

    # Categorical y-axis labels for each table (table name, last update time)
    y_ticks = [total_height_y - title_gap]
    y_labels = [f"<span style='font-size:13px;'>Table (latest commit)</span>"]
    for i, plot_data in enumerate(plot_data_list, start=1):
        history_pdf = plot_data.history_pdf
        # Cast timestamps to pandas Timestamp type to access floor method
        predicted_next_data_update = pd.Timestamp(
            plot_data.freshness_info.predicted_next_data_update
        )
        predicted_upper_bound_next_data_update = pd.Timestamp(
            plot_data.freshness_info.predicted_upper_bound_next_data_update
        )
        evaluated_at = pd.Timestamp(plot_data.freshness_info.evaluated_at)

        # Calculate the y position for this table's row
        y_base = (
            total_height_y - title_gap - (i * row_height)
        )  # Each table is positioned with row_gap
        y_ticks.append(y_base + 0.5 * bar_height)  # Center the tick label within the row

        # Calculate the latest update time for each table
        latest_update_time = history_pdf["timestamp"].max()
        formatted_latest_update_time = (
            latest_update_time.strftime("%Y-%m-%d %H:%M")
            if pd.notnull(latest_update_time)
            else "No updates"
        )
        # Combine table name and last update time for the y-axis label
        y_labels.append(
            f"<b>{plot_data.freshness_info.table_name}</b><br>(Last update: {formatted_latest_update_time})"
        )

        # Calculate hourly update counts
        history_pdf["timestamp_hour"] = history_pdf["timestamp"].dt.floor("H")
        update_counts = (
            history_pdf["timestamp_hour"].value_counts().reindex(hourly_range, fill_value=0)
        )

        # Initialize bar colors and hover text based on update frequency and conditions
        colors = []
        hover_texts = []
        for timestamp in hourly_range:
            count = update_counts[timestamp]
            colors.append(_decide_bar_color(timestamp, update_counts))
            # Custom hover text for each bar, the "<extra></extra>" part is for disabling the secondary
            # hover-over text box.
            hover_texts.append(
                f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M')}<br>Updates: {count}<extra></extra>"
            )

        # Add hourly bars for this table with custom hover text
        fig.add_trace(
            go.Bar(
                x=hourly_range,
                y=[bar_height] * len(hourly_range),  # Set fixed bar height
                marker=dict(color=colors),
                width=bar_width,
                base=y_base,
                showlegend=False,
                # Customize hover text to show timestamp and update count
                hovertext=hover_texts,
                hoverinfo="text",
                hovertemplate="%{hovertext}",
                # Color the edge of the bar the same color as the background to hide the border
                # but keep the gap between bars
                marker_line=dict(color=SCENARIO_COLOR_MAP["background"], width=1),
            )
        )

        # Add a shaded region to highlight the predicted window
        _add_prediction_window_highlight(fig, plot_data.freshness_info, y_base, bar_height)

        # Add a vertical reference line for evaluated_at for this table row
        fig.add_trace(
            go.Scatter(
                x=[evaluated_at, evaluated_at],
                y=[y_base - 0.2, y_base + bar_height + 0.2],  # Cover the row range with some margin
                mode="lines",
                line=dict(color=SCENARIO_COLOR_MAP["now"], width=2, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add thin solid reference lines for the beginning of each day in pale grey
    for day_start in daily_range:
        fig.add_trace(
            go.Scatter(
                x=[day_start, day_start],
                y=[-0.1, total_height_y - title_gap],  # Span across all table rows
                mode="lines",
                line=dict(color=SCENARIO_COLOR_MAP["day_divider"], width=0.5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add titles and legends
    fig.add_annotation(
        x=start_time
        + pd.Timedelta(hours=1),  # Align the title to the left side of the bars with some offset
        y=y_ticks[0],
        text=f"<span style='font-size:13px;'>Past Week Commits</span>",
        showarrow=False,
        font=dict(size=12),
        align="left",
        xanchor="left",
        hovertext=None,
    )

    legend_items = {
        "Update Commits": SCENARIO_COLOR_MAP["past_update_1_commit"],
        "Stale Window": SCENARIO_COLOR_MAP["stale_window"],
        "Predicted Window": SCENARIO_COLOR_MAP["predicted_window"],
    }
    for i, (label, color) in enumerate(legend_items.items()):
        x_pos = start_time + pd.Timedelta(days=4) + pd.Timedelta(days=i)

        # Add the legend bar
        fig.add_trace(
            go.Bar(
                x=[x_pos],
                y=[bar_height * 0.5],  # Set legend bars to half the height of the hourly bars
                base=y_ticks[0] - bar_height * 0.25,  # Center the legend bar vertically
                width=bar_width,
                marker=dict(color=color),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Add a label next to the legend bar
        fig.add_annotation(
            x=x_pos + pd.Timedelta(hours=1),  # 1 bar width to the right of the legend bar
            y=y_ticks[0],
            text=label,
            showarrow=False,
            font=dict(size=10),
            align="left",
            xanchor="left",
            hovertext=None,
        )

    # Update layout to show y-axis labels with table names and last update time
    fig.update_layout(
        yaxis=dict(
            title=None,
            tickvals=y_ticks,  # Set tick positions based on each table's y_base
            ticktext=y_labels,  # Set tick labels as (table names, last update time)
            showgrid=False,
            zeroline=False,  # Hide the zero line to keep the plot clean
            range=[-row_gap, total_height_y],  # Adjusted y-axis range for equal row spacing
            autorange=False,  # Disable auto-ranging
        ),
        xaxis=dict(
            title=None,
            range=[start_time - pd.Timedelta(hours=1), end_time],  # Adjust x-axis range
            showgrid=False,
            type="date",
            side="top",
        ),
        plot_bgcolor=SCENARIO_COLOR_MAP["background"],
        height=plot_height_px,
        bargap=0,
        margin=dict(l=150, r=10, t=10, b=10),  # Increase left margin for long y-tick labels
    )
    return fig


def plot_freshness_summary(
    freshness_summary: Dict[str, TableFreshnessInfo],
    n_tables_per_plot: int = 10,
):
    """Plots the freshness summary for a list of tables with update history as reference."""
    # Get plot data for each table that has a freshness result
    plot_data_list = _get_plot_data(
        freshness_summary={
            k: v
            for (k, v) in freshness_summary.items()
            if v.commit_freshness_status not in [ResultStatus.UNKNOWN, ResultStatus.SKIPPED]
        }
    )

    # Short circuit if there are no tables to plot
    if len(plot_data_list) == 0:
        _logger.info("No tables with valid result status to plot.")
        return

    # Define start and end time for a 7-day look-back window based on the maximum evaluation timestamp
    overall_end_time = pd.Timestamp(max([d.freshness_info.evaluated_at for d in plot_data_list]))
    start_time = overall_end_time.floor("H") - pd.Timedelta(days=7)
    end_time = overall_end_time.floor("H") + pd.Timedelta(days=1)

    # Plot tables in groups of n_tables_per_plot
    for i in range(0, len(plot_data_list), n_tables_per_plot):
        fig = _get_figure_for_plot_data(
            plot_data_list[i : i + n_tables_per_plot],
            start_time,
            end_time,
        )
        fig.show()
