"""
This module provides util methods for checking for completeness anomalies
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import dataclasses

from datetime import datetime, timezone

from databricks.data_monitoring.anomalydetection.completeness_info import (
    TableCompletenessInfo,
    CompletenessResult,
)
from databricks.data_monitoring.anomalydetection import (
    completeness_utils,
    errors,
)
from databricks.data_monitoring.anomalydetection.utils import common_utils
import databricks.data_monitoring.anomalydetection.model_config as model_config_utils
from databricks.data_monitoring.anomalydetection import blast_radius


def row_volumes_from_history_df(history_df: pd.DataFrame) -> np.ndarray:
    """
    Process the history dataframe to return row volume sums grouped by 24-hour buckets relative to the current time.

    :param history_df:
        The history dataframe containing the following columns:
        - 'timestamp': A timestamp (of type timestamp) indicating when the operation occurred.
        - 'operation': A string representing the type of operation (e.g., 'WRITE', 'MERGE').
        - 'operationMetrics': A dictionary containing metrics for the operation,
          such as 'numOutputRows', 'numTargetRowsInserted', or 'numUpdatedRows'.

    :return:
        A 1-dimensional NumPy array of row volume sums. Each element in the array corresponds to the
        total row volume for a 24-hour time bucket. The array is sorted in descending order of time buckets,
        where the last element is the most recent bucket.
    """
    if history_df.empty:
        return np.array([])

    # Ensure 'timestamp' is a datetime object
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], utc=True)

    # Mapping from operation to metric
    operation_to_metric = {
        "WRITE": "numOutputRows",
        "CREATE TABLE AS SELECT": "numOutputRows",
        "REPLACE TABLE AS SELECT": "numOutputRows",
        "COPY INTO": "numOutputRows",
        "STREAMING UPDATE": "numOutputRows",
        "MERGE": "numTargetRowsInserted",
        "UPDATE": "numUpdatedRows",
        "CREATE OR REPLACE TABLE AS SELECT": "numOutputRows",
    }

    # Extract metrics and add a new column with the metric value
    def extract_metric(row):
        metric_key = operation_to_metric.get(row["operation"], None)
        base_metric = int(row["operationMetrics"].get(metric_key, 0)) if metric_key else 0
        num_rows_deleted = int(row["operationMetrics"].get("numDeletedRows", 0))
        return base_metric - num_rows_deleted

    history_df["row_volume"] = history_df.apply(extract_metric, axis=1)

    now = datetime.now(timezone.utc)

    history_df["bucket"] = (
        (now - history_df["timestamp"]).dt.total_seconds() // (24 * 3600)
    ).astype(int)
    grouped_df = history_df.groupby("bucket")["row_volume"].sum()
    all_buckets = range(0, grouped_df.index.max() + 1)
    grouped_df = grouped_df.reindex(all_buckets, fill_value=0)

    return grouped_df.sort_index(ascending=False).values


def check_is_complete_override(
    catalog_name: str,
    schema_name: str,
    table_name: str,
    last_window_row_volume: int,
    threshold: int,
    evaluated_at: datetime,
) -> TableCompletenessInfo:
    status = (
        CompletenessResult.HEALTHY
        if last_window_row_volume >= threshold
        else CompletenessResult.UNHEALTHY
    )
    return TableCompletenessInfo(
        table_name=table_name,
        completeness_status=status,
        last_window_row_volume=last_window_row_volume,
        predicted_row_volume_lower_bound=threshold,
        table_lineage_link=common_utils.get_lineage_link(
            f"{catalog_name}.{schema_name}.{table_name}"
        ),
        evaluated_at=evaluated_at,
    )


def get_single_table_completeness_info(
    catalog_name: str,
    schema_name: str,
    table_name: str,
    history_df: pd.DataFrame,
    evaluated_at: datetime,
    table_threshold_overrides: Optional[Dict[str, int]] = None,
) -> TableCompletenessInfo:
    """
    Get the completeness info for a single table.
    """
    # if not enough commits in table history
    if len(history_df) < model_config_utils.get_min_commit_training_points():
        return TableCompletenessInfo(
            table_name=table_name,
            completeness_status=CompletenessResult.UNKNOWN,
            evaluated_at=evaluated_at,
            table_lineage_link=common_utils.get_lineage_link(
                f"{catalog_name}.{schema_name}.{table_name}"
            ),
            error_message=errors.ERROR_CODE_TO_MESSAGE[errors.ErrorCode.NOT_ENOUGH_UPDATE_OP],
            error_code=errors.ErrorCode.NOT_ENOUGH_UPDATE_OP,
        )

    # create completeness training data
    training_and_eval_data = completeness_utils.row_volumes_from_history_df(history_df)
    # if not enough training datapoints for completness, return unknown
    if len(training_and_eval_data) <= model_config_utils.get_min_completeness_training_data_size():
        return TableCompletenessInfo(
            table_name=table_name,
            completeness_status=CompletenessResult.UNKNOWN,
            evaluated_at=evaluated_at,
            table_lineage_link=common_utils.get_lineage_link(
                f"{catalog_name}.{schema_name}.{table_name}"
            ),
            error_message=errors.ERROR_CODE_TO_MESSAGE[errors.ErrorCode.NOT_ENOUGH_TABLE_HISTORY],
            error_code=errors.ErrorCode.NOT_ENOUGH_TABLE_HISTORY,
        )

    training_data = training_and_eval_data[:-1]
    eval_point = training_and_eval_data[-1]

    if table_name in table_threshold_overrides:
        res = check_is_complete_override(
            catalog_name,
            schema_name,
            table_name,
            training_and_eval_data[-1],
            table_threshold_overrides[table_name],
            evaluated_at,
        )
    else:
        # fit auto arima model for completeness
        res = check_is_complete_auto_arima(
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            train=training_data,
            eval_point=eval_point,
            evaluated_at=evaluated_at,
            alpha=common_utils.DEFAULT_SENSITIVITY,
        )

    if (
        model_config_utils.get_enable_blast_radius_computation()
        and res.completeness_status
        and res.completeness_status == CompletenessResult.UNHEALTHY
    ):
        # keep health status but surface that we could not compute blast radius.
        try:
            res.blast_radius = blast_radius.get_blast_radius(catalog_name, schema_name, table_name)
        except Exception as e:
            res.error_code = errors.ErrorCode.BLAST_RADIUS_COMPUTATION_ERROR
            res.error_message = f"{errors.ERROR_CODE_TO_MESSAGE[errors.ErrorCode.BLAST_RADIUS_COMPUTATION_ERROR]}: {str(e)}"
    return res


def check_is_complete_auto_arima(
    catalog_name: str,
    schema_name: str,
    table_name: str,
    train: np.ndarray,
    eval_point: int,
    evaluated_at: datetime,
    alpha: float = 0.05,
) -> TableCompletenessInfo:
    res = TableCompletenessInfo(
        table_name=table_name,
        evaluated_at=evaluated_at,
        table_lineage_link=common_utils.get_lineage_link(
            f"{catalog_name}.{schema_name}.{table_name}"
        ),
        confidence_level=1 - alpha,
    )
    # fitting auto arima model
    model_config = model_config_utils.get_model_config().autoarima_model_config
    assert model_config is not None, "Unexpected missing model config"
    res.model_config = dataclasses.asdict(model_config)

    try:
        # fit a seasonal (if-applicable) and non-seasonal model and pick the best one
        selected_model_params, res_model_config = common_utils.fit_multiple_models(
            train=train,
            model_config=model_config,
        )

        # fill out model config and hyper params
        res.model_config.update(res_model_config)
        res.model_hyperparameters = selected_model_params.model.get_params()

        # Forecast the next volume
        next_volume, conf_int = common_utils.predict_auto_arima(
            model_params=selected_model_params, n_periods=1, alpha=alpha
        )

        next_volume_lower_bound = max(int(conf_int[0][0]), 0)

        res.completeness_status = (
            CompletenessResult.UNHEALTHY
            if eval_point < next_volume_lower_bound
            else CompletenessResult.HEALTHY
        )
        res.last_window_row_volume = eval_point
        res.predicted_row_volume_lower_bound = next_volume_lower_bound

    except Exception as e:
        res.completeness_status = CompletenessResult.UNKNOWN
        res.error_message = str(e)
        res.error_code = errors.match_error_message_to_code(res.error_message)

    finally:
        return res
