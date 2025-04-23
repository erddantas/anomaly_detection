"""Utility functions for getting the history of a table from the Delta Lake transaction log."""

import dataclasses
import time
import pandas as pd
from pyspark.sql.functions import col, lit, to_timestamp, date_sub
from pyspark.sql import DataFrame
from typing import Optional
from databricks.data_monitoring.anomalydetection.context import Context
from databricks.data_monitoring.anomalydetection import (
    model_config as model_config_utils,
    errors,
)
from datetime import datetime

# based on https://docs.databricks.com/en/delta/history.html#operation-metrics-keys
UPDATE_OP_KEYS = [
    "WRITE",
    "CREATE TABLE",
    "CREATE TABLE AS SELECT",
    "REPLACE TABLE AS SELECT",
    "COPY INTO",
    "STREAMING UPDATE",
    "MERGE",
    "UPDATE",
    "CREATE OR REPLACE TABLE AS SELECT",
    "CLONE",
]


@dataclasses.dataclass
class TableHistory:
    table_name: str
    evaluation_epoch_sec: int
    history_df: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None


def get_table_history(
    table_name: str,
    catalog_name: str,
    schema_name: str,
    should_limit: bool = False,
    days_limit: Optional[int] = None,
) -> TableHistory:
    """
    Get the history of a table from the Delta Lake transaction log in descending time, limited to the last
    max_commit_training_points entries.

    :param table_name: The name of the table.
    :param catalog_name: The name of the catalog.
    :param schema_name: The name of the schema.
    :param should_limit: Whether to limit the number of records to the last max_commit_training_points.
    :param days_limit: The number of days to limit the history to. Typically set when should_limit is False.

    :return: A tuple containing the table name, the history dataframe, and the evaluation timestamp.
    """
    evaluation_epoch_sec = int(time.time())
    try:
        limit_clause = ""
        if should_limit:
            max_commit_training_points = model_config_utils.get_max_commit_training_points()
            limit_clause = f"LIMIT {max_commit_training_points}"

        history_df = Context.spark.sql(
            f"DESCRIBE HISTORY `{catalog_name}`.`{schema_name}`.`{table_name}` {limit_clause}"
        ).select(["timestamp", "operation", "operationMetrics"])

        # Apply filtering to get only rows with timestamp >= cutoff_timestamp when days_limit is passed
        if days_limit:
            filtered_history_df_limit = history_df.filter(
                col("timestamp")
                >= date_sub(
                    to_timestamp(lit(datetime.fromtimestamp(evaluation_epoch_sec))), days_limit
                )
            )
            filtered_history_df_limit = _filter_history_ops(filtered_history_df_limit)

            # If there are not enough entries in the filtered history after applying the days limit + filtering, limit by max commit timestamps
            # Needs at least 1 more point than min, since number of durations = num points - 1.
            if (
                len(
                    filtered_history_df_limit.take(
                        model_config_utils.get_min_commit_training_points() + 1
                    )
                )
                <= model_config_utils.get_min_commit_training_points()
            ):
                filtered_history_df = history_df.limit(
                    model_config_utils.get_max_commit_training_points()
                )
                filtered_history_df = _filter_history_ops(filtered_history_df)
            else:
                filtered_history_df = filtered_history_df_limit
        else:
            filtered_history_df = history_df
            filtered_history_df = _filter_history_ops(filtered_history_df)
        return TableHistory(
            table_name=table_name,
            history_df=filtered_history_df.select(
                ["timestamp", "operation", "operationMetrics"]
            ).toPandas(),
            evaluation_epoch_sec=evaluation_epoch_sec,
        )
    except Exception as e:
        error_message = str(e)
        if "PERMISSION_DENIED" in error_message:
            error_message = errors.ERROR_CODE_TO_MESSAGE[errors.ErrorCode.PERMISSION_DENIED]
        return TableHistory(
            table_name=table_name,
            history_df=None,
            evaluation_epoch_sec=evaluation_epoch_sec,
            error_message=error_message,
        )


def _filter_history_ops(history_df: DataFrame) -> DataFrame:
    filtered_history_df = history_df.filter(col("operation").isin(UPDATE_OP_KEYS))

    # Skip empty writes
    filtered_history_df = filtered_history_df.filter(
        ((col("operationMetrics.numFiles").isNull()) | (col("operationMetrics.numFiles") != 0))
        & (
            (col("operationMetrics.numOutputRows").isNull())
            | (col("operationMetrics.numOutputRows") != 0)
        )
        & (
            (col("operationMetrics.numOutputBytes").isNull())
            | (col("operationMetrics.numOutputBytes") != 0)
        )
    )
    return filtered_history_df
