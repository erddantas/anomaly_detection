"""
This module provides util methods for checking for freshness anomalies
"""

import dataclasses
import concurrent.futures
import json
import time
import logging
import pandas as pd
import pmdarima as pm
import numpy as np
from pyspark.sql import functions as F, types as T

import concurrent.futures

import databricks.data_monitoring.anomalydetection.model_config as model_config_utils
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from databricks.data_monitoring.anomalydetection.freshness_info import (
    ResultStatus,
    RESULT_STATUS_ORDER,
    TableFreshnessInfo,
)
from databricks.data_monitoring.anomalydetection.context import Context
from databricks.data_monitoring.anomalydetection.utils import common_utils
from databricks.data_monitoring.anomalydetection import errors
from databricks.data_monitoring.anomalydetection import blast_radius

_logger = logging.getLogger(__name__)


def json_dump_helper(
    table_name: str,
    overall_freshness_status: ResultStatus,
    commit_freshness_status: ResultStatus,
    event_freshness_status: ResultStatus,
    error_message: str,
    error_code: errors.ErrorCode,
    evaluated_at: int,
    table_lineage_link: str,
):
    return json.dumps(
        {
            "table_name": table_name,
            "overall_freshness_status": overall_freshness_status.value,
            "commit_freshness_status": commit_freshness_status.value,
            "event_freshness_status": event_freshness_status.value,
            "error_message": error_message,
            "error_code": error_code.value,
            "evaluated_at": evaluated_at,
            "table_lineage_link": table_lineage_link,
        },
        default=str,
    )


def seconds_to_days_hours_minutes(seconds: int) -> Optional[str]:
    if pd.isna(seconds):
        return None

    # when there are Nones in pd, all values turn into floats even if originally an int, hence need to convert to int
    days = int(seconds // (24 * 3600))  # Calculate number of days
    hours = int(seconds % (24 * 3600)) // 3600  # Calculate number of hours
    minutes = int(seconds % 3600) // 60  # Calculate number of minutes
    seconds_remaining = int(seconds % 60)  # Calculate remaining seconds

    parts = []
    if days > 0:
        parts.append(f"{days} {'days' if days > 1 else 'day'}")
    if hours > 0:
        parts.append(f"{hours} hr")
    if minutes > 0:
        parts.append(f"{minutes} min")

    # Only add seconds if no higher time units (days, hours, minutes) exist
    if not (days > 0 or hours > 0 or minutes > 0):
        parts.append(f"{seconds_remaining} sec")

    # Join the parts together with a space if they exist
    return " ".join(parts) if parts else None


# Helper function used for custom sorting entries in the freshness summary
def compute_sort_order(
    freshness_status: ResultStatus, is_static: bool, staleness_age_seconds: int, table_name: str
) -> Tuple:
    # if None or not in result list, set as inf
    if freshness_status is None or freshness_status not in RESULT_STATUS_ORDER:
        status_order = float("inf")  # Push None or invalid statuses to the end
    else:
        status_order = RESULT_STATUS_ORDER.index(freshness_status)

    table_name = table_name if table_name is not None else ""

    # set as negative to allow for desc order
    staleness_age_seconds = (
        -staleness_age_seconds if staleness_age_seconds is not None else float("-inf")
    )
    # status order asc, is_static desc (false then true), staleness_age_seconds desc (more stale first), table_name asc
    return (status_order, is_static, staleness_age_seconds, table_name)


# Custom sort freshness summary dict
def sort_freshness_summary(
    freshness_summary: Dict[str, TableFreshnessInfo],
) -> Dict[str, TableFreshnessInfo]:
    def sort_key(item: Tuple[str, TableFreshnessInfo]):
        info = item[1]
        return compute_sort_order(
            info.overall_freshness_status.value,
            info.is_static(),
            info.staleness_age_seconds,
            info.table_name,
        )

    sorted_freshness_summary = dict(sorted(freshness_summary.items(), key=sort_key))
    return sorted_freshness_summary


def preprocess_ml_data(filtered_data: pd.DataFrame, drop_na: bool = False) -> pd.DataFrame:
    filtered_data = filtered_data.sort_values("timestamp")
    filtered_data["previous_timestamp"] = filtered_data["timestamp"].shift(1)

    filtered_data["duration_to_next_timestamp"] = np.where(
        filtered_data["timestamp"].isna() | filtered_data["previous_timestamp"].isna(),
        np.nan,  # Set NaN when either timestamp is NaN/NaT
        (pd.to_datetime(filtered_data["timestamp"]).astype("int64") // 1e9)
        - (pd.to_datetime(filtered_data["previous_timestamp"]).astype("int64") // 1e9),
    )
    # cap max num training points
    filtered_data = filtered_data.iloc[-model_config_utils.get_max_commit_training_points() :, :]
    if drop_na:
        filtered_data = filtered_data.dropna()

    return filtered_data[["timestamp", "duration_to_next_timestamp"]]


def check_is_fresh_auto_arima(
    pdf: pd.DataFrame,
    name,
    current_ts_sec,
    alpha=0.05,
    commit_freshness_model_params: Optional[model_config_utils.ModelParams] = None,
) -> Tuple[TableFreshnessInfo, Optional[model_config_utils.ModelParams]]:
    res = TableFreshnessInfo(
        table_name=name,
        evaluated_at=datetime.fromtimestamp(current_ts_sec),
        confidence_level=1 - alpha,
    )
    model_config = model_config_utils.get_model_config().autoarima_model_config
    assert model_config is not None, "Unexpected missing model config"
    res.commit_model_config = dataclasses.asdict(model_config)
    selected_model_params = None
    try:
        orig_train = pdf.loc[:, "duration_to_next_timestamp"].values  # holds original data
        train = orig_train  # holds the training data

        # if model_dict exists, update the model with the new data
        if commit_freshness_model_params and commit_freshness_model_params.model:
            selected_model_params = common_utils.update_model(
                train=train, model_config=model_config, model_params=commit_freshness_model_params
            )
            # fill out model config and hyper params
            res.commit_model_hyperparameters = selected_model_params.model.get_params()
        else:
            # train model from scratch
            if len(train) <= 1:
                # Set status of table histories with <= 1 data point as Unknown
                res.commit_freshness_status = ResultStatus.UNKNOWN
                res.error_message = errors.ERROR_CODE_TO_MESSAGE[
                    errors.ErrorCode.NOT_ENOUGH_UPDATE_OP
                ]
                res.error_code = errors.ErrorCode.NOT_ENOUGH_UPDATE_OP
                return res, None

            # fit a seasonal (if-applicable) and non-seasonal model and pick the best one
            selected_model_params, res_model_config = common_utils.fit_multiple_models(
                train=train,
                model_config=model_config,
            )

            # transfer over last update timestamp from original model params
            if commit_freshness_model_params:
                selected_model_params.last_update_timestamp = (
                    commit_freshness_model_params.last_update_timestamp
                )

            # fill out model config and hyper params
            res.commit_model_config.update(res_model_config)
            res.commit_model_hyperparameters = selected_model_params.model.get_params()

        # Forecast the next time difference
        n_periods = 1  # Number of steps to predict (we want the next timestamp)
        final_forecast, conf_int = common_utils.predict_auto_arima(
            model_params=selected_model_params, n_periods=n_periods, alpha=alpha
        )

        # handle case when predicted duration is negative
        if final_forecast <= 0:
            final_forecast = selected_model_params.min_val

        # Calculate the predicted next timestamp based on the forecast
        # check if we have a predefined last update timestamp in the original model params arg
        if selected_model_params.last_update_timestamp:
            last_update_sec = selected_model_params.last_update_timestamp
        else:
            last_update_sec = pdf["timestamp"].iloc[-1].timestamp()

        # Get the confidence interval for the forecast
        predicted_timestamp = final_forecast + last_update_sec
        upper_bound_timestamp_sec = conf_int[0, 1] + last_update_sec

        # table is fresh if forecasted next update is in the future
        res.commit_freshness_status = (
            ResultStatus.FRESH
            if current_ts_sec <= upper_bound_timestamp_sec
            else ResultStatus.STALE
        )
        res.last_data_update = datetime.fromtimestamp(last_update_sec)
        res.predicted_next_data_update = datetime.fromtimestamp(predicted_timestamp)
        res.predicted_upper_bound_next_data_update = datetime.fromtimestamp(
            upper_bound_timestamp_sec
        )

    except Exception as e:
        res.error_message = str(e)
        res.error_code = errors.match_error_message_to_code(res.error_message)
        res.commit_freshness_status = ResultStatus.UNKNOWN
    finally:
        return res, selected_model_params


def preprocess_timestamps_for_event_processing(commit_timestamps: List[datetime]) -> List[datetime]:
    # Get descending timestamps
    sorted_indices = sorted(
        range(len(commit_timestamps)), key=lambda i: commit_timestamps[i], reverse=True
    )
    processed_timestamps = []
    last_processed_time = None

    for i in sorted_indices:
        current_time = commit_timestamps[i]
        if (
            last_processed_time is None
            or (last_processed_time - current_time).total_seconds()
            > model_config_utils.get_min_duration_between_event_times_sec()
        ):
            processed_timestamps.append(current_time)
            last_processed_time = current_time
        if len(processed_timestamps) > model_config_utils.get_max_event_training_points():
            break

    processed_timestamps.reverse()
    return processed_timestamps


def get_event_freshness_training_data(
    pdf: pd.DataFrame,
    catalog_name: str,
    schema_name: str,
    table_name: str,
    event_timestamp_col_name: Optional[str],
) -> pd.DataFrame:
    """
    Retrieves training data by computing the maximum event time (`event_time`)
    for a given table as of specified commit timestamps, in parallel.

    :return: pd.DataFrame
        A Pandas DataFrame containing two columns:
        - `event_time`: The maximum value of `event_timestamp_col_name` as of each
          commit timestamp.
        - `eval_time`: The corresponding commit timestamp.

    Example output:
    +-------------------+-------------------+
    |       event_time  |        eval_time  |
    +-------------------+-------------------+
    | 2024-12-01T11:59:59|2024-12-01T12:00:00|
    | 2024-12-01T12:59:59|2024-12-01T13:00:00|
    +-------------------+-------------------+
    """
    commit_timestamps = pdf["timestamp"].tolist()

    def process_commit_timestamp(commit_timestamp) -> Optional[pd.DataFrame]:
        try:
            query = f"""
            SELECT MAX({event_timestamp_col_name}) AS event_time
            FROM `{catalog_name}`.`{schema_name}`.`{table_name}`
            TIMESTAMP AS OF '{commit_timestamp}'
            """
            return (
                Context.spark.sql(query).withColumn("eval_time", F.lit(commit_timestamp)).toPandas()
            )
        except Exception as e:
            # Log error and return None to skip this data point
            _logger.info(
                f"Failed to get max event timestamp for commit {commit_timestamp}: {str(e)}"
            )
            return None

    # Initialize with an empty DataFrame with the expected schema
    training_data_schema = T.StructType(
        [
            T.StructField("event_time", T.TimestampType(), True),
            T.StructField("eval_time", T.TimestampType(), True),
        ]
    )
    final_result_pdf = Context.spark.createDataFrame([], schema=training_data_schema).toPandas()

    # Only process at most one commit every X minutes for event times
    commit_times_to_process = preprocess_timestamps_for_event_processing(commit_timestamps)
    result_pdfs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_commit_timestamp, ts) for ts in commit_times_to_process]

        for future in concurrent.futures.as_completed(futures):
            result_pdf = future.result()
            if result_pdf is not None:  # Skip empty data points
                result_pdfs.append(result_pdf)
    final_result_pdf = pd.concat(result_pdfs) if len(result_pdfs) > 0 else final_result_pdf
    # Sort because the result pdfs can finish in any order
    final_result_pdf = final_result_pdf.sort_values("eval_time")
    return final_result_pdf


# Forecasts next latency for event freshness using ARIMA
def forecast_next_event_time_latency_arima(
    durations, model_config: model_config_utils.ModelConfig, res: TableFreshnessInfo, alpha=0.05
) -> Tuple[float, float]:
    # fit a seasonal (if-applicable) and non-seasonal model and pick the best one
    selected_model_params, res_model_config = common_utils.fit_multiple_models(
        train=durations.values, model_config=model_config
    )

    # fill out model config and hyper params
    res.event_model_config = res_model_config
    res.event_model_hyperparameters = selected_model_params.model.get_params()

    # Forecast the next duration
    n_periods = 1
    forecasted_latency, conf_int = common_utils.predict_auto_arima(
        model_params=selected_model_params, n_periods=n_periods, alpha=alpha
    )
    forecasted_upper_bound_latency = conf_int[0][1]

    return forecasted_latency, forecasted_upper_bound_latency


def check_is_fresh_event_auto_arima(
    res: TableFreshnessInfo,
    pdf: pd.DataFrame,
    catalog_name: str,
    schema_name: str,
    table_name: str,
    event_timestamp_col_name: Optional[str],
    alpha=0.05,
) -> Tuple[TableFreshnessInfo, pd.Series]:
    """
    Checks the freshness of a table based on event timestamps and predicted latencies using an Auto ARIMA model.

    :param res: The current TableFreshnessInfo object to be updated.
    :param pdf: The input DataFrame containing event and evaluation timestamps.
    :param catalog_name: The catalog name for the table.
    :param schema_name: The schema name for the table.
    :param table_name: The table name.
    :param event_timestamp_col_name: The name of the column containing event timestamps.
    :param alpha: The significance level for confidence interval calculations (default is 0.05).

    :return: Tuple[TableFreshnessInfo, pd.Series]:
            - Updated TableFreshnessInfo object with the following fields:
                - event_freshness_status (ResultStatus): Indicates if the table is fresh or stale.
                - event_staleness_age_seconds (int): The number of seconds the table is stale, or None if fresh.
                - event_predicted_latency_seconds (int): The predicted latency for the next event.
                - event_predicted_upper_bound_latency_seconds (int): The upper bound of the predicted latency.
            - A pandas Series of historical latencies calculated from the provided DataFrame.
    """
    if event_timestamp_col_name is None:
        return res, pd.Series(dtype=float)
    try:
        # get model config for training
        model_config = model_config_utils.get_model_config().autoarima_model_config
        assert model_config is not None, "Unexpected missing model config"
        res.event_model_config = dataclasses.asdict(model_config)

        training_df = get_event_freshness_training_data(
            pdf, catalog_name, schema_name, table_name, event_timestamp_col_name
        )
        if len(training_df) <= 2:
            # Set status of table histories with <= 2 data point as Unknown since we exclude the last element
            res.event_freshness_status = ResultStatus.UNKNOWN
            res.error_message = errors.ERROR_CODE_TO_MESSAGE[errors.ErrorCode.NOT_ENOUGH_UPDATE_OP]
            res.error_code = errors.ErrorCode.NOT_ENOUGH_UPDATE_OP
            return res, pd.Series(dtype=float)
        latencies = (
            pd.to_datetime(training_df["eval_time"]) - pd.to_datetime(training_df["event_time"])
        ).dt.total_seconds()
        actual_latency = latencies.iloc[-1]

        # Exclude the last element from the training set, since we evaluate at it.
        latencies_training = latencies.iloc[:-1]

        next_latency, next_upper_bound_latency = forecast_next_event_time_latency_arima(
            latencies_training, model_config, res, alpha
        )
        last_event_datetime = pd.to_datetime(training_df["event_time"].iloc[-1])
        freshness_status = (
            ResultStatus.FRESH
            if (actual_latency <= next_upper_bound_latency)
            else ResultStatus.STALE
        )
        updated_res = dataclasses.replace(
            res,
            event_freshness_status=freshness_status,
            last_event=last_event_datetime,
            event_predicted_latency_seconds=int(next_latency),
            event_predicted_upper_bound_latency_seconds=int(next_upper_bound_latency),
        )
    except Exception as e:
        updated_res = dataclasses.replace(
            res,
            event_freshness_status=ResultStatus.UNKNOWN,
            error_message=str(e),
            error_code=errors.match_error_message_to_code(str(e)),
        )
        latencies = pd.Series(dtype=float)

    return updated_res, latencies


def check_is_fresh_constant(
    df: pd.DataFrame,
    threshold: timedelta,
    name: str,
    current_ts_sec=time.time(),
) -> TableFreshnessInfo:
    current_ts = datetime.fromtimestamp(current_ts_sec)
    res = TableFreshnessInfo(table_name=name, evaluated_at=current_ts)
    try:
        latest_update_timestamp = df["timestamp"].max()

        # Calculate the upper bound timestamp
        upper_bound_timestamp = latest_update_timestamp + threshold

        # Check if the latest timestamp is within threshold
        res.commit_freshness_status = (
            ResultStatus.FRESH if current_ts <= upper_bound_timestamp else ResultStatus.STALE
        )
        res.predicted_next_data_update = upper_bound_timestamp
        res.predicted_upper_bound_next_data_update = upper_bound_timestamp
        res.last_data_update = latest_update_timestamp
    except Exception as e:
        res.error_message = str(e)
        res.error_code = errors.match_error_message_to_code(res.error_message)
    finally:
        return res


def check_is_static_table(
    df: pd.DataFrame,
    name: str,
    current_ts_sec=time.time(),
    static_table_threshold_override: Optional[timedelta] = None,
    model_param: Optional[model_config_utils.ModelParams] = None,
) -> Optional[TableFreshnessInfo]:
    current_ts = datetime.fromtimestamp(current_ts_sec)

    if model_param is not None:
        latest_update_timestamp = model_param.last_update_timestamp
    else:
        latest_update_timestamp = df["timestamp"].max()

    static_threshold = (
        static_table_threshold_override
        if static_table_threshold_override
        else timedelta(days=model_config_utils.get_static_table_age_threshold_days())
    )
    if latest_update_timestamp < current_ts - static_threshold:
        return TableFreshnessInfo(
            table_name=name,
            evaluated_at=current_ts,
            commit_freshness_status=ResultStatus.FRESH,
            last_data_update=latest_update_timestamp,
        )
    return None


def alert_heuristic(
    df_processed: pd.DataFrame, event_latencies: pd.Series, result: TableFreshnessInfo
) -> TableFreshnessInfo:
    """
    Adjusts freshness thresholds using a grace period to reduce noisy alerts for stale tables.

    - **Commit-based freshness**:
      - Grace period is derived from the median update interval (`history_df`).
      - `predicted_upper_bound_next_data_update` is updated with the grace period.
      - `commit_freshness_status` is re-evaluated based on the new upper bound.

         ---------------------------------------------------
        | median_difference  |         grace_period       |
        |--------------------|----------------------------|
        | <=15 min           | 15 min                     |
        | 15 min - 30 min    | median_difference          |
        | 30 mins - 24 hours | 0.5 * median_difference    |
        | >=24 hours         | 12 hours                   |
        ---------------------------------------------------

    - **Event-based freshness**:
      - Grace period is derived from the median latency (`latencies`).
      - `event_predicted_upper_bound_latency_seconds` is updated with the grace period.
      - `event_staleness_age_seconds` and `event_freshness_status` are re-evaluated.

        ---------------------------------------------------
        | median_latency     |         grace_period       |
        |--------------------|----------------------------|
        | <=30 min           | 30 min                     |
        | 30 min - 60 min    | median_latency             |
        | 60 mins - 24 hours | 0.5 * median_latency       |
        | >=24 hours         | 12 hours                   |
        ---------------------------------------------------

        :param df_processed: a pre-processed df of the table history update operations
        :param event_latencies: a series of the historical latencies
        :param result: the result object outputted after a model's prediction

        :return: An updated result object with adjusted upper bounds and freshness statuses.
    """
    # override result based on alerting heuristic (would only work for arima)
    if len(df_processed) <= 1:
        return result

    # add alert heuristic for commit based freshness
    if result.predicted_next_data_update is not None:  # no override for constant threshold
        # 1) determine the granularity frequency of updates (mean)
        time_differences = df_processed["duration_to_next_timestamp"]
        median_difference_seconds = time_differences.median()
        median_difference_timedelta = timedelta(seconds=median_difference_seconds)

        if median_difference_timedelta < timedelta(minutes=30):
            # high granularity, grace period on top of pred = min 15 min
            grace_period_timedelta = max(median_difference_timedelta, timedelta(minutes=15))
        else:
            # have an upper bound for our grace period of 12 hours
            grace_period_timedelta = min(
                timedelta(hours=12), timedelta(seconds=(median_difference_seconds * 0.5))
            )

        # Override grace period for testing purposes if needed
        if model_config_utils.get_max_grace_period_minutes() > 0:
            grace_period_timedelta = min(
                grace_period_timedelta,
                timedelta(minutes=model_config_utils.get_max_grace_period_minutes()),
            )

        # Use max since if model is super confident, still want to add a grace period
        if model_config_utils.get_use_grace_period_on_predicted_upper_bound():
            result.predicted_upper_bound_next_data_update = max(
                result.predicted_upper_bound_next_data_update,
                result.predicted_next_data_update
                + grace_period_timedelta * model_config_utils.get_grace_period_multiplier(),
            )
        else:
            result.predicted_upper_bound_next_data_update = (
                result.predicted_next_data_update
                + grace_period_timedelta * model_config_utils.get_grace_period_multiplier()
            )
        # if the table is eval time is in our new range, mark as Fresh instead
        if result.evaluated_at <= result.predicted_upper_bound_next_data_update:
            result.commit_freshness_status = ResultStatus.FRESH
        else:
            result.commit_freshness_status = ResultStatus.STALE

    # add alert heuristic for event based freshness
    if result.event_predicted_upper_bound_latency_seconds is not None:  # no override
        median_latency = event_latencies.median()
        actual_latency = event_latencies.iloc[-1]

        if median_latency < timedelta(minutes=60).total_seconds():
            # High granularity, grace period on top of pred = min 30 min
            grace_period_sec = max(median_latency, timedelta(minutes=30).total_seconds())
        else:
            # Have an upper bound for our grace period of 12 hours
            grace_period_sec = min(timedelta(hours=12).total_seconds(), median_latency * 0.5)

        # Override grace period for testing purposes if needed
        if model_config_utils.get_max_grace_period_minutes() > 0:
            grace_period_sec = min(
                grace_period_sec,
                timedelta(
                    minutes=model_config_utils.get_max_grace_period_minutes()
                ).total_seconds(),
            )

        # add the grace period on top of the confidence interval to get the new upper bound
        result.event_predicted_upper_bound_latency_seconds = int(
            result.event_predicted_upper_bound_latency_seconds + grace_period_sec
        )
        result.event_freshness_status = (
            ResultStatus.FRESH
            if actual_latency <= result.event_predicted_upper_bound_latency_seconds
            else ResultStatus.STALE
        )
        if result.event_freshness_status == ResultStatus.STALE:
            result.event_staleness_age_seconds = int(
                max(actual_latency - result.event_predicted_upper_bound_latency_seconds, 0)
            )
        else:
            result.event_staleness_age_seconds = None

    return result


def check_is_fresh_event_override(
    table_name: str,
    df_processed: pd.DataFrame,
    evaluation_epoch_sec: float,
    threshold: timedelta,
    event_timestamp_col_name: Optional[str],
    catalog_name: Optional[str] = None,
    schema_name: Optional[str] = None,
) -> TableFreshnessInfo:
    current_ts = datetime.fromtimestamp(evaluation_epoch_sec)
    res = TableFreshnessInfo(table_name=table_name, evaluated_at=current_ts)

    if event_timestamp_col_name is None:
        res.event_freshness_status = ResultStatus.UNKNOWN
        res.error_message = (
            "Event timestamp column not available, but latency overrides are configured"
        )
        res.error_code = errors.ErrorCode.USER_ERROR
        return res
    try:
        last_commit_pdf = df_processed.tail(1)
        training_pdf = get_event_freshness_training_data(
            last_commit_pdf, catalog_name, schema_name, table_name, event_timestamp_col_name
        )
        actual_latency = (
            (
                datetime.fromtimestamp(evaluation_epoch_sec)
                - pd.to_datetime(training_pdf["event_time"])
            )
            .dt.total_seconds()
            .iloc[-1]
        )
        res.event_freshness_status = (
            ResultStatus.FRESH
            if actual_latency <= threshold.total_seconds()
            else ResultStatus.STALE
        )
        res.last_event = pd.to_datetime(training_pdf["event_time"].iloc[-1])
        res.event_predicted_latency_seconds = int(threshold.total_seconds())
        res.event_predicted_upper_bound_latency_seconds = int(threshold.total_seconds())

    except Exception as e:
        res.error_message = str(e)
        res.error_code = errors.match_error_message_to_code(res.error_message)
    finally:
        return res


def check_single_table_freshness(
    df_processed: pd.DataFrame,
    table_name: str,
    evaluation_epoch_sec: float,
    table_threshold_overrides: Dict[str, timedelta],
    table_latency_threshold_overrides: Dict[str, timedelta],
    catalog_name: Optional[str] = None,
    schema_name: Optional[str] = None,
    event_timestamp_col_name: Optional[str] = None,
    static_table_threshold_override: Optional[timedelta] = None,
    commit_freshness_model_params: Optional[model_config_utils.ModelParams] = None,
) -> Tuple[TableFreshnessInfo, Optional[Dict]]:
    # Note: df_processed has been validated to be non-empty
    # If table has not been updated in a long time, treat as static table that is fresh
    result = check_is_static_table(
        df_processed, table_name, evaluation_epoch_sec, static_table_threshold_override
    )
    returned_commit_freshness_model_params = None
    if result is not None:
        pass
    elif table_name in table_threshold_overrides:
        result = check_is_fresh_constant(
            df_processed,
            table_threshold_overrides[table_name],
            table_name,
            evaluation_epoch_sec,
        )
    elif table_name in table_latency_threshold_overrides:
        result = check_is_fresh_event_override(
            table_name,
            df_processed,
            evaluation_epoch_sec,
            table_latency_threshold_overrides[table_name],
            event_timestamp_col_name,
            catalog_name,
            schema_name,
        )
    else:
        df_processed = df_processed.dropna()
        # If no model params are provided, we are training from scratch and will hence need a minimum number of points
        # if model params are provided, we can continue pipeline even with an emtpy train df, since we will simply re-use the prior model for predictions.
        if (
            commit_freshness_model_params is None or commit_freshness_model_params.model is None
        ) and len(df_processed) < model_config_utils.get_min_commit_training_points():
            return TableFreshnessInfo(
                table_name=table_name,
                overall_freshness_status=ResultStatus.UNKNOWN,
                commit_freshness_status=ResultStatus.UNKNOWN,
                event_freshness_status=ResultStatus.UNKNOWN,
                error_message=errors.ERROR_CODE_TO_MESSAGE[errors.ErrorCode.NOT_ENOUGH_UPDATE_OP],
                error_code=errors.ErrorCode.NOT_ENOUGH_UPDATE_OP,
                evaluated_at=datetime.fromtimestamp(evaluation_epoch_sec),
            ), None

        model_type = model_config_utils.get_model_type()
        if model_type == model_config_utils.ModelType.AUTOARIMA:
            result, returned_commit_freshness_model_params = check_is_fresh_auto_arima(
                df_processed,
                table_name,
                evaluation_epoch_sec,
                alpha=common_utils.DEFAULT_SENSITIVITY,
                commit_freshness_model_params=commit_freshness_model_params,
            )
            result, latencies = check_is_fresh_event_auto_arima(
                result,
                df_processed,
                catalog_name,
                schema_name,
                table_name,
                event_timestamp_col_name,
                alpha=common_utils.DEFAULT_SENSITIVITY,
            )
            result = alert_heuristic(
                df_processed=df_processed, event_latencies=latencies, result=result
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    if (
        model_config_utils.get_enable_blast_radius_computation()
        and result.overall_freshness_status
        and result.overall_freshness_status == ResultStatus.STALE
    ):
        # keep health status but surface that we could not compute blast radius.
        try:
            result.blast_radius = blast_radius.get_blast_radius(
                catalog_name, schema_name, table_name
            )
        except Exception as e:
            result.error_code = errors.ErrorCode.BLAST_RADIUS_COMPUTATION_ERROR
            result.error_message = f"{errors.ERROR_CODE_TO_MESSAGE[errors.ErrorCode.BLAST_RADIUS_COMPUTATION_ERROR]}: {str(e)}"
    return result, returned_commit_freshness_model_params


# Note: outputs a serialized dict that stores datetime types as epoch
# seconds + bools as strings
def get_single_table_freshness_info(
    table_name: str,
    history_json_str: str,
    table_lineage_link: str,
    table_threshold_overrides: Dict[str, timedelta],
    table_latency_threshold_overrides: Dict[str, timedelta],
    static_table_threshold_override: Optional[timedelta],
    tables_to_skip: List[str],
    tables_to_scan: List[str],
    evaluation_epoch_sec: int,
    catalog_name: str,
    schema_name: str,
    event_timestamp_col_name: Optional[str] = None,
) -> str:
    # Process single table
    # TODO: remove since this is not needed anymore
    if (tables_to_scan and table_name not in tables_to_scan) or (
        tables_to_skip and table_name in tables_to_skip
    ):
        return json_dump_helper(
            table_name=table_name,
            overall_freshness_status=ResultStatus.SKIPPED,
            commit_freshness_status=ResultStatus.SKIPPED,
            event_freshness_status=ResultStatus.SKIPPED,
            error_message=errors.ERROR_CODE_TO_MESSAGE[errors.ErrorCode.USER_CONFIGURED_SKIP],
            error_code=errors.ErrorCode.USER_CONFIGURED_SKIP,
            evaluated_at=evaluation_epoch_sec,
            table_lineage_link=table_lineage_link,
        )

    try:
        history_df = pd.read_json(history_json_str)
        # catch case where history df has no entries
        if len(history_df) == 0:
            return json_dump_helper(
                table_name=table_name,
                overall_freshness_status=ResultStatus.UNKNOWN,
                commit_freshness_status=ResultStatus.UNKNOWN,
                event_freshness_status=ResultStatus.UNKNOWN,
                error_message=errors.ERROR_CODE_TO_MESSAGE[
                    errors.ErrorCode.NO_UPDATES_IN_TABLE_HISTORY
                ],
                error_code=errors.ErrorCode.NO_UPDATES_IN_TABLE_HISTORY,
                evaluated_at=evaluation_epoch_sec,
                table_lineage_link=table_lineage_link,
            )

        # Get pre-processed data (Guaranteed history has at least 1 update)
        df_processed = preprocess_ml_data(history_df, drop_na=False)

        # determine status of table
        result, _ = check_single_table_freshness(
            df_processed=df_processed,
            table_name=table_name,
            evaluation_epoch_sec=evaluation_epoch_sec,
            table_threshold_overrides=table_threshold_overrides,
            table_latency_threshold_overrides=table_latency_threshold_overrides,
            catalog_name=catalog_name,
            schema_name=schema_name,
            event_timestamp_col_name=event_timestamp_col_name,
            static_table_threshold_override=static_table_threshold_override,
        )
        result.table_lineage_link = table_lineage_link
        freshness_dict = result.asdict()
        for column in freshness_dict.keys():
            if isinstance(freshness_dict[column], datetime):
                freshness_dict[column] = int(freshness_dict[column].timestamp())
        # Add the commit history to the returned dict
        freshness_dict["commit_timestamps"] = (
            history_df["timestamp"]
            .apply(lambda x: int(x.timestamp()))
            .tolist()[: model_config_utils.get_num_commit_timestamps_logged()]
        )
        return json.dumps(freshness_dict, default=str)
    except Exception as e:
        error_message = str(e)
        error_code = errors.match_error_message_to_code(error_message)
        # When there has been an exception getting table history, history_json_str
        # will be the exception message string instead of table history as a json string
        if not history_json_str.startswith("{"):
            error_message = history_json_str
        return json_dump_helper(
            table_name=table_name,
            overall_freshness_status=ResultStatus.UNKNOWN,
            commit_freshness_status=ResultStatus.UNKNOWN,
            event_freshness_status=ResultStatus.UNKNOWN,
            error_message=error_message,
            error_code=error_code,
            evaluated_at=evaluation_epoch_sec,
            table_lineage_link=table_lineage_link,
        )


def rolling_forecast_single_table(
    df_processed: pd.DataFrame,
    catalog_name: str,
    schema_name: str,
    table_name: str,
    start_timestamp: datetime,
    end_timestamp: datetime,
    interval_hr: float,
    table_threshold_overrides: Dict[str, timedelta],
    table_latency_threshold_overrides: Dict[str, timedelta],
    static_table_threshold_override: timedelta,
    table_lineage_link: str,
) -> List[TableFreshnessInfo]:
    evaluation_results = []
    # do not include end_timestamp since that is current eval timestamp, which is already been evaluated and is in logging table
    evaluation_timestamps = pd.date_range(
        start=start_timestamp,
        end=end_timestamp,
        freq=timedelta(hours=interval_hr),
        inclusive="left",
    )

    # Prepare valid data used for training
    filtered_df = df_processed.loc[df_processed.timestamp <= end_timestamp]
    # Get table status at each evaluation_timestamp
    commit_freshness_model_params = model_config_utils.ModelParams()
    last_eval_timestamp = None
    for iter, evaluation_timestamp in enumerate(evaluation_timestamps):
        # Look back 2 weeks and train on that data (capped at max points) if we train model from scratch
        if commit_freshness_model_params.model is None:
            lookback_time = evaluation_timestamp - timedelta(
                days=model_config_utils.get_max_lookback_days()
            )
            train = filtered_df.loc[
                (filtered_df.timestamp <= evaluation_timestamp)
                & (filtered_df.timestamp >= lookback_time)
            ].iloc[-model_config_utils.get_max_backtesting_commit_training_points() :, :]
            # Make sure we have at least min number of points if there isn't enough points in last timeframe
            if len(train) < model_config_utils.get_min_commit_training_points():
                train = filtered_df.loc[filtered_df.timestamp <= evaluation_timestamp].iloc[
                    -model_config_utils.get_min_commit_training_points() :, :
                ]
        else:
            # only include new datapoints
            train = filtered_df.loc[
                (filtered_df.timestamp <= evaluation_timestamp)
                & (filtered_df.timestamp > last_eval_timestamp)
            ]

        # set the last update timestamp
        if len(train) > 0:
            commit_freshness_model_params.last_update_timestamp = (
                train["timestamp"].max().timestamp()
            )

        # determine status of table
        result, commit_freshness_model_params = check_single_table_freshness(
            df_processed=train,
            table_name=table_name,
            catalog_name=catalog_name,
            schema_name=schema_name,
            evaluation_epoch_sec=evaluation_timestamp.timestamp(),
            table_threshold_overrides=table_threshold_overrides,
            table_latency_threshold_overrides=table_latency_threshold_overrides,
            static_table_threshold_override=static_table_threshold_override,
            commit_freshness_model_params=commit_freshness_model_params,
        )
        result.table_lineage_link = table_lineage_link
        evaluation_results.append(result)

        last_eval_timestamp = evaluation_timestamp

        # Reset model params and train from scratch next iteration. We do this to ensure the model that is updated does not degredate in performance over a period of time.
        # Training from scratch leads to best predictions, so we do this every x iterations, controlled by flag eval_points_before_retrain.
        # Corner case: If we have error in predictions and None is retured, reset the model so we train from scratch next iteration
        if commit_freshness_model_params is None or (
            iter % model_config_utils.get_eval_points_before_retrain() == 0 and iter != 0
        ):
            commit_freshness_model_params = model_config_utils.ModelParams()

    return evaluation_results


def backtest_forecast_quality(backtest_result_pdf: pd.DataFrame) -> Dict[str, float]:
    # only consider what model predicts to be fresh in computing ts metrics
    # to avoid biasing the metrics from the delayed updates
    pdf = backtest_result_pdf.loc[
        backtest_result_pdf["commit_freshness_status"] == ResultStatus.FRESH.value,
        [
            "forecast_diff",
            "predicted_next_data_update",
            "actual_next_data_update",
            "commit_freshness_status",
        ],
    ]

    # short circut if no data point to compute quality metrics
    if len(pdf) < 1:
        return {"rmse": None, "smape": None}

    # forecast quality
    rmse = np.sqrt(np.mean(pdf["forecast_diff"].transform(lambda x: x.total_seconds()) ** 2))
    smape = pm.metrics.smape(pdf["actual_next_data_update"], pdf["predicted_next_data_update"])

    return {"rmse": rmse, "smape": smape}


def backtest_prediction_quality(
    backtest_result_pdf: pd.DataFrame,
    history_pdf: pd.DataFrame,
    known_anomaly_timestamps: List[datetime],
) -> Dict[str, float]:
    # make a copy and label the anomaly update timestamps
    history_with_label = history_pdf.loc[:, ["timestamp"]]
    known_anomaly_timestamps_unix = [int(t.timestamp()) for t in known_anomaly_timestamps]
    history_with_label["label"] = history_pdf.loc[:, "timestamp"].apply(
        lambda x: int(x.timestamp()) in known_anomaly_timestamps_unix
    )

    # use the model prediction to annotate whether the updates were delayed / on time
    delayed_update_timestamps = (
        backtest_result_pdf.loc[
            backtest_result_pdf.result == ResultStatus.STALE.value, "actual_next_data_update"
        ]
        .dropna()
        .unique()
    )
    history_with_label["prediction"] = history_with_label.loc[:, "timestamp"].apply(
        lambda x: x in delayed_update_timestamps
    )

    # prediction quality
    tp = ((history_with_label.prediction is True) & (history_with_label.label)).sum()
    fp = ((history_with_label.prediction is True) & (history_with_label.label)).sum()
    fn = ((history_with_label.prediction is False) & (history_with_label.label)).sum()
    tn = ((history_with_label.prediction is False) & (history_with_label.label)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0  # False Negative Rate

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "true_negative_rate": tnr,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }
