"""
This module provides util methods that are common across different metrics
"""

import pmdarima as pm
import numpy as np
import pandas as pd
import logging
from pyspark.sql import types as T, DataFrame
import pyspark.sql.functions as F
from dataclasses import replace

from typing import List, Dict, Optional, Tuple, Any, Union

from databricks.data_monitoring.anomalydetection.context import Context
from databricks.data_monitoring.anomalydetection import model_config
from databricks.data_monitoring.anomalydetection import freshness_info
from databricks.data_monitoring.anomalydetection import completeness_info
from databricks.data_monitoring.anomalydetection.metrics import AnomalyDetectionMetrics
from databricks.data_monitoring.anomalydetection import errors
import databricks.data_monitoring.anomalydetection.model_config as model_config_utils
from databricks.data_monitoring.anomalydetection.utils.logging_table_utils import (
    DOWNSTREAM_IMPACT_COLUMN_NAME,
)

from databricks.sdk.service.catalog import TableInfo, TableType
from databricks.sdk.errors import DatabricksError

_logger = logging.getLogger(__name__)

# Maps to a 95% confidence interval
DEFAULT_SENSITIVITY = 0.05


def sort_current_run_logging_table(current_run_sdf: DataFrame) -> DataFrame:
    """
    Sort the current run logging table based on the following criteria:
    1. Status: order - Freshness status Ordering + Completeness status ordering
    2. Downstream impact (impact level): order - [4 --> 1]
    3. Quality Check Type: order - Freshness, Completeness
    4. Table Name order: alphabetically
    """
    # Order for quality check types
    check_order = ["Freshness", "Completeness"]
    # Generate mappings for order rankings
    check_order_mapping = {v: i for i, v in enumerate(check_order)}

    # Generate mappings for health status rankings (freshness and completeness)
    # TODO: when we migrate fresh --> healthy, we can remove this mapping and just have ordering of unhealthy, healthy, unknown
    freshness_mapping = {v: i for i, v in enumerate(freshness_info.RESULT_STATUS_ORDER)}
    completeness_mapping = {v: i for i, v in enumerate(completeness_info.COMPLETENESS_STATUS_ORDER)}
    health_status_mapping = {**freshness_mapping, **completeness_mapping}

    # Helper function to create CASE WHEN expressions for ranking
    def generate_case_expr(column: str, order_mapping: dict) -> F.Column:
        case_expr = (
            "CASE "
            + " ".join(
                [f"WHEN {column} = '{value}' THEN {rank}" for value, rank in order_mapping.items()]
            )
            + " ELSE 999 END"
        )
        return F.expr(case_expr)

    # Add ranking columns
    sorted_sdf = current_run_sdf.withColumn(
        "quality_check_rank", generate_case_expr("quality_check_type", check_order_mapping)
    ).withColumn("status_rank", generate_case_expr("status", health_status_mapping))
    # Check if `downstream_impact` exists before using it
    if DOWNSTREAM_IMPACT_COLUMN_NAME in current_run_sdf.columns:
        sorted_sdf = sorted_sdf.withColumn(
            "num_queries_rank",
            -F.when(
                F.col(DOWNSTREAM_IMPACT_COLUMN_NAME).isNotNull()
                & F.col(
                    f"{DOWNSTREAM_IMPACT_COLUMN_NAME}.num_queries_on_affected_tables"
                ).isNotNull(),
                F.col(f"{DOWNSTREAM_IMPACT_COLUMN_NAME}.num_queries_on_affected_tables"),
            ).otherwise(-1),  # Assign lowest priority if num_queries_on_affected_tables is None
        )
    else:
        sorted_sdf = sorted_sdf.withColumn(
            "num_queries_rank", F.lit(-1)
        )  # Default impact rank when column is missing

    # Sort and remove helper columns
    sorted_sdf = sorted_sdf.orderBy(
        "status_rank", "num_queries_rank", "quality_check_rank", F.col("table_name").asc()
    ).drop("quality_check_rank", "status_rank", "num_queries_rank")

    return sorted_sdf


def get_tables_to_eval(
    catalog_name: str,
    schema_name: str,
    should_omit_uc_columns: bool,
    tables_to_scan: List[str],
    tables_to_skip: List[str],
    logging_table_full_name: str,
) -> Tuple[List[TableInfo], List[TableInfo]]:
    """
    Get the tables to evaluate in the schema

    :return: A tuple of tables to evaluate and view tables
    """

    # Filter out logging table from tables_to_eval if logging table defined is in same schema
    logging_table_full_name = logging_table_full_name.replace("`", "")

    logging_table_name = None
    if f"{catalog_name}.{schema_name}" in logging_table_full_name:
        logging_table_name = logging_table_full_name.split(".")[-1]

    w = Context.current.get_workspace_client()
    tables_list = list(
        w.tables.list(
            catalog_name=catalog_name,
            schema_name=schema_name,
            include_delta_metadata=False,
            omit_columns=should_omit_uc_columns,
            omit_properties=True,
        )
    )

    # Respect the tables_to_scan or tables_to_skip configurations
    if len(tables_to_scan) > 0:
        tables_set = set(tables_to_scan)
        tables_list = [table for table in tables_list if table.name in tables_set]
    if len(tables_to_skip) > 0:
        tables_set = set(tables_to_skip)
        tables_list = [table for table in tables_list if table.name not in tables_set]

    # Filter out summary delta table and metric tables generated from LHM
    view_tables = []
    tables_to_eval = []
    for table in tables_list:
        if table.table_type == TableType.VIEW or table.table_type == TableType.MATERIALIZED_VIEW:
            view_tables.append(table)
        elif table.name != logging_table_name:
            tables_to_eval.append(table)
    return tables_to_eval, view_tables


# get the workspace url from the context
def get_workspace_url():
    ctx = Context.current.get_dbutils().notebook.entry_point.getDbutils().notebook().getContext()
    api_url = ctx.apiUrl().get()
    hostname = ctx.browserHostName()
    return f"https://{hostname.get()}" if hostname.isDefined() else api_url


# get lineage link for table
def get_lineage_link(full_table_name):
    format_table_name = full_table_name.replace(".", "/")
    workspace_url = get_workspace_url()
    lineage_link = f"explore/data/{format_table_name}?activeTab=lineage"
    link = f"{workspace_url}/{lineage_link}"
    return link


# Aims to generate jittered synthetic data based on the train data
# Returns the modified train data and a boolean indicating whether synthetic data was actually added or not
def generate_jittered_synthetic_data(
    model_config: model_config.AutoArimaModelConfig,
    train_duration_list: np.ndarray,
    seasonal_period: int,
) -> Tuple[np.ndarray, bool]:
    # determine how many points to generate until we reach the sufficient training points
    max_synthetic_points = model_config.sufficient_training_points - len(train_duration_list)

    if max_synthetic_points <= 0:
        return train_duration_list, False

    # set seed for reproducability
    np.random.seed(0)

    # if train_duration_list is less tham max_synth_points, whole array is considered
    train_duration_list = train_duration_list[:max_synthetic_points]

    # consider how many points which make up complete peropdic cycles (number is evenly divisible by seasonal period)
    num_points_to_copy = len(train_duration_list) - (len(train_duration_list) % seasonal_period)

    # ensure we have non-zero data to copy
    if num_points_to_copy == 0:
        return train_duration_list, False

    # how many sets of points to copy will we add to the train data
    num_of_sets_of_copies = min(
        max_synthetic_points // num_points_to_copy, model_config.jitter_repetitions
    )

    # restrict to fist num_points_to_copy
    train_snippet = train_duration_list[:num_points_to_copy]

    # the amount of jitter is relative to the median duration difference in the train data
    median = np.median(train_duration_list)
    jitter_amount = min(
        median // 10, 600
    )  # Adjust this to control the jitter level (max will be 600 = 10 minutes)

    jitter_arr = np.tile(train_snippet, num_of_sets_of_copies)
    jitter_arr += np.random.randint(-jitter_amount, jitter_amount + 1, jitter_arr.shape)
    combined_jitter_train_arr = np.append(jitter_arr.flatten(), train_duration_list)

    return combined_jitter_train_arr, True


def infer_seasonality_auto_arima(
    train: np.ndarray,
    model_config: model_config.AutoArimaModelConfig,
) -> Tuple[np.ndarray, Optional[int], Dict[str, Any]]:
    # set seasonal period as None; will be overridden if seasonality is inferred
    res_model_config = {}
    seasonal_period = None
    if model_config.infer_seasonality_acf:
        try:
            # Compute autocorrelation function values on the training data
            # with maximum lag of `acf_max_lag` (i.e. the seasonal period to look for
            # is at most `acf_max_lag`).
            # `train_acf` is a list of autocorrelation values for lag [0, nlags], where
            # lag 0 means the correlation of the series with itself.
            train_acf = pm.utils.acf(train, nlags=min(model_config.acf_max_lag, len(train)))

            # Skip the first element since auto correlation of itself is always 1,
            # Look for the next peak in the ACF values to find the most likely
            # period that this data is repeating.
            period = np.argmax(train_acf[1:]) + 1
            acf_val = train_acf[period]

            # captures the initial predicted value before we boost seasonal period acf val
            res_model_config["pre_boost_proposed_seasonal_period"] = period
            res_model_config["pre_boost_proposed_acf_val"] = acf_val

            # whether we generate syntehtic data to boost seasonal period acf
            generated_synthetic_data = False
            if model_config.boost_seasonal_period_acf:
                jittered_train, generated_synthetic_data = generate_jittered_synthetic_data(
                    model_config=model_config, train_duration_list=train, seasonal_period=period
                )

                # we actually added synthetic data --> recompute acf vals and period
                if generated_synthetic_data:
                    # re-compute acf given the generated synthetic_data. Lag should not be greater than the original data size given.
                    j_train_acf = pm.utils.acf(jittered_train, nlags=min(52, len(train)))
                    period = np.argmax(j_train_acf[1:]) + 1
                    acf_val = j_train_acf[period]

                    # whether we use the jittered train data for model fitting
                    if model_config.train_on_jittered_data:
                        train = jittered_train  # sets training data to be jittered data

            # set whether we ended up adding synthetic data or not
            res_model_config["generated_synthetic_data"] = generated_synthetic_data

            res_model_config["final_proposed_seasonal_period"] = period
            res_model_config["final_proposed_acf_val"] = acf_val

            # If the ACF value is below the threshold, we consider it non-significant
            # and will not fit a seasonal model on the data.
            if acf_val < model_config.acf_threshold:
                seasonal_period = None
            else:
                seasonal_period = period

        except Exception:
            # If ACF fails, default to not fitting a seasonal model
            seasonal_period = None

    # generate jittered synthetic data for non-seasonal case if we do not so in infer seasonality case
    elif model_config.train_on_jittered_data:
        train, generated_synthetic_data = generate_jittered_synthetic_data(
            model_config=model_config, train_duration_list=train, seasonal_period=1
        )
        res_model_config["generated_synthetic_data"] = generated_synthetic_data

    return train, seasonal_period, res_model_config


# returns a model and an optional preprocessing feature transform
def fit_auto_arima(
    train: np.ndarray,
    effective_seasonal_period: int = 1,
    differencing_order: int = 0,
    use_fourier_transform: bool = False,
    with_intercept: bool = False,
    model_params: Optional[model_config_utils.ModelParams] = None,
) -> Optional[model_config_utils.ModelParams]:
    if train is None or effective_seasonal_period is None:
        return None
    try:
        y = train  # holds the durations for training
        X = None  # holds the exogenous array
        trans = None  # transformation obj
        m = effective_seasonal_period  # seasonality of the data
        min_val = np.min(train)

        if model_params and model_params.model:
            model = model_params.model
            trans = model_params.transform
            m = model_params.seasonal_period
            if trans:
                y, X = trans.update_and_transform(train)
            model.update(y, X=X)

        else:
            # use fourier transform to capture seasonality
            if effective_seasonal_period != 1 and use_fourier_transform:
                # FourierFeaturizer takes 2 params: m = seasonal period, k = The number of sine and cosine terms (each) to include
                # k defaulted to m / 2 and should never be greater than that.
                # The seasonal pattern is smooth for small values of K (but more wiggly seasonality can be handled by increasing K)
                trans = pm.preprocessing.FourierFeaturizer(m=effective_seasonal_period)
                y, X = trans.fit_transform(train)

                # use regular arima (no seasonality) after transforming the data with fourier
                m = effective_seasonal_period = 1

            model = pm.auto_arima(
                y,
                X=X,
                start_p=1,
                start_q=1,
                d=differencing_order,
                m=effective_seasonal_period,
                with_intercept=with_intercept,
                seasonal=effective_seasonal_period > 1,
                trace=False,
                suppress_warnings=True,
                error_action="ignore",  # Suppress noisy logs where ARIMA fails to fit
            )
        # if no fourier transform, auto arima (for seasonal and non-seasonal) will be used
        return model_config_utils.ModelParams(
            model=model,
            transform=trans,
            seasonal_period=m,
            min_val=min_val,
            last_update_timestamp=model_params.last_update_timestamp if model_params else None,
        )
    except Exception:
        return None


# Fit multiple models and return the best performing one (seasonal vs non-seasonal model)
def fit_multiple_models(
    train: np.ndarray,
    model_config: model_config.AutoArimaModelConfig,
) -> Tuple[model_config_utils.ModelParams, Dict[str, Any]]:
    # infer seasonal period and generate synthetic data if needed
    train, seasonal_period, res_model_config = infer_seasonality_auto_arima(train, model_config)

    # fit a non-seasonal model and seasonal model (if applicable)
    model_results = {}
    model_results["seasonal"] = fit_auto_arima(
        train=train,
        effective_seasonal_period=seasonal_period,
        differencing_order=model_config.differencing_order,
        use_fourier_transform=model_config.use_fourier_transform,
        with_intercept=True,
    )
    model_results["non_seasonal"] = fit_auto_arima(
        train=train,
        effective_seasonal_period=1,
        differencing_order=model_config.differencing_order,
        use_fourier_transform=False,
        with_intercept=True,
    )

    # fail to fit both models
    if model_results["seasonal"] is None and model_results["non_seasonal"] is None:
        raise ValueError(errors.ERROR_CODE_TO_MESSAGE[errors.ErrorCode.FAILED_TO_FIT_MODEL])

    # pick the best performing model based on BIC metric
    selected_model_params = None
    if model_results["seasonal"] and model_results["non_seasonal"]:
        # get bic values on train data for both models
        seasonal_bic = model_results["seasonal"].model.bic()
        non_seasonal_bic = model_results["non_seasonal"].model.bic()

        selected_model_params = (
            model_results["seasonal"]
            if seasonal_bic < non_seasonal_bic
            else model_results["non_seasonal"]
        )
        res_model_config["effective_seasonal_period"] = (
            seasonal_period if seasonal_bic < non_seasonal_bic else 1
        )
        res_model_config["bic_value_seasonal"] = seasonal_bic
        res_model_config["bic_value_non_seasonal"] = non_seasonal_bic
    else:
        # one of the models is None
        selected_model_params = model_results["seasonal"] or model_results["non_seasonal"]
        res_model_config["effective_seasonal_period"] = (
            seasonal_period if model_results["seasonal"] else 1
        )

    # save model params
    res_model_config["final_selected_model"] = (
        "non_seasonal" if res_model_config["effective_seasonal_period"] == 1 else "seasonal"
    )

    return selected_model_params, res_model_config


def update_model(
    train: np.ndarray,
    model_config: model_config.AutoArimaModelConfig,
    model_params: model_config_utils.ModelParams,
) -> model_config_utils.ModelParams:
    # reuse same model
    if len(train) == 0:
        return model_params

    updated_model_params = fit_auto_arima(
        train=train,
        effective_seasonal_period=model_config.seasonal_period,
        differencing_order=model_config.differencing_order,
        model_params=model_params,
    )
    if updated_model_params is None:
        raise ValueError(errors.ERROR_CODE_TO_MESSAGE[errors.ErrorCode.FAILED_TO_FIT_MODEL])
    return updated_model_params


# Generates a forecast and Confidence interval given a model, periods to forecast, and alpha value
def predict_auto_arima(
    model_params: model_config_utils.ModelParams, n_periods: int, alpha: float = 0.05
) -> Tuple[float, Tuple[float, float]]:
    config = model_config.get_model_config().autoarima_model_config
    if config is None:
        raise ValueError(f"Unsupported model type: {model_config.get_model_type()}")
    try:
        # Forecast the next duration
        n_periods = 1
        forecast = None
        conf_int = None
        if model_params.transform is not None:
            # gets exogenous array for the forecast (pass in None for y and X)
            _, X_future = model_params.transform.transform(None, None, n_periods=n_periods)
            forecast, conf_int = model_params.model.predict(
                n_periods=n_periods, X=X_future, return_conf_int=True, alpha=alpha
            )
            if config.use_std_residuals:
                # Get residuals using actual - fitted values from training data
                residuals = model_params.model.resid()
                std_residuals = np.std(residuals)
                num_residuals = config.num_std_for_residuals
                conf_int = np.array(
                    [[-num_residuals * std_residuals, num_residuals * std_residuals]]
                )
            # Read first row from forecast (when training modelfrom scratch, forecast is a pd, but after model udpates its a np array)
            forecast = (
                float(forecast.iloc[0]) if isinstance(forecast, pd.Series) else float(forecast[0])
            )
        else:
            forecast, conf_int = model_params.model.predict(
                n_periods=n_periods, return_conf_int=True, alpha=alpha
            )
            # extract forecast from array
            forecast = forecast[0]

        return forecast, conf_int
    except Exception:
        raise ValueError(errors.ERROR_CODE_TO_MESSAGE[errors.ErrorCode.FAILED_TO_FIT_MODEL])


def log_results(
    summary_pdf: pd.DataFrame,
    logging_table_full_name: str,
    logging_table_schema: T.StructType,
    enable_print: bool = True,
):
    if enable_print:
        print(f"Logging to {logging_table_full_name}")

    summary_pdf = summary_pdf[logging_table_schema.fieldNames()]

    try:
        df = Context.spark.createDataFrame(
            summary_pdf,
            schema=logging_table_schema,
        )
        df.write.mode("append").option("mergeSchema", "true").saveAsTable(logging_table_full_name)
    except Exception as e:
        print(f"Logging failed due to error: {e}")
        raise e


def get_token():
    return (
        Context.current.get_dbutils()
        .notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .apiToken()
        .get()
    )


def make_api_call(method_name: str, api_path: str, body: Optional[Dict] = None) -> Dict:
    """Make an API call to the Databricks API"""
    w = Context.current.get_workspace_client()

    # Retrieve the Databricks token
    token = get_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        return w.api_client.do(
            method_name,
            api_path,
            headers=headers,
            body=body,
        )
    except DatabricksError as db_err:
        _logger.info(f"Databricks error occurred: {db_err}")  # Status code and message
    except Exception as e:
        _logger.info(f"An unexpected error occurred: {e}")

    raise Exception("An internal error has occurred. Please contact support.")


def send_anomaly_detection_metrics(metrics: AnomalyDetectionMetrics, support_retry: bool = True):
    """
    Sends anomaly detection metrics to the Databricks API endpoint.

    :param metrics: The anomaly detection metrics to be sent.
    """
    try:
        make_api_call(
            method_name="POST",
            api_path="/api/2.1/quality-monitoring/anomaly-detection-metrics",
            body=metrics.to_json_dict(),
        )
        _logger.info("Metrics sent successfully.")
    except Exception as e:
        if support_retry:
            # Clear metrics and error messages
            table_freshness_infos = metrics.table_freshness_infos
            for info in table_freshness_infos:
                info["commit_timestamps"] = []
                info["error_message"] = "Redacted" if info["error_message"] else None
            redacted_metrics = replace(metrics, table_freshness_infos=table_freshness_infos)
            send_anomaly_detection_metrics(redacted_metrics, support_retry=False)
        else:
            _logger.info(f"Metrics failed to send: {e}")
            # Swallow exception


def check_feature_enabled():
    """
    Checks if the feature is enabled
    """
    response = make_api_call(
        method_name="GET", api_path="/api/2.1/quality-monitoring/bootstrap-info"
    )
    if response["eligibility"] != "AD_SUPPORTED":
        raise Exception(f"Feature is not enabled: {response['error_reasons']}")
    else:
        # set enable_put_health_checks flag based on value returned from bootstrap info
        model_config_utils.set_enable_put_health_checks(
            response.get("enable_python_client_put_health_checks", False)
        )
        # set enable_blast_radius_computation flag based on value returned from bootstrap info
        model_config_utils.set_enable_blast_radius_computation(
            response.get("enable_blast_radius_computation", False)
        )
        # set enable_dashboard flag based on value returned from bootstrap info
        model_config_utils.set_enable_dashboard(response.get("enable_dashboard", False))
        # set enable_backtesting flag based on value returned from bootstrap info
        model_config_utils.set_enable_backtesting(response.get("enable_backtesting", False))
        # set enable_limit_history_by_timestamp flag based on value returned from bootstrap info
        model_config_utils.set_enable_limit_history_by_timestamp(
            response.get("enable_limit_history_by_timestamp", False)
        )


def transform_to_display_schema(df: pd.DataFrame, exclude_fields: List[str]) -> pd.DataFrame:
    """
    Transforms a DataFrame with the logging schema into a DataFrame with the display schema,
    allowing for customizable exclusion of specific debug fields.

    Args:
        df: DataFrame with the logging schema.
        exclude_fields: List of field names to exclude from `additional_debug_info`.

    Returns:
        DataFrame with the display schema.
    """

    def remove_debug_keys(debug_info):
        if not isinstance(debug_info, dict):
            return debug_info  # Ensure robustness if data is malformed

        cleaned_debug_info = {}
        for key, value in debug_info.items():
            if isinstance(value, dict):
                # Remove specified fields from debug information
                cleaned_value = {k: v for k, v in value.items() if k not in exclude_fields}
                cleaned_debug_info[key] = cleaned_value
            else:
                cleaned_debug_info[key] = value

        return cleaned_debug_info

    ret_df = df.copy()
    ret_df["additional_debug_info"] = ret_df["additional_debug_info"].apply(remove_debug_keys)
    return ret_df
