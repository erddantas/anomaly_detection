"""
This module provides the client facing entrypoint into checking for freshness anomalies
"""

import time

import concurrent.futures
import json
import pandas as pd
import logging
import random
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict

from databricks.data_monitoring.anomalydetection.base_checker import BaseChecker
from databricks.data_monitoring.anomalydetection.utils.logging_table_utils import (
    LOGGING_TABLE_FIELDS,
    LOGGING_TABLE_SCHEMA,
    DOWNSTREAM_IMPACT_COLUMN_NAME,
)
from databricks.data_monitoring.anomalydetection.metrics import (
    AnomalyDetectionMetrics,
    HealthChecks,
)
from databricks.data_monitoring.anomalydetection import (
    freshness_utils as utils,
    errors,
)
from databricks.data_monitoring.anomalydetection.utils import (
    table_history_utils as history_utils,
    common_utils,
)
from databricks.data_monitoring.anomalydetection.visualization import visualization as vis_utils
from databricks.data_monitoring.anomalydetection.context import Context
from databricks.data_monitoring.anomalydetection.freshness_info import (
    ResultStatus,
    TableFreshnessInfo,
)
import databricks.data_monitoring.anomalydetection.model_config as model_config_utils
from databricks.sdk.service import jobs
from tqdm import tqdm

_logger = logging.getLogger(__name__)

MAX_FRESHNESS_INFOS_WITH_COMMITS = 100
# TODO: get this as param from boostrap info api call
MAX_HEALTH_CHECKS_BATCH_SIZE = 10


class FreshnessChecker(BaseChecker):
    """
    Checker that in parallel checks for freshness anomalies for all tables in a schema
    """

    def __init__(
        self,
        catalog_name: str,
        schema_name: str,
        tables_to_skip: Optional[List[str]] = None,
        tables_to_scan: Optional[List[str]] = None,
        table_threshold_overrides: Optional[Dict[str, timedelta]] = None,
        table_latency_threshold_overrides: Optional[Dict[str, timedelta]] = None,
        static_table_threshold_override: Optional[timedelta] = None,
        logging_table_name: Optional[str] = None,
        event_timestamp_col_names: Optional[List[str]] = None,
        enable_debug_info: Optional[bool] = False,
    ):
        """
        Initializes the freshness checker

        :param catalog_name: Name of the catalog to check freshness
        :param schema_name: Name of the schema to check freshness
        :param tables_to_skip: (Optional) Tables to ignore in detecting freshness
            anomalies
        :param tables_to_scan: (Optional) Tables to scan for freshness anomalies
        :param table_threshold_overrides: (Optional) Dictionary of table names to
            freshness thresholds
        :param table_latency_threshold_overrides: (Optional) Dictionary of table names to event based latency thresholds
        :param static_table_threshold_override: (Optional) Threshold used to determine whether a table is static.
        :param logging_table_name: (deprecated) Name of the table to log freshness.
            Defaults to _quality_monitoring_summary in the same catalog / schema to check freshness on.
        :param event_timestamp_col_names: (Optional) List of column names representing event timestamps in the schema.
            If a table has a column name in this list, we will evaluate event based freshness for that table.
        :param enable_debug_info: (Optional) Whether to enable debug information in the freshness summary
        """
        super().__init__(
            catalog_name=catalog_name,
            schema_name=schema_name,
            tables_to_skip=tables_to_skip,
            tables_to_scan=tables_to_scan,
            logging_table_name=logging_table_name,
        )
        self._num_tables_in_schema: Optional[int] = None

        # Check that there's no overlap in keys in the threshold overrides
        if table_threshold_overrides and table_latency_threshold_overrides:
            if set(table_threshold_overrides.keys()) & set(
                table_latency_threshold_overrides.keys()
            ):
                raise ValueError(
                    "table_threshold_overrides and table_latency_threshold_overrides cannot have overlapping tables."
                )

        self._table_threshold_overrides = table_threshold_overrides or {}
        self._static_table_threshold_override = static_table_threshold_override
        self._table_latency_threshold_overrides = table_latency_threshold_overrides or {}
        self._schema_full_name = f"{self._catalog_name}.{self._schema_name}"

        self._event_timestamp_col_names = event_timestamp_col_names or []
        self._enable_debug_info = enable_debug_info

    def run_checks(self) -> None:
        self.run_freshness_checks()

    def _build_anomaly_metrics(self, check_results: Dict) -> AnomalyDetectionMetrics:
        """
        Build metrics from check results.

        Args:
            check_results: Dictionary containing check results

        Returns:
            AnomalyDetectionMetrics: Metrics object containing aggregated information
        """
        table_freshness_dicts = []

        # Only include commit timestamps for the first MAX_FRESHNESS_INFOS_WITH_COMMITS tables
        num_infos_with_commits = 0
        # Note: the freshness info is sorted with STALE first and then FRESH, then others.
        for freshness_info in check_results.values():
            freshness_dict = freshness_info.to_metric_dict()
            if num_infos_with_commits < MAX_FRESHNESS_INFOS_WITH_COMMITS:
                num_infos_with_commits += 1
            else:
                del freshness_dict["commit_timestamps"]
            table_freshness_dicts.append(freshness_dict)

        return AnomalyDetectionMetrics(
            catalog_name=self._catalog_name,
            schema_name=self._schema_name,
            threshold_overrides={
                k: v.total_seconds() for k, v in self._table_threshold_overrides.items()
            },
            event_based_threshold_overrides={
                k: v.total_seconds() for k, v in self._table_latency_threshold_overrides.items()
            },
            static_table_threshold_override=self._static_table_threshold_override.total_seconds()
            if self._static_table_threshold_override
            else None,
            disabled_tables=self._tables_to_skip,
            table_freshness_infos=table_freshness_dicts,
            num_tables_in_schema=self._num_tables_in_schema,
        )

    def run_freshness_checks(self, visualize_results: bool = True) -> None:
        """
        Runs the freshness checks for all tables in the schema
        :param visualize_results: (Optional) Whether to visualize the results
        """
        freshness_summary = self._check_freshness_no_udfs()
        self._log_and_display_results(freshness_summary, visualize_results)

    def run_backtesting(self) -> None:
        """
        Runs backtesting for all tables in the schema to help gain insight and trends of health of tables at different times in the past.
        """

        tables_to_eval, _ = common_utils.get_tables_to_eval(
            self._catalog_name,
            self._schema_name,
            len(self._event_timestamp_col_names) == 0,
            self._tables_to_scan,
            self._tables_to_skip,
            self._logging_table_full_name,
        )
        table_names = [table.name for table in tables_to_eval]

        # Set up progress bar with total number of tables
        pbar = tqdm(total=len(table_names), desc="Backtesting: Checking table freshness")

        table_eval_erors = []

        # Parallelize the backtesting for each table
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_table = {
                executor.submit(self._single_backtest, table_name): table_name
                for table_name in table_names
            }

            for future in concurrent.futures.as_completed(future_to_table):
                table_name = future_to_table[future]
                try:
                    # write out the results for table's backtest
                    logging_df = future.result()
                    common_utils.log_results(
                        logging_df,
                        self._logging_table_full_name,
                        LOGGING_TABLE_SCHEMA,
                        enable_print=False,
                    )
                except Exception as e:
                    table_eval_erors.append(f"Backtesting for table {table_name} failed: {e}")

                pbar.update(1)
        pbar.close()

        for error in table_eval_erors:
            print(error)

        print(f"Logging to {self._logging_table_full_name}")

    def _single_backtest(self, table_name: str = None) -> pd.DataFrame:
        """
        Runs a backtest on the table in the schema. This is equivalent to time travel to prior timestamps
        and checking health status of the table.

        :param table_name: Name of the table to run backtest.

        :return: A backtest result spark dataframe of the table.
        """
        # (1) Get table history if not provided only if boolean is true
        history_result = history_utils.get_table_history(
            table_name,
            self._catalog_name,
            self._schema_name,
            should_limit=False,
            # multiple max backtesting lookback by 2 so you at first eval points you have data to train on
            days_limit=model_config_utils.get_max_lookback_days() * 2,
        )

        # if we can't grab history, throw error
        if history_result.error_message is not None or history_result.history_df is None:
            raise ValueError(history_result.error_message)

        history_pdf = history_result.history_df
        df_processed = utils.preprocess_ml_data(history_pdf, drop_na=True)
        # Check we have enough points for backtesting
        if len(df_processed) <= model_config_utils.get_min_commit_training_points():
            raise ValueError(
                errors.ERROR_CODE_TO_MESSAGE[errors.ErrorCode.NOT_ENOUGH_UPDATE_OP_BACKTESTING]
            )

        # (2) Determine start and end time
        eval_time = datetime.fromtimestamp(history_result.evaluation_epoch_sec)
        # Find first timestamp at which we have min number of points
        earliest_backtest_timestamp = df_processed["timestamp"].iloc[
            model_config_utils.get_min_commit_training_points()
        ]

        # we will look back the max backtesting lookback days
        farthest_backtest_timestamp = eval_time - timedelta(
            days=model_config_utils.get_max_lookback_days()
        )
        start_timestamp = max(earliest_backtest_timestamp, farthest_backtest_timestamp)
        end_timestamp = eval_time

        # (3) Determine cadence of updates
        check_interval_hr = self._backtest_avg_cadence_bucket(df_processed)

        # Get table status for all backtesting
        result_list = utils.rolling_forecast_single_table(
            df_processed,
            self._catalog_name,
            self._schema_name,
            table_name,
            start_timestamp,
            end_timestamp,
            check_interval_hr,
            self._table_threshold_overrides,
            self._table_latency_threshold_overrides,
            self._static_table_threshold_override,
            table_lineage_link=common_utils.get_lineage_link(
                f"{self._catalog_name}.{self._schema_name}.{table_name}"
            ),
        )

        # Convert to spark dataframe
        logging_df = self._transform_backtesting_to_logging_schema(result_list)

        return logging_df

    def _backtest_avg_cadence_bucket(self, df_processed: pd.DataFrame):
        """
        Returns number of hours we should evaluate the table at based on its average cadence.
        """
        time_differences = df_processed["duration_to_next_timestamp"]
        median_difference_seconds = time_differences.median()
        median_diff_timedelta = timedelta(seconds=median_difference_seconds)

        if median_diff_timedelta <= timedelta(hours=12):
            # frequently updates tables
            return 6
        elif median_diff_timedelta <= timedelta(days=3.5):
            # somewhat daily tables
            return 24
        else:
            # weekly or longer udpated tables
            median_diff_timedelta < timedelta(weeks=1)
            return 7 * 24

    def create_job(
        self,
        email_notifications: List[str] = [],
        interval_hours: Optional[int] = None,
        whl_override: Optional[str] = None,
    ):
        """
        Creates and triggers a Databricks notebook job from a Git provider to perform a freshness check.
        Source of truth for wheel version is github notebook.
        :param interval_hours: (Optional) Schedule frequency in hour. 6 hours by default.
        :param whl_override: (Internal only) whl url override. This is the whl version that freshness check running on
        :return: A scheduled job that runs freshness check.
        """
        DEFAULT_INTERVAL_HOURS = 6
        w = Context.current.get_workspace_client()
        workspace_url = common_utils.get_workspace_url()
        try:
            created_job = w.jobs.create(
                name=f"[Freshness Monitoring] on schema {self._schema_full_name}",
                tasks=[
                    jobs.Task(
                        description=f"Job that monitor freshness on {self._schema_full_name} schema and log results into {self._logging_table_full_name}",
                        notebook_task=jobs.NotebookTask(
                            notebook_path="freshness_monitoring",
                            base_parameters={
                                "catalog_name": self._catalog_name,
                                "schema_name": self._schema_name,
                                "logging_table_name": self._logging_table_name,
                                # use notebook default wheel if no whl_override is provided
                                "whl_override": whl_override if whl_override is not None else "",
                                # convert from list to string, ex: ["t1", "t2"] to 't1, t2'
                                "tables_to_skip": ", ".join(
                                    f"{item}" for item in self._tables_to_skip
                                )
                                if self._tables_to_skip
                                else "",
                                "tables_to_scan": ", ".join(
                                    f"{item}" for item in self._tables_to_scan
                                )
                                if self._tables_to_scan
                                else "",
                                "table_threshold_overrides": json.dumps(
                                    {
                                        k: v.total_seconds()
                                        for k, v in self._table_threshold_overrides.items()
                                    }
                                )
                                if self._table_threshold_overrides
                                else "",
                                "table_latency_threshold_overrides": json.dumps(
                                    {
                                        k: v.total_seconds()
                                        for k, v in self._table_latency_threshold_overrides.items()
                                    }
                                )
                                if self._table_latency_threshold_overrides
                                else "",
                                "static_table_threshold_override": self._static_table_threshold_override.total_seconds()
                                if self._static_table_threshold_override
                                else "",
                                "event_timestamp_col_names": ", ".join(
                                    self._event_timestamp_col_names
                                )
                                if self._event_timestamp_col_names
                                else "",
                            },
                            source=jobs.Source.GIT,
                        ),
                        task_key="freshness_monitoring",
                        timeout_seconds=0,
                        max_retries=0,  # explicitly sets max retries to 0 (default value) for job
                        disable_auto_optimization=True,  # disbales auto optimization (retries) for job
                    )
                ],
                email_notifications=jobs.JobEmailNotifications(on_failure=email_notifications),
                trigger=jobs.TriggerSettings(
                    pause_status=jobs.PauseStatus.UNPAUSED,
                    periodic=jobs.PeriodicTriggerConfiguration(
                        interval=interval_hours
                        if interval_hours is not None
                        else DEFAULT_INTERVAL_HOURS,
                        unit=jobs.PeriodicTriggerConfigurationTimeUnit.HOURS,
                    ),
                ),
                git_source=jobs.GitSource(
                    git_url="https://github.com/databricks/expectations",
                    git_provider=jobs.GitProvider.GIT_HUB,
                    git_branch="main",
                ),
            )

            w.jobs.run_now(job_id=created_job.job_id)
            print(
                f"Job created and triggered successfully! You can view the job details here: {workspace_url}#job/{created_job.job_id}"
            )
        except:
            print(f"Failed to create and trigger the job: {traceback.print_exc()}")
        return

    # splits the freshness summary report into 3 tables: an overall freshness summary, commit based debug info, and
    # event based debug info
    def _format_and_print_freshness_summary(self, freshness_summary_pdf: pd.DataFrame):
        # Coerce into schema in case values are all errors (schema inference would fail)
        sdf = Context.spark.createDataFrame(
            freshness_summary_pdf,
            schema=LOGGING_TABLE_FIELDS,
        )
        Context.display(sdf)

    def _build_anomaly_metrics(
        self, sorted_freshness_infos: Dict[str, TableFreshnessInfo]
    ) -> List[Dict]:
        table_freshness_dicts = []

        # Only include commit timestamps for the first MAX_FRESHNESS_INFOS_WITH_COMMITS tables
        num_infos_with_commits = 0
        # Note: the freshness info is sorted with STALE first and then FRESH, then others.
        for freshness_info in sorted_freshness_infos.values():
            freshness_dict = freshness_info.to_metric_dict()
            if num_infos_with_commits < MAX_FRESHNESS_INFOS_WITH_COMMITS:
                num_infos_with_commits += 1
            else:
                del freshness_dict["commit_timestamps"]
            table_freshness_dicts.append(freshness_dict)

        return table_freshness_dicts

    def _transform_to_logging_schema_helper(
        self, info: TableFreshnessInfo, from_backtesting=False
    ) -> List[Dict]:
        additional_debug_info = {}

        # Add commit staleness info if available
        if info.commit_freshness_status is not None:
            commit_staleness_seconds = info.staleness_age_seconds
            commit_staleness_str = utils.seconds_to_days_hours_minutes(commit_staleness_seconds)

            # Calculate the threshold staleness for commit freshness
            threshold_staleness_seconds = None
            if info.predicted_upper_bound_next_data_update and info.last_data_update:
                threshold_staleness_seconds = int(
                    (
                        info.predicted_upper_bound_next_data_update - info.last_data_update
                    ).total_seconds()
                )

            threshold_staleness_str = utils.seconds_to_days_hours_minutes(
                threshold_staleness_seconds
            )

            commit_debug = {
                "actual_value": commit_staleness_str if commit_staleness_str else "N/A",
                "expectation": f"actual_value < {threshold_staleness_str}"
                if threshold_staleness_str
                else "N/A",
                "is_violated": str(info.commit_freshness_status == ResultStatus.STALE).lower(),
                "error_code": info.error_code.value if info.error_code else "None",
                "is_static": str(info.is_static()).lower(),
                "actual_value_seconds": str(commit_staleness_seconds),
                "expectation_threshold_seconds": str(threshold_staleness_seconds),
                "from_backtesting": str(from_backtesting).lower(),
            }
            additional_debug_info["commit_staleness"] = commit_debug

        # Add event staleness info if available
        if info.event_freshness_status is not None:
            event_staleness_seconds = info.event_staleness_age_seconds
            event_staleness_str = utils.seconds_to_days_hours_minutes(event_staleness_seconds)

            # Get the threshold for event staleness directly from event_predicted_upper_bound_latency_seconds
            event_threshold_str = utils.seconds_to_days_hours_minutes(
                info.event_predicted_upper_bound_latency_seconds
            )

            event_debug = {
                "actual_value": event_staleness_str if event_staleness_str else "N/A",
                "expectation": f"actual_value < {event_threshold_str}"
                if event_threshold_str
                else "N/A",
                "is_violated": str(info.event_freshness_status == ResultStatus.STALE).lower(),
                "error_code": info.error_code.value if info.error_code else "None",
                "actual_value_seconds": str(event_staleness_seconds),
                "expectation_threshold_seconds": str(
                    info.event_predicted_upper_bound_latency_seconds
                ),
            }
            additional_debug_info["event_staleness"] = event_debug

        status = info.overall_freshness_status.value if info.overall_freshness_status else "Unknown"
        # TODO: cleanup once this is rolled out (blocked on backend changes)
        if model_config_utils.get_rename_fresh_stale_to_healthy_unhealthy():
            if status == "Fresh":
                status = "Healthy"
            elif status == "Stale":
                status = "Unhealthy"

        record = {
            "evaluated_at": info.evaluated_at,
            "catalog": self._catalog_name,
            "schema": self._schema_name,
            "table_name": info.table_name,
            "quality_check_type": "Freshness",
            "status": status if info.overall_freshness_status else ResultStatus.UNKNOWN.value,
            DOWNSTREAM_IMPACT_COLUMN_NAME: asdict(info.blast_radius) if info.blast_radius else None,
            "additional_debug_info": additional_debug_info,
            "error_message": info.error_message,
            "table_lineage_link": info.table_lineage_link,
        }

        return record

    def _transform_to_logging_schema(
        self, freshness_summary: Dict[str, TableFreshnessInfo]
    ) -> pd.DataFrame:
        """
        Transforms a dictionary of TableFreshnessInfo objects into a DataFrame with the logging schema.

        Args:
            freshness_summary: Dictionary mapping table names to TableFreshnessInfo objects

        Returns:
            Pandas DataFrame with the logging schema
        """
        records = []

        for table_name, info in freshness_summary.items():
            records.append(self._transform_to_logging_schema_helper(info))

        return pd.DataFrame(records)

    def _transform_backtesting_to_logging_schema(
        self, backtesting_results: List[TableFreshnessInfo]
    ) -> pd.DataFrame:
        records = []
        for backtest_info in backtesting_results:
            record = self._transform_to_logging_schema_helper(backtest_info, from_backtesting=True)
            records.append(record)

        return pd.DataFrame(records)

    def _log_and_display_results(
        self, sorted_freshness_summary: Dict[str, TableFreshnessInfo], visualize_results: bool
    ):
        logging_df = self._transform_to_logging_schema(sorted_freshness_summary)

        # Log results to logging table
        self._min_run_eval_time = logging_df["evaluated_at"].min()
        common_utils.log_results(logging_df, self._logging_table_full_name, LOGGING_TABLE_SCHEMA)

        table_freshness_dicts = self._build_anomaly_metrics(sorted_freshness_summary)

        anomaly_detection_metrics = AnomalyDetectionMetrics(
            # TODO populate job id and run id
            catalog_name=self._catalog_name,
            schema_name=self._schema_name,
            threshold_overrides={
                k: v.total_seconds() for k, v in self._table_threshold_overrides.items()
            },
            event_based_threshold_overrides={
                k: v.total_seconds() for k, v in self._table_latency_threshold_overrides.items()
            },
            static_table_threshold_override=self._static_table_threshold_override.total_seconds()
            if self._static_table_threshold_override
            else None,
            disabled_tables=self._tables_to_skip,
            table_freshness_infos=table_freshness_dicts,
            num_tables_in_schema=self._num_tables_in_schema,
        )
        # Log results to service
        # TODO: log model config

        common_utils.send_anomaly_detection_metrics(anomaly_detection_metrics)

        # only show results if dashboard has not been released yet.
        if not model_config_utils.get_enable_dashboard() and visualize_results:
            vis_utils.plot_freshness_summary(sorted_freshness_summary)

        # Report result
        num_stale_tables = sum(
            [
                freshness_info.overall_freshness_status == ResultStatus.STALE
                for freshness_info in sorted_freshness_summary.values()
            ]
        )
        num_total_tables = len(sorted_freshness_summary)

        print(
            f"Found {num_stale_tables} unhealthy tables out of the {num_total_tables} "
            f"tables in schema {self._catalog_name}.{self._schema_name}."
        )

    def _check_freshness_no_udfs(self) -> Dict[str, TableFreshnessInfo]:
        """Check the freshness of all tables in the schema without using udfs

        Returns:
            Dict[str, TableFreshnessInfo]: A dictionary of table names to
                whether tables are stale, sorted so stale tables show first
        """
        tables_to_eval, _ = common_utils.get_tables_to_eval(
            self._catalog_name,
            self._schema_name,
            len(self._event_timestamp_col_names) == 0,
            self._tables_to_scan,
            self._tables_to_skip,
            self._logging_table_full_name,
        )
        self._num_tables_in_schema = len(tables_to_eval)

        table_columns_map = {
            table.name: [col.name for col in table.columns] if table.columns else []
            for table in tables_to_eval
        }

        table_name_to_id_map = {table.name: table.table_id for table in tables_to_eval}

        table_freshness_results = {}
        table_freshness_batch = []
        batch_index = 0
        # Set up progress bar with total number of tables
        pbar = tqdm(total=len(tables_to_eval), desc="Checking table freshness")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_table = {
                executor.submit(
                    self._test_single_table,
                    table.name,
                    next(
                        (
                            col
                            for col in self._event_timestamp_col_names
                            if col in table_columns_map.get(table.name, [])
                        ),
                        None,
                    ),
                ): table.name
                for table in tables_to_eval
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_table):
                freshness_info = future.result()
                if freshness_info is not None:
                    assert freshness_info.table_name is not None
                    freshness_info.table_id = table_name_to_id_map[freshness_info.table_name]
                    table_freshness_results[freshness_info.table_name] = freshness_info

                    # onlt put health checks if its enabled in the model config
                    if model_config_utils.get_enable_put_health_checks():
                        table_freshness_batch.append(freshness_info.to_metric_dict())

                        # Send Health Checks to Backend
                        # Note: no need to include commit timestamps since they are not needed for health checks
                        if len(table_freshness_batch) % MAX_HEALTH_CHECKS_BATCH_SIZE == 0:
                            self._put_health_checks_batch(table_freshness_batch, batch_index)
                            table_freshness_batch.clear()
                            batch_index += 1
                pbar.update(1)

        # Send the remaining Health Checks to Backend if model config flag is enabled
        if model_config_utils.get_enable_put_health_checks() and table_freshness_batch:
            self._put_health_checks_batch(table_freshness_batch, batch_index)
            table_freshness_batch.clear()

        pbar.close()

        table_freshness_results = utils.sort_freshness_summary(table_freshness_results)
        return table_freshness_results

    def _test_single_table(
        self, table_name: str, event_timestamp_col_name: Optional[str] = None
    ) -> TableFreshnessInfo:
        """Test the freshness of a single table."""
        if model_config_utils.get_enable_limit_history_by_timestamp():
            history_result = history_utils.get_table_history(
                table_name,
                self._catalog_name,
                self._schema_name,
                days_limit=model_config_utils.get_max_lookback_days(),
            )
        else:
            history_result = history_utils.get_table_history(
                table_name, self._catalog_name, self._schema_name, should_limit=True
            )
        evaluation_epoch_sec = history_result.evaluation_epoch_sec
        if history_result.error_message is not None or history_result.history_df is None:
            return TableFreshnessInfo(
                table_name=table_name,
                commit_freshness_status=ResultStatus.UNKNOWN,
                event_freshness_status=ResultStatus.UNKNOWN,
                error_message=history_result.error_message,
                error_code=errors.match_error_message_to_code(history_result.error_message),
            )
        history_json_str = history_result.history_df.to_json()

        freshness_result_dict_str = utils.get_single_table_freshness_info(
            table_name=table_name,
            history_json_str=history_json_str,
            table_lineage_link=common_utils.get_lineage_link(
                f"{self._catalog_name}.{self._schema_name}.{table_name}"
            ),
            table_threshold_overrides=self._table_threshold_overrides,
            table_latency_threshold_overrides=self._table_latency_threshold_overrides,
            static_table_threshold_override=self._static_table_threshold_override,
            tables_to_skip=self._tables_to_skip,
            tables_to_scan=self._tables_to_scan,
            evaluation_epoch_sec=evaluation_epoch_sec,
            catalog_name=self._catalog_name,
            schema_name=self._schema_name,
            event_timestamp_col_name=event_timestamp_col_name,
        )

        freshness_result_dict = json.loads(freshness_result_dict_str)
        return TableFreshnessInfo.fromdict(json_dict=freshness_result_dict)

    def _put_health_checks_batch(self, table_freshness_batch: List[Dict], batch_index: int = 0):
        """
        Sends health checks to the putHealthChecks Databricks API endpoint.
        Supports one retry for each batch where we call put_health_checks.

        :param table_freshness_batch: The batch of table freshness metrics to be sent.
        :param batch_index: The index of the batch being sent.
        """

        put_health_check_fields = [
            "id",
            "result",
            "evaluated_at_timestamp",
            "last_update_timestamp",
            "staleness_age",
            "event_staleness_age",
            "error_message",
            "error_code",
        ]
        health_checks = HealthChecks(
            job_id=None,
            run_id=None,
            table_freshness_infos=[
                {field: d[field] for field in put_health_check_fields}
                for d in table_freshness_batch
            ],
        )
        self._put_health_checks_batch_helper(health_checks, batch_index)

        _logger.info("Completed Put Health checks.")

    def _put_health_checks_batch_helper(
        self, health_checks_batch: HealthChecks, batch_index: int = 0
    ):
        """
        Sends a batch of health checks to the Databricks API endpoint supporting a retry if there is a failure.

        :param health_checks_batch: The batch of health checks to be sent.
        :param batch_index: The index of the batch being sent.
        """
        try:
            common_utils.make_api_call(
                method_name="PUT",
                api_path="/api/2.1/quality-monitoring/health-checks",
                body=health_checks_batch.to_json_dict(),
            )
            _logger.info(f"Health checks batch {batch_index} sent successfully.")
        except Exception as e:
            _logger.info(f"Health checks failed to send batch {batch_index}: {e}")
