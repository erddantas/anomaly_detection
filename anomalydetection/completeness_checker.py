"""
This module provides the client facing entrypoint into checking for completeness anomalies
"""

import concurrent.futures
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import asdict

import pandas as pd
from tqdm import tqdm

from databricks.data_monitoring.anomalydetection.base_checker import BaseChecker
from databricks.data_monitoring.anomalydetection.context import Context
from databricks.data_monitoring.anomalydetection.completeness_info import (
    TableCompletenessInfo,
    CompletenessResult,
    COMPLETENESS_STATUS_ORDER,
)
from databricks.data_monitoring.anomalydetection import (
    completeness_utils,
    model_config as model_config_utils,
)
from databricks.data_monitoring.anomalydetection.utils import (
    table_history_utils as history_utils,
    common_utils,
)
from databricks.data_monitoring.anomalydetection import errors
from databricks.data_monitoring.anomalydetection.utils.logging_table_utils import (
    LOGGING_TABLE_SCHEMA,
    DOWNSTREAM_IMPACT_COLUMN_NAME,
)
from databricks.data_monitoring.anomalydetection.metrics import AnomalyDetectionMetrics


class CompletenessChecker(BaseChecker):
    """
    Checker that in parallel checks for completeness anomalies for all tables in a schema

        :param catalog_name: Name of the catalog to check completeness
        :param schema_name: Name of the schema to check completeness
        :param tables_to_skip: (Optional) Tables to ignore in detecting completeness
            anomalies
        :param tables_to_scan: (Optional) Tables to scan for completeness anomalies
        :param table_threshold_overrides: (Optional) Dictionary of table names to
            completeness thresholds
        :param logging_table_name: (deprecated) Name of the table to log completeness.
            Defaults to _quality_monitoring_summary in the same catalog / schema to check completeness on
    """

    def __init__(
        self,
        catalog_name: str,
        schema_name: str,
        tables_to_skip: Optional[List[str]] = None,
        tables_to_scan: Optional[List[str]] = None,
        table_threshold_overrides: Optional[Dict[str, int]] = None,
        logging_table_name: Optional[str] = None,
    ):
        super().__init__(
            catalog_name=catalog_name,
            schema_name=schema_name,
            tables_to_skip=tables_to_skip,
            tables_to_scan=tables_to_scan,
            logging_table_name=logging_table_name,
        )
        self._num_tables_in_schema: Optional[int] = None
        self._table_threshold_overrides = table_threshold_overrides or {}

    def run_checks(self):
        self.run_completeness_checks()

    def run_completeness_checks(self) -> None:
        completeness_summary = self._check_completeness()
        results_df = self._display_results(completeness_summary)
        # Log results to logging table
        self._min_run_eval_time = results_df["evaluated_at"].min()
        common_utils.log_results(results_df, self._logging_table_full_name, LOGGING_TABLE_SCHEMA)
        metrics = self._build_anomaly_metrics(completeness_summary)
        common_utils.send_anomaly_detection_metrics(metrics)

        num_unhealthy_tables = sum(
            [
                table_info.completeness_status == CompletenessResult.UNHEALTHY
                for _, table_info in completeness_summary.items()
            ]
        )
        num_total_tables = len(completeness_summary)
        print(
            f"Found {num_unhealthy_tables} unhealthy tables out of the {num_total_tables} "
            f"tables in schema {self._catalog_name}.{self._schema_name}."
        )

    def _test_single_table(self, table_name: str) -> TableCompletenessInfo:
        evaluated_at = datetime.now()
        try:
            if model_config_utils.get_enable_limit_history_by_timestamp():
                table_history = history_utils.get_table_history(
                    table_name,
                    self._catalog_name,
                    self._schema_name,
                    days_limit=model_config_utils.get_max_lookback_days(),
                )
            else:
                table_history = history_utils.get_table_history(
                    table_name, self._catalog_name, self._schema_name, should_limit=False
                )
            if table_history.history_df is None:
                raise ValueError(
                    table_history.error_message
                    or errors.ERROR_CODE_TO_MESSAGE[errors.ErrorCode.NOT_ENOUGH_TABLE_HISTORY]
                )

            return completeness_utils.get_single_table_completeness_info(
                self._catalog_name,
                self._schema_name,
                table_name,
                table_history.history_df,
                evaluated_at,
                self._table_threshold_overrides,
            )

        except Exception as e:
            error_message = str(e)
            error_code = errors.match_error_message_to_code(error_message)
            return TableCompletenessInfo(
                table_name=table_name,
                completeness_status=CompletenessResult.UNKNOWN,
                evaluated_at=evaluated_at,
                table_lineage_link=common_utils.get_lineage_link(
                    f"{self._catalog_name}.{self._schema_name}.{table_name}"
                ),
                error_message=error_message,
                error_code=error_code,
            )

    def _check_completeness(self) -> Dict[str, TableCompletenessInfo]:
        tables_to_eval, _ = common_utils.get_tables_to_eval(
            self._catalog_name,
            self._schema_name,
            True,
            self._tables_to_scan,
            self._tables_to_skip,
            self._logging_table_full_name,
        )
        self._num_tables_in_schema = len(tables_to_eval)

        table_completeness_results = {}
        table_name_to_id_map = {table.name: table.table_id for table in tables_to_eval}

        # Set up progress bar with total number of tables
        pbar = tqdm(total=len(tables_to_eval), desc="Checking table completeness")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_table = {
                executor.submit(
                    self._test_single_table,
                    table.name,
                ): table.name
                for table in tables_to_eval
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_table):
                completeness_info = future.result()
                completeness_info.table_id = table_name_to_id_map[completeness_info.table_name]
                table_completeness_results[completeness_info.table_name] = completeness_info
                pbar.update(1)
        pbar.close()

        return table_completeness_results

    def _build_anomaly_metrics(
        self, completeness_infos: Dict[str, TableCompletenessInfo]
    ) -> AnomalyDetectionMetrics:
        metrics = AnomalyDetectionMetrics(
            # TODO populate job id and run id
            catalog_name=self._catalog_name,
            schema_name=self._schema_name,
            completeness_threshold_overrides=self._table_threshold_overrides,
            completeness_disabled_tables=self._tables_to_skip,
            table_completeness_infos=[
                info.to_metric_dict() for info in completeness_infos.values()
            ],
            num_tables_in_schema=self._num_tables_in_schema,
        )
        return metrics

    def _display_results(
        self, completeness_summary: Dict[str, TableCompletenessInfo]
    ) -> pd.DataFrame:
        # Convert the completeness summary to a DataFrame, and only display desired columns
        df = self._to_logging_dataframe(completeness_summary)

        # Map completeness_status to sort order and sort
        df["status_order"] = df["status"].map(
            lambda status: COMPLETENESS_STATUS_ORDER.index(status)
        )
        df = df.sort_values(by=["status_order", "table_name"]).drop(columns="status_order")
        return df

    def _to_logging_dataframe(
        self, completeness_summary: Dict[str, TableCompletenessInfo]
    ) -> pd.DataFrame:
        """
        Convert the completeness summary to a DataFrame with the specified schema for logging.

        Args:
            completeness_summary: Dictionary mapping table names to TableCompletenessInfo objects.

        Returns:
            pd.DataFrame: DataFrame with the logging schema structure.
        """
        rows = []

        for table_name, info in completeness_summary.items():
            # Always add debug info with the error code, even if row volumes are None
            additional_debug_info = {}
            actual_value = (
                str(info.last_window_row_volume)
                if info.last_window_row_volume is not None
                else None
            )
            expectation = (
                f"actual_value > {info.predicted_row_volume_lower_bound}"
                if info.predicted_row_volume_lower_bound is not None
                else None
            )
            is_violated = "Unknown"
            if info.completeness_status == CompletenessResult.HEALTHY:
                is_violated = "False"
            elif info.completeness_status == CompletenessResult.UNHEALTHY:
                is_violated = "True"
            debug_info = {
                "actual_value": actual_value,
                "expectation": expectation,
                "is_violated": is_violated,
                "error_code": str(info.error_code.value) if info.error_code else "None",
                "expectation_threshold": str(info.predicted_row_volume_lower_bound),
            }
            additional_debug_info["daily_row_count"] = debug_info

            row = {
                "evaluated_at": info.evaluated_at,
                "catalog": self._catalog_name,
                "schema": self._schema_name,
                "table_name": table_name,
                "quality_check_type": "Completeness",
                "status": info.completeness_status.value,
                DOWNSTREAM_IMPACT_COLUMN_NAME: asdict(info.blast_radius)
                if info.blast_radius
                else None,
                "additional_debug_info": additional_debug_info,
                "error_message": info.error_message if info.error_message else None,
                "table_lineage_link": info.table_lineage_link
                if info.table_lineage_link != (None,)
                else None,
            }

            rows.append(row)

        return pd.DataFrame(rows)

    def run_backtesting(self) -> None:
        """
        Runs backtesting on tables in the schema and processes the results.
        """
        raise ValueError("Backtesting is not supported for completeness checks.")
