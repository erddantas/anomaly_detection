"""
AnomalyDetector class that enables running various metric checks.
"""

from typing import Optional

from databricks.data_monitoring.anomalydetection.freshness_checker import FreshnessChecker
from databricks.data_monitoring.anomalydetection.completeness_checker import CompletenessChecker
from databricks.data_monitoring.anomalydetection.context import Context
import databricks.data_monitoring.anomalydetection.model_config as model_config_utils
from databricks.data_monitoring.anomalydetection.utils import common_utils
from databricks.data_monitoring.anomalydetection.context import Context
import pyspark.sql.functions as F
from pyspark.sql import DataFrame


class AnomalyDetector:
    """Manages multiple MetricCheckers with optional configurations."""

    def __init__(
        self,
        logging_table_full_name: str,
        freshness_checker: Optional[FreshnessChecker] = None,
        completeness_checker: Optional[CompletenessChecker] = None,
    ):
        """
        :param logging_table_full_name: Full name of the table for logging results.
        :param freshness_checker: Instance of FreshnessChecker.
        :param completeness_checker: Instance of CompletenessChecker.
        """
        self.logging_table_full_name = logging_table_full_name
        self.freshness_checker = freshness_checker
        self.completeness_checker = completeness_checker
        self._min_run_eval_time = None
        self._enable_backtesting = False

    def run_checks(self) -> None:
        # check if backtesting should be run
        self._enable_backtesting = self._should_run_backtesting()

        error_messages = []

        if self.freshness_checker:
            try:
                self.freshness_checker.run_checks()
            except Exception as e:
                error_messages.append(f"Freshness check failed unexpectedly: {str(e)}")

        if self.completeness_checker:
            try:
                self.completeness_checker.run_checks()
            except Exception as e:
                error_messages.append(f"Completeness check failed unexpectedly: {str(e)}")

        if error_messages:
            formatted_errors = "\n".join(error_messages)
            num_failed = len(error_messages)
            raise Exception(f"{num_failed} check(s) failed to evaluate:\n{formatted_errors}")

        # run backtesting
        self._run_backtesting()

        # get min run eval time from all checks and logging table
        self._update_run_eval_time()
        logging_table_sdf = self._current_run_logging_table()

        return logging_table_sdf

    def _current_run_logging_table(self) -> Optional[DataFrame]:
        """Outputs the current run's logging table with the enabled checks."""
        if self._min_run_eval_time:
            logging_table_sdf = Context.spark.table(self.logging_table_full_name)
            filtered_run_sdf = logging_table_sdf.filter(
                F.col("evaluated_at") >= self._min_run_eval_time
            )
            # sort the logging table
            sorted_sdf = common_utils.sort_current_run_logging_table(
                current_run_sdf=filtered_run_sdf,
            )
            return sorted_sdf
        return None

    def _update_run_eval_time(self):
        """Updates the _min_run_eval_time after checks run."""
        # Collect eval times from executed checkers
        eval_times = [
            checker._min_run_eval_time
            for checker in [self.freshness_checker, self.completeness_checker]
            if checker and checker._min_run_eval_time is not None
        ]

        # Set the earliest evaluation time
        self._min_run_eval_time = min(eval_times) if eval_times else None

    def _run_backtesting(self) -> None:
        """
        Runs backtesting for each checker on all tables.
        :return: None
        """
        should_run_backtesting = self._enable_backtesting
        if not should_run_backtesting:
            return

        print(f"Running backtesting and logging results to {self.logging_table_full_name}")
        error_messages = []
        if self.freshness_checker:
            try:
                self.freshness_checker.run_backtesting()
            except Exception as e:
                error_messages.append(f"Freshness check failed unexpectedly: {str(e)}")

        if error_messages:
            formatted_errors = "\n".join(error_messages)
            num_failed = len(error_messages)
            raise Exception(
                f"{num_failed} check(s) failed to exucute backtesting:\n{formatted_errors}"
            )

    def _should_run_backtesting(self) -> bool:
        """
        Determines if backtesting should be run based on model config flag and existence of logging table.
        :return: True if backtesting should be run, False otherwise.
        """
        # safe guard backtesting with a model config
        if model_config_utils.get_enable_backtesting():
            # check existence of logging table and entries
            if not Context.spark.catalog.tableExists(self.logging_table_full_name):
                return True
            else:
                logging_df = Context.spark.table(self.logging_table_full_name)
                if logging_df.count() == 0:
                    return True
                return False
        return False

    def _log_output(self, results):
        """Logs output to unified schema."""
        # Implement delta table logging logic here
        pass

    def _send_metrics(self, results):
        """Sends metrics from all metric checkers."""
        # Implement sendAnomalyDetectionMetrics call
        pass
