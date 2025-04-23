"""
This module provides a base checker class that contains common functionality for different
types of data monitoring checks (e.g., freshness, completeness)
"""

import abc
from typing import List, Optional

from databricks.data_monitoring.anomalydetection.context import Context
from databricks.data_monitoring.anomalydetection.databricks_context import (
    DatabricksContext,
)
from databricks.data_monitoring.anomalydetection.utils import (
    logging_table_utils,
    common_utils,
)


class BaseChecker(abc.ABC):
    """
    Base class for different types of data monitoring checkers.

    Provides common functionality for setting up logging tables, running checks,
    and processing results.
    """

    def __init__(
        self,
        catalog_name: str,
        schema_name: str,
        tables_to_skip: Optional[List[str]] = None,
        tables_to_scan: Optional[List[str]] = None,
        logging_table_name: Optional[str] = None,
    ):
        """
        Initialize the base checker.

        :param catalog_name: Name of the catalog to check
        :param schema_name: Name of the schema to check
        :param tables_to_skip: Tables to ignore in detecting anomalies
        :param tables_to_scan: Tables to scan for anomalies
        :param logging_table_name: Name of the table for logging results.
            Defaults to _quality_monitoring_summary in the same schema.
        """
        self._catalog_name = catalog_name
        self._schema_name = schema_name
        self._num_tables_in_schema: Optional[int] = None

        if tables_to_skip and tables_to_scan:
            raise ValueError("tables_to_skip and tables_to_scan cannot be set at the same time.")

        self._tables_to_skip = tables_to_skip or []
        self._tables_to_scan = tables_to_scan or []
        self._schema_full_name = f"{self._catalog_name}.{self._schema_name}"

        self.set_logging_table(logging_table_name)

        # determine the earliest eval time in run
        self._min_run_eval_time = None

        if not Context.active:
            Context.current = DatabricksContext()

        common_utils.check_feature_enabled()

    def set_logging_table(self, logging_table_name: Optional[str]) -> None:
        """
        Sets the logging table name.

        Args:
            logging_table_name: Name of the table to log results
        """
        self._logging_table_name = logging_table_utils.get_logging_table_name(logging_table_name)
        self._logging_table_full_name = logging_table_utils.autocomplete_table_name(
            self._catalog_name, self._schema_name, self._logging_table_name
        )

    @abc.abstractmethod
    def run_checks(self) -> None:
        """
        Runs checks on tables in the schema and processes the results.
        """
        pass

    @abc.abstractmethod
    def run_backtesting(self) -> None:
        """
        Runs backtesting on tables in the schema and processes the results.
        """
        pass
