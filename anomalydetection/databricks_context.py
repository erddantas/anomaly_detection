"""
This module provides the Databricks version of Context and a decorator to set it for API calls
"""

from pyspark.sql import SparkSession

from databricks.data_monitoring.anomalydetection.context import Context, ContextMeta
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config


class DatabricksContext(Context, metaclass=ContextMeta):
    """
    Anomaly detection context for Databricks execution.

    NOTE: This class is not covered by unit tests and is meant to be tested through
    DUST suites that run this code on an actual Databricks cluster.
    """

    def __init__(self, spark=None):
        if spark:
            self._spark = spark
        else:
            self._spark = SparkSession.builder.appName("databricks.data_monitoring").getOrCreate()

        self._workspace_client = WorkspaceClient(
            config=Config(
                http_timeout_seconds=10,
            )
        )
        self._dbutils = self._get_dbutils()

    def get_spark(self) -> SparkSession:
        return self._spark

    def get_workspace_client(self) -> WorkspaceClient:
        return self._workspace_client

    def get_dbutils(self):
        return self._dbutils

    def display_html(self, html: str) -> None:
        # pylint: disable=protected-access
        self._dbutils.notebook.displayHTML(html)

    @classmethod
    def _get_dbutils(cls):
        """
        Returns an instance of dbutils.
        """
        try:
            from databricks.sdk.runtime import dbutils

            return dbutils
        except ImportError:
            import IPython

            dbutils = IPython.get_ipython().user_ns["dbutils"]
        return dbutils
