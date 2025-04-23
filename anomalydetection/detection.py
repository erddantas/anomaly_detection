from typing import Optional, Iterable
import traceback
import json

from databricks.data_monitoring.anomalydetection.metric_config import (
    FreshnessConfig,
    CompletenessConfig,
    MetricConfig,
)
from databricks.data_monitoring.anomalydetection.freshness_checker import FreshnessChecker
from databricks.data_monitoring.anomalydetection.completeness_checker import CompletenessChecker
from databricks.data_monitoring.anomalydetection.anomaly_detector import AnomalyDetector
from databricks.data_monitoring.anomalydetection.context import Context
from databricks.data_monitoring.anomalydetection.databricks_context import (
    DatabricksContext,
)
from databricks.data_monitoring.anomalydetection.utils.logging_table_utils import (
    DEFAULT_LOGGING_TABLE_NAME,
    autocomplete_table_name,
)
from databricks.data_monitoring.anomalydetection.utils import common_utils
import databricks.data_monitoring.anomalydetection.model_config as model_config_utils
from databricks.data_monitoring.anomalydetection.visualization import dashboard as dashboard_utils
from databricks.sdk.service import jobs
from pyspark.sql import DataFrame


def run_anomaly_detection(
    catalog_name: str,
    schema_name: str,
    metric_configs: Iterable[MetricConfig] = (),
    logging_table_name: Optional[str] = DEFAULT_LOGGING_TABLE_NAME,
) -> Optional[DataFrame]:
    """
    Initializes and runs anomaly detection based on the provided configurations.

    :param catalog_name: Catalog to run anomaly detection on
    :param schema_name: Schema to run anomaly detection on
    :param metric_configs: A list of MetricConfig objects (e.g., FreshnessConfig, CompletenessConfig).
    :param logging_table_name: (Optional) Name of the table for logging results.
        Defaults to _quality_monitoring_summary in the same schema.
    """
    if not Context.active:
        Context.current = DatabricksContext()

    # Default checkers with no overrides
    freshness_checker, completeness_checker, logging_table_full_name = setup_anomaly_detection_run(
        catalog_name=catalog_name,
        schema_name=schema_name,
        metric_configs=metric_configs,
        logging_table_name=logging_table_name,
    )

    ad = AnomalyDetector(
        logging_table_full_name=logging_table_full_name,
        freshness_checker=freshness_checker,
        completeness_checker=completeness_checker,
    )
    return ad.run_checks()


def create_anomaly_detection_job(
    catalog_name: str,
    schema_name: str,
    metric_configs: Iterable[MetricConfig] = (),
    email_notifications: Iterable[str] = (),
    interval_hours: Optional[int] = None,
    logging_table_name: Optional[str] = DEFAULT_LOGGING_TABLE_NAME,
    whl_override: Optional[str] = None,
):
    """
    Creates and triggers a Databricks notebook job from a Git provider to perform a freshness check.
    Source of truth for wheel version is GitHub notebook.
    :param catalog_name: Catalog to run anomaly detection on
    :param schema_name: Schema to run anomaly detection on
    :param metric_configs: A list of MetricConfig objects (e.g., FreshnessConfig, CompletenessConfig).
    :param email_notifications: List of email addresses to notify
    :param interval_hours: (Optional) Schedule frequency in hour. 6 hours by default.
    :param logging_table_name: (Optional) Name of the table for logging results.
        Defaults to _quality_monitoring_summary in the same schema.
    :param whl_override: (Internal only) whl url override. This is the whl version that freshness check running on
    """
    if not Context.active:
        Context.current = DatabricksContext()

    # validate configs, and create dashboard
    setup_anomaly_detection_run(
        catalog_name=catalog_name,
        schema_name=schema_name,
        metric_configs=metric_configs,
        logging_table_name=logging_table_name,
    )

    schema_full_name = f"{catalog_name}.{schema_name}"

    # Convert configs to json compatible dicts
    configs_dict_list = []
    for config in metric_configs:
        config_dict = config.to_dict()
        # Add a marker to help us know which class to use during decoding.
        config_dict["metric_type"] = type(config).__name__
        configs_dict_list.append(config_dict)

    DEFAULT_INTERVAL_HOURS = 6
    w = Context.current.get_workspace_client()
    workspace_url = common_utils.get_workspace_url()
    try:
        created_job = w.jobs.create(
            name=f"[Quality Anomaly Detection] on schema {schema_full_name}",
            tasks=[
                jobs.Task(
                    description=f"Job to run quality anomaly detection on schema {schema_full_name}",
                    notebook_task=jobs.NotebookTask(
                        notebook_path="anomaly_detection",
                        base_parameters={
                            "catalog_name": catalog_name,
                            "schema_name": schema_name,
                            "metric_configs": json.dumps(configs_dict_list),
                            "logging_table_name": logging_table_name or "",
                            # use notebook default wheel if no whl_override is provided
                            "whl_override": whl_override or "",
                        },
                        source=jobs.Source.GIT,
                    ),
                    task_key="anomaly_detection",
                    timeout_seconds=0,
                    max_retries=0,  # explicitly sets max retries to 0 (default value) for job
                )
            ],
            email_notifications=jobs.JobEmailNotifications(on_failure=email_notifications),
            trigger=jobs.TriggerSettings(
                pause_status=jobs.PauseStatus.UNPAUSED,
                periodic=jobs.PeriodicTriggerConfiguration(
                    interval=interval_hours or DEFAULT_INTERVAL_HOURS,
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


def setup_anomaly_detection_run(
    catalog_name: str,
    schema_name: str,
    metric_configs: Iterable[MetricConfig],
    logging_table_name: str,
) -> (FreshnessChecker, CompletenessChecker, str):
    """
    Parses the metric configs and initializes the checkers accordingly.
    Validates that at least one metric is enabled.
    Also checks for the existence of the dashboard, and if not, creates it.

    :param catalog_name: Catalog to run anomaly detection on
    :param schema_name: Schema to run anomaly detection on
    :param metric_configs: A list of MetricConfig objects (e.g., FreshnessConfig, CompletenessConfig).
    :return:
    """
    common_utils.check_feature_enabled()

    if not catalog_name or not schema_name:
        raise AssertionError("Valid catalog and schema required")

    # Default checkers with logging table and no overrides
    if logging_table_name is None:
        logging_table_name = DEFAULT_LOGGING_TABLE_NAME
    logging_table_full_name = autocomplete_table_name(catalog_name, schema_name, logging_table_name)
    freshness_checker = FreshnessChecker(
        catalog_name=catalog_name,
        schema_name=schema_name,
        logging_table_name=logging_table_full_name,
    )
    completeness_checker = CompletenessChecker(
        catalog_name=catalog_name,
        schema_name=schema_name,
        logging_table_name=logging_table_full_name,
    )

    for config in metric_configs:
        filtered_config = {
            key: value for key, value in config.__dict__.items() if key not in {"disable_check"}
        }
        # override with full name
        filtered_config["logging_table_name"] = logging_table_full_name
        if isinstance(config, FreshnessConfig):
            if config.disable_check:
                freshness_checker = None
            else:
                freshness_checker = FreshnessChecker(
                    catalog_name=catalog_name,
                    schema_name=schema_name,
                    **filtered_config,
                )
        elif isinstance(config, CompletenessConfig):
            if config.disable_check:
                completeness_checker = None
            else:
                completeness_checker = CompletenessChecker(
                    catalog_name=catalog_name,
                    schema_name=schema_name,
                    **filtered_config,
                )

    enabled_check_names = [
        checker.__class__.__name__.replace("Checker", "")
        for checker in [freshness_checker, completeness_checker]
        if checker
    ]

    if not enabled_check_names:
        raise AssertionError("At least one metric must be enabled.")

    # Dashboard creation
    if model_config_utils.get_enable_dashboard():
        dash_id = dashboard_utils.create_dashboard_if_not_exists()
        dashboard_utils.display_view_dashboard_button(dash_id, logging_table_full_name)

    print("Quality metrics: " + ", ".join(enabled_check_names))
    print("Logging table: " + logging_table_full_name)

    return freshness_checker, completeness_checker, logging_table_full_name
