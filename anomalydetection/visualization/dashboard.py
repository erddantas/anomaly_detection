from importlib.resources import files
import json
import os
from typing import Optional
from urllib.parse import urlparse


from databricks.data_monitoring.anomalydetection.context import Context
from databricks.sdk.errors import platform as platform_errors
import jinja2

_DASHBOARD_FOLDER_NAME = "Databricks Anomaly Detection"
_DASHBOARD_NAME = "Anomaly Detection"
_DASHBOARD_FILE_NAME = _DASHBOARD_NAME + ".lvdash.json"

_DASHBOARD_QUALITY_OVERVIEW_PAGE_NAME = "quality-overview"
_DASHBOARD_TABLE_DETAILS_PAGE_NAME = "table-quality-details"
_DASHBOARD_QUALITY_OVERVIEW_LOGGING_TABLE_WIDGET_NAME = "quality-overview-logging-table-name"
_DASHBOARD_TABLE_DETAILS_LOGGING_TABLE_WIDGET_NAME = "table-quality-details-logging-table-name"


def _get_dashboard_if_exists(folder_path: str) -> Optional[str]:
    """
    Returns the anomaly detection dashboard id it exists in the specified folder,
    otherwise returns None.
    """

    workspace_client = Context.current.get_workspace_client()
    try:
        for item in workspace_client.workspace.list(folder_path):
            if os.path.basename(item.path) == _DASHBOARD_FILE_NAME:
                return item.resource_id
        return None
    # The folder may not exist yet if this is the first run.
    except platform_errors.ResourceDoesNotExist:
        return None


def _get_serialized_dashboard() -> str:
    """
    Returns the serialized dashboard JSON.  The dashboard json file should be in the
    anomalydetection.resources package and should be named "Anomaly Detection.lvdash.json".
    """
    resource_path = files("databricks.data_monitoring.anomalydetection.resources").joinpath(
        _DASHBOARD_FILE_NAME
    )
    with resource_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return json.dumps(data, ensure_ascii=False)


def get_dashboard_url(dashboard_id: str, logging_table_name: str) -> str:
    """
    Construct the dashboard URL using workspace url from spark configuration.
    Set the logging table in current run as the widget parameter.
    """
    workspace_url = Context.current.get_spark().conf.get("spark.databricks.workspaceUrl", None)
    if workspace_url and not urlparse(workspace_url).scheme:
        workspace_url = "https://" + workspace_url
        if workspace_url.endswith("/"):
            workspace_url = workspace_url[:-1]
    if not workspace_url:
        raise Exception("Could not determine workspace URL")

    # Construct the dashboard URL with the logging table name as a query parameter for the quality overview page and table details page
    logging_table_query_param = f"f_{_DASHBOARD_QUALITY_OVERVIEW_PAGE_NAME}~{_DASHBOARD_QUALITY_OVERVIEW_LOGGING_TABLE_WIDGET_NAME}={logging_table_name}&f_{_DASHBOARD_TABLE_DETAILS_PAGE_NAME}~{_DASHBOARD_TABLE_DETAILS_LOGGING_TABLE_WIDGET_NAME}={logging_table_name}"
    dashboard_url = f"{workspace_url}/sql/dashboardsv3/{dashboard_id}/pages/{_DASHBOARD_QUALITY_OVERVIEW_PAGE_NAME}?{logging_table_query_param}"
    return dashboard_url


def create_dashboard_if_not_exists() -> str:
    """
    Creates a new anomaly detection dashboard if it does not already exist and return dashboard id.
    We will create one dashboard per workspace in a shared folder accessible to all users.
    """
    try:
        workspace_client = Context.current.get_workspace_client()
        dashboard_folder_path = f"/Shared/{_DASHBOARD_FOLDER_NAME}"
        dashboard_id = _get_dashboard_if_exists(dashboard_folder_path)

        if dashboard_id is None:
            print("Existing dashboard not found, creating a new dashboard.")
            workspace_client.workspace.mkdirs(path=dashboard_folder_path)
            dashboard_id = workspace_client.lakeview.create(
                display_name=_DASHBOARD_NAME,
                parent_path=dashboard_folder_path,
                serialized_dashboard=_get_serialized_dashboard(),
            ).dashboard_id
        else:
            print("Existing dashboard found, skipping creation.")

        return dashboard_id

    except Exception as e:
        print(f"Error while creating or retrieving the dashboard: {e}")
        return None


def display_view_dashboard_button(dashboard_id: str, logging_table_name: str) -> None:
    """
    Displays a button to view the dashboard.
    """
    # Create a Jinja2 environment and load the template
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("databricks.data_monitoring.anomalydetection", "resources"),
        autoescape=jinja2.select_autoescape(["html"]),
    )
    template = env.get_template("view_dashboard.html")

    # Render the template with the data
    view_dashboard_button = template.render(
        {
            "dashboard_url": get_dashboard_url(dashboard_id, logging_table_name),
        }
    )
    Context.current.display_html(view_dashboard_button)
