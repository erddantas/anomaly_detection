from datetime import datetime, timedelta, timezone

from databricks.data_monitoring.anomalydetection.blast_radius_info import BlastRadiusInfo
from databricks.data_monitoring.anomalydetection.utils import common_utils
import databricks.data_monitoring.anomalydetection.model_config as model_config_utils


def get_downstream_tables(catalog_name: str, schema_name: str, table_name: str) -> list:
    """Fetches the downstream tables from the lineage API. Returns root table + downstream tables."""
    root_table_name = f"{catalog_name}.{schema_name}.{table_name}"

    start_timestamp_millis = int(
        (
            datetime.now() - timedelta(days=model_config_utils.get_downstream_table_lookback_days())
        ).timestamp()
        * 1000
    )

    lineage_api_method = "GET"
    lineage_api_path = "/api/2.0/lineage-tracking/securable/securables"
    request_body = {
        "securable_full_name": root_table_name,
        "securable_type": "TABLE_TYPE",
        "lineage_direction": "DOWNSTREAM",
        "securable_response_filter": "TABLE_TYPE",
        "start_timestamp": str(start_timestamp_millis),
    }

    response = common_utils.make_api_call(lineage_api_method, lineage_api_path, body=request_body)
    table_name_list = [root_table_name]

    # the lineages key may not be in the response if there's no lineage info available from the last 3 months
    if "lineages" in response:
        for lineage in response["lineages"]:
            table_info = lineage.get("tableInfo", {})
            if table_info:
                if table_info.get("has_permission") is False:
                    continue  # Skip this entry

                fq_table_name = (
                    f"{table_info['catalog_name']}.{table_info['schema_name']}.{table_info['name']}"
                )
                table_name_list.append(fq_table_name)

    return table_name_list


def get_table_popularity(table_name_list: list) -> int:
    """Fetches the total query count for a list of tables from the popularity API."""
    popularity_api_method = "GET"
    popularity_api_path = "/api/2.0/lineage-tracking/popularity/popular-tables"
    request_body = {
        "scope": "TableList",
        "table_name_list": table_name_list,
    }

    response = common_utils.make_api_call(
        popularity_api_method, popularity_api_path, body=request_body
    )
    num_queries_impacted = 0

    if "table_popularity_list" in response:
        for table_info in response["table_popularity_list"]:
            query_count = table_info.get("query_count", 0)
            num_queries_impacted += query_count

    return num_queries_impacted


def determine_impact_level(num_queries_impacted: int) -> int:
    """Determines the impact level based on the number of queries impacted."""
    if num_queries_impacted <= 30:
        return 1
    elif num_queries_impacted <= 300:
        return 2
    elif num_queries_impacted <= 3000:
        return 3
    else:
        return 4


def get_blast_radius(catalog_name: str, schema_name: str, table_name: str) -> BlastRadiusInfo:
    """Calculates the blast radius of an anomaly affecting a table."""
    table_name_list = get_downstream_tables(catalog_name, schema_name, table_name)
    num_queries_impacted = get_table_popularity(table_name_list)
    impact_level = determine_impact_level(num_queries_impacted)

    return BlastRadiusInfo(
        impact_level=impact_level,
        num_downstream_tables=len(table_name_list) - 1,
        num_queries_on_affected_tables=num_queries_impacted,
    )
