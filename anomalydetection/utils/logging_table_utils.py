"""
This module provides util methods for logging table information
"""

from pyspark.sql import types as T
from typing import Optional


DEFAULT_LOGGING_TABLE_NAME = "_quality_monitoring_summary"

DOWNSTREAM_IMPACT_COLUMN_NAME = "downstream_impact"
LOGGING_TABLE_FIELDS = [
    T.StructField("evaluated_at", T.TimestampType()),
    # The catalog name
    T.StructField("catalog", T.StringType()),
    # The schema name
    T.StructField("schema", T.StringType()),
    # The table name
    T.StructField("table_name", T.StringType()),
    # e.g. Freshness or Completeness
    T.StructField("quality_check_type", T.StringType()),
    # e.g. Healthy, Unhealthy, Unknown
    T.StructField("status", T.StringType()),
    # Downstream impact analysis
    T.StructField(
        DOWNSTREAM_IMPACT_COLUMN_NAME,
        T.StructType(
            [
                T.StructField("impact_level", T.IntegerType()),
                T.StructField("num_downstream_tables", T.IntegerType()),
                T.StructField("num_queries_on_affected_tables", T.IntegerType()),
            ]
        ),
    ),
    # Array of debug info objects with variable schema
    T.StructField(
        "additional_debug_info",
        T.MapType(T.StringType(), T.MapType(T.StringType(), T.StringType())),
    ),
    T.StructField("error_message", T.StringType()),
    T.StructField("table_lineage_link", T.StringType()),
]

LOGGING_TABLE_SCHEMA = T.StructType(LOGGING_TABLE_FIELDS)


def get_logging_table_name(logging_table_name: Optional[str]) -> str:
    return logging_table_name or DEFAULT_LOGGING_TABLE_NAME


def get_logging_table_full_name(
    catalog: str, schema: str, logging_table_name: Optional[str]
) -> str:
    table_name = get_logging_table_name(logging_table_name)
    return autocomplete_table_name(catalog, schema, table_name)


def autocomplete_table_name(input_catalog: str, input_schema: str, table_name: str) -> str:
    """
    Parses the table name to return the catalog, schema, and table name, and booleans to indicate whether catalog
        and schema were autocompleted
    :param input_catalog: The catalog name
    :param input_schema: The schema name
    :param table_name: The name of the table. Qualified names:
                        - Can be qualified with catalog and schema like "{catalog}.{schema}.{table}". or
                        - Can be qualified with schema, like "{schema}.{table}", or
                        - Can be qualified with just table name, like "{table}".

    :return: the autocompleted table name
    """
    # remove backticks in logging table name since we autopopulate them again in autocomplete
    table_name = table_name.replace("`", "")
    parts = table_name.split(".")
    if len(parts) > 3:
        raise Exception(f"Invalid logging table name {table_name}")

    if len(parts) == 1:
        catalog = input_catalog
        schema = input_schema
        name = parts[0]
    elif len(parts) == 2:
        catalog = input_catalog
        schema, name = parts
    else:  # len(parts) == 3
        catalog, schema, name = parts

    return f"`{catalog}`.`{schema}`.`{name}`"
