"""
Data classes for evaluation metrics of anomaly detection.
"""

import dataclasses
from typing import Dict, Optional


@dataclasses.dataclass
class AnomalyDetectionMetrics:
    job_id: Optional[int] = None
    run_id: Optional[int] = None
    catalog_name: Optional[str] = None
    schema_name: Optional[str] = None
    # Threshold overrides for commit-based freshness
    threshold_overrides: dict[str, int] = dataclasses.field(default_factory=dict)
    # Threshold overrides for event-based freshness
    event_based_threshold_overrides: dict[str, int] = dataclasses.field(default_factory=dict)
    # Static table threshold override for commit-based freshness
    static_table_threshold_override: Optional[float] = None
    # Threshold overrides for completeness
    completeness_threshold_overrides: dict[str, int] = dataclasses.field(default_factory=dict)
    # Disabled tables for freshness
    disabled_tables: list[str] = dataclasses.field(default_factory=list)
    # Disabled tables for completeness
    completeness_disabled_tables: list[str] = dataclasses.field(default_factory=list)
    # Dicts from TableFreshnessInfo.to_metric_dict()
    table_freshness_infos: list[Dict] = dataclasses.field(default_factory=list)
    # Dicts from TableCompletenessInfo.to_metric_dict()
    table_completeness_infos: list[Dict] = dataclasses.field(default_factory=list)
    num_tables_in_schema: Optional[int] = None
    has_notifications_enabled: Optional[bool] = None

    def to_json_dict(self):
        return {"anomaly_detection_metrics": dataclasses.asdict(self)}


@dataclasses.dataclass
class HealthChecks:
    job_id: Optional[int] = None
    run_id: Optional[int] = None
    table_freshness_infos: list[Dict] = dataclasses.field(default_factory=list)

    def to_json_dict(self):
        return dataclasses.asdict(self)
