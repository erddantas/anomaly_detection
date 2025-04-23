"""
Data classes representing the output of the completeness checker.
"""

import dataclasses
from datetime import datetime
from enum import Enum
from pyspark.sql import types as T
from typing import Optional, Dict
from databricks.data_monitoring.anomalydetection.errors import ErrorCode
from databricks.data_monitoring.anomalydetection.blast_radius_info import BlastRadiusInfo


class CompletenessResult(Enum):
    HEALTHY = "Healthy"
    UNHEALTHY = "Unhealthy"
    UNKNOWN = "Unknown"


COMPLETENESS_STATUS_ORDER = [
    CompletenessResult.UNHEALTHY.value,
    CompletenessResult.HEALTHY.value,
    CompletenessResult.UNKNOWN.value,
]


@dataclasses.dataclass
class TableCompletenessInfo:
    table_name: Optional[str] = None
    table_id: Optional[str] = None  # only used for internal logging
    completeness_status: Optional[CompletenessResult] = None
    last_window_row_volume: Optional[int] = None
    predicted_row_volume_lower_bound: Optional[int] = None
    evaluated_at: Optional[datetime] = None
    error_message: Optional[str] = None
    error_code: Optional[ErrorCode] = None
    table_lineage_link: Optional[str] = (None,)
    # fields below not shown to users
    confidence_level: Optional[float] = None
    model_config: Optional[Dict] = (None,)
    model_hyperparameters: Optional[Dict] = None
    blast_radius: Optional[BlastRadiusInfo] = None

    # Convert this to a dictionary with the relevant metric fields for service/lumberjack logging
    def to_metric_dict(self):
        return {
            "name": self.table_name,
            "id": self.table_id,
            "result": str(self.completeness_status.value) if self.completeness_status else None,
            "actual_row_volume": None
            if self.last_window_row_volume is None
            else int(self.last_window_row_volume),
            "predicted_row_volume": None
            if self.predicted_row_volume_lower_bound is None
            else int(self.predicted_row_volume_lower_bound),
            "evaluated_at_timestamp": int(self.evaluated_at.timestamp())
            if self.evaluated_at
            else None,
            "error_message": self.error_message,
            "error_code": self.error_code.value if self.error_code else None,
        }
