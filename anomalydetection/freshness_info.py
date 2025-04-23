"""
Data classes representing the output of the freshness checker.
"""

import dataclasses
from datetime import datetime
from enum import Enum
from pyspark.sql import types as T
from typing import Dict, Optional
from databricks.data_monitoring.anomalydetection.errors import ErrorCode
from databricks.data_monitoring.anomalydetection.blast_radius_info import BlastRadiusInfo


# order for priority when displaying freshness status
# Note: these values should align with `toHealthCheckStatusEnumFromProto` in the
# server `HealthCheckRecord` class.
class ResultStatus(Enum):
    FRESH = "Fresh"
    STALE = "Stale"
    UNKNOWN = "Unknown"
    SKIPPED = "Skipped"


RESULT_STATUS_ORDER = [
    ResultStatus.STALE.value,
    ResultStatus.FRESH.value,
    ResultStatus.UNKNOWN.value,
    ResultStatus.SKIPPED.value,
]


@dataclasses.dataclass
class TableFreshnessInfo:
    table_name: Optional[str] = None
    table_id: Optional[str] = None  # only used for internal logging
    commit_freshness_status: Optional[ResultStatus] = None
    last_data_update: Optional[datetime] = None
    staleness_age_seconds: Optional[int] = None
    predicted_next_data_update: Optional[datetime] = None
    predicted_upper_bound_next_data_update: Optional[datetime] = None
    evaluated_at: Optional[datetime] = None
    error_message: Optional[str] = None
    error_code: Optional[ErrorCode] = None  # used for internal logging of errors
    confidence_level: Optional[float] = None
    table_lineage_link: Optional[str] = None
    commit_timestamps: list[int] = dataclasses.field(
        default_factory=list
    )  # only used for internal logging
    commit_model_config: Optional[Dict] = None
    commit_model_hyperparameters: Optional[Dict] = None
    event_model_config: Optional[Dict] = None
    event_model_hyperparameters: Optional[Dict] = None
    # Event freshness fields
    event_freshness_status: Optional[ResultStatus] = None
    last_event: Optional[datetime] = None
    event_staleness_age_seconds: Optional[int] = None
    event_predicted_latency_seconds: Optional[int] = None
    event_predicted_upper_bound_latency_seconds: Optional[int] = None
    overall_freshness_status: Optional[ResultStatus] = None
    blast_radius: Optional[BlastRadiusInfo] = None

    def __setattr__(self, key, value):
        # Intercept attribute updates for evaluated_at and last_data_update
        super().__setattr__(key, value)
        if key in {
            "evaluated_at",
            "last_data_update",
            "commit_freshness_status",
        }:
            self._update_staleness_age()

        if key in {
            "evaluated_at",
            "last_event",
            "event_freshness_status",
        }:
            self._update_event_staleness_age()

        if key in {"commit_freshness_status", "event_freshness_status"}:
            self._update_overall_freshness_status()

    def __post_init__(self):
        # Explicitly call __setattr__ for each field to trigger custom behavior so overall freshness status is updated
        self.__setattr__("commit_freshness_status", self.commit_freshness_status)
        self.__setattr__("event_freshness_status", self.event_freshness_status)

    def is_static(self) -> bool:
        return (
            self.overall_freshness_status == ResultStatus.FRESH
            and self.predicted_upper_bound_next_data_update is None
        )

    def _update_staleness_age(self):
        if self.evaluated_at and self.last_data_update:
            staleness_age_seconds = (self.evaluated_at - self.last_data_update).total_seconds()
            self.staleness_age_seconds = int(staleness_age_seconds)

    def _update_event_staleness_age(self):
        if self.evaluated_at and self.last_event:
            event_staleness_age_seconds = (self.evaluated_at - self.last_event).total_seconds()
            self.event_staleness_age_seconds = int(event_staleness_age_seconds)

    def _update_overall_freshness_status(self):
        priority_statuses = [
            ResultStatus.STALE,
            ResultStatus.UNKNOWN,
            ResultStatus.SKIPPED,
        ]

        for status in priority_statuses:
            if self.commit_freshness_status == status or self.event_freshness_status == status:
                self.overall_freshness_status = status
                return
        self.overall_freshness_status = ResultStatus.FRESH

    def asdict(self):
        # dataclasses.asdict returns {} in udfs, so manually coding the dict conversion.
        # Only returns the fields that are surfaced to users in the final table,
        # so fields such as table_id and commit_timestamps are not returned here.
        return {
            "table_name": self.table_name,
            "overall_freshness_status": self.overall_freshness_status.value
            if self.overall_freshness_status
            else None,
            "commit_freshness_status": self.commit_freshness_status.value
            if self.commit_freshness_status
            else None,
            "staleness_age_seconds": self.staleness_age_seconds,
            "evaluated_at": self.evaluated_at,
            "last_data_update": self.last_data_update,
            "predicted_next_data_update": self.predicted_next_data_update,
            "predicted_upper_bound_next_data_update": self.predicted_upper_bound_next_data_update,
            "confidence_level": self.confidence_level,
            "error_message": self.error_message,
            "error_code": self.error_code.value if self.error_code else None,
            "table_lineage_link": self.table_lineage_link,
            "commit_model_config": self.commit_model_config,
            "commit_model_hyperparameters": self.commit_model_hyperparameters,
            "event_model_config": self.event_model_config,
            "event_model_hyperparameters": self.event_model_hyperparameters,
            "event_freshness_status": self.event_freshness_status.value
            if self.event_freshness_status
            else None,
            "last_event": self.last_event,
            "event_staleness_age_seconds": self.event_staleness_age_seconds,
            "event_predicted_latency_seconds": self.event_predicted_latency_seconds,
            "event_predicted_upper_bound_latency_seconds": self.event_predicted_upper_bound_latency_seconds,
            "blast_radius": dataclasses.asdict(self.blast_radius) if self.blast_radius else None,
        }

    @staticmethod
    def fromdict(json_dict: Dict):
        # Convert epoch seconds to timestamps
        for ts_col in [
            "last_data_update",
            "predicted_next_data_update",
            "predicted_upper_bound_next_data_update",
            "evaluated_at",
            "last_event",
        ]:
            if ts_col in json_dict and json_dict[ts_col] is not None:
                json_dict[ts_col] = datetime.fromtimestamp(json_dict[ts_col])
        if (
            "overall_freshness_status" in json_dict
            and json_dict["overall_freshness_status"] is not None
        ):
            json_dict["overall_freshness_status"] = ResultStatus(
                json_dict["overall_freshness_status"]
            )
        if (
            "commit_freshness_status" in json_dict
            and json_dict["commit_freshness_status"] is not None
        ):
            json_dict["commit_freshness_status"] = ResultStatus(
                json_dict["commit_freshness_status"]
            )
        if (
            "event_freshness_status" in json_dict
            and json_dict["event_freshness_status"] is not None
        ):
            json_dict["event_freshness_status"] = ResultStatus(json_dict["event_freshness_status"])
        if "error_code" in json_dict and json_dict["error_code"] is not None:
            json_dict["error_code"] = ErrorCode(json_dict["error_code"])
        if "blast_radius" in json_dict and json_dict["blast_radius"] is not None:
            json_dict["blast_radius"] = BlastRadiusInfo(**json_dict["blast_radius"])
        return TableFreshnessInfo(**json_dict)

    # Convert this to a dictionary with the relevant metric fields for service/lumberjack logging
    def to_metric_dict(self):
        return {
            "name": self.table_name,
            "id": self.table_id,
            "result": str(self.overall_freshness_status.value)
            if self.overall_freshness_status
            else None,
            "staleness_age": self.staleness_age_seconds,
            "evaluated_at_timestamp": int(self.evaluated_at.timestamp())
            if self.evaluated_at
            else None,
            "last_update_timestamp": int(self.last_data_update.timestamp())
            if self.last_data_update
            else None,
            "predicted_timestamp": int(self.predicted_next_data_update.timestamp())
            if self.predicted_next_data_update
            else None,
            "predicted_upper_bound_timestamp": int(
                self.predicted_upper_bound_next_data_update.timestamp()
            )
            if self.predicted_upper_bound_next_data_update
            else None,
            "confidence_level": self.confidence_level,
            # Clip the error message to avoid going over max content length
            "error_message": self.error_message[:500] if self.error_message else None,
            "error_code": self.error_code.value if self.error_code else None,
            "commit_timestamps": self.commit_timestamps,
            "event_staleness_age": self.event_staleness_age_seconds,
            "event_based_freshness_enabled": self.event_freshness_status is not None,
        }

    @staticmethod
    def from_pyspark_row(row: T.Row):
        return TableFreshnessInfo(
            table_name=row.table_name,
            overall_freshness_status=ResultStatus(row.overall_freshness_status)
            if row.overall_freshness_status
            else None,
            commit_freshness_status=ResultStatus(row.commit_freshness_status)
            if row.commit_freshness_status
            else None,
            staleness_age_seconds=row.staleness_age,
            evaluated_at=row.evaluated_at,
            last_data_update=row.last_data_update,
            predicted_next_data_update=row.predicted_next_data_update,
            predicted_upper_bound_next_data_update=row.predicted_upper_bound_next_data_update,
            confidence_level=row.confidence_level,
            error_message=row.error_message,
            table_lineage_link=row.table_lineage_link,
            event_freshness_status=ResultStatus(row.event_freshness_status)
            if row.event_freshness_status
            else None,
            last_event=row.last_event,
            event_staleness_age_seconds=row.event_staleness_age,
            event_predicted_latency_seconds=row.event_predicted_latency_seconds,
            event_predicted_upper_bound_latency_seconds=row.event_predicted_upper_bound_latency_seconds,
        )
