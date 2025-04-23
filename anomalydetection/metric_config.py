"""
This module contains configuration classes for metric checkers.
"""

from datetime import timedelta
from typing import Optional, List, Dict


class MetricConfig:
    """Base configuration for all metric checkers, containing shared settings."""

    def __init__(
        self,
        disable_check: Optional[bool] = False,
        tables_to_skip: Optional[List[str]] = None,
        tables_to_scan: Optional[List[str]] = None,
        logging_table_name: Optional[str] = None,
    ):
        self.disable_check = disable_check
        self.tables_to_skip = tables_to_skip
        self.tables_to_scan = tables_to_scan
        self.logging_table_name = logging_table_name

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class FreshnessConfig(MetricConfig):
    """Configuration for FreshnessChecker, extending MetricConfig."""

    def __init__(
        self,
        table_threshold_overrides: Optional[Dict[str, timedelta]] = None,
        table_latency_threshold_overrides: Optional[Dict[str, timedelta]] = None,
        static_table_threshold_override: Optional[timedelta] = None,
        event_timestamp_col_names: Optional[List[str]] = None,
        enable_debug_info: Optional[bool] = False,
        **kwargs,  # Pass shared configs to the base class
    ):
        super().__init__(**kwargs)
        self.table_threshold_overrides = table_threshold_overrides
        self.table_latency_threshold_overrides = table_latency_threshold_overrides
        self.static_table_threshold_override = static_table_threshold_override
        self.event_timestamp_col_names = event_timestamp_col_names
        self.enable_debug_info = enable_debug_info

    def _serialize_timedelta(self, td: Optional[timedelta]):
        if td is None:
            return None
        return td.total_seconds()  # store as seconds

    def _serialize_timedelta_dict(self, d: Optional[Dict[str, timedelta]]):
        if d is None:
            return None
        return {k: self._serialize_timedelta(v) for k, v in d.items()}

    def to_dict(self):
        base = super().to_dict()
        base.update(
            {
                "table_threshold_overrides": self._serialize_timedelta_dict(
                    self.table_threshold_overrides
                ),
                "table_latency_threshold_overrides": self._serialize_timedelta_dict(
                    self.table_latency_threshold_overrides
                ),
                "static_table_threshold_override": self._serialize_timedelta(
                    self.static_table_threshold_override
                ),
                "event_timestamp_col_names": self.event_timestamp_col_names,
                "enable_debug_info": self.enable_debug_info,
            }
        )
        return base

    @classmethod
    def from_dict(cls, d):
        # Convert numeric seconds back to timedelta
        def deserialize_td(val):
            return timedelta(seconds=val) if val is not None else None

        if d.get("table_threshold_overrides"):
            d["table_threshold_overrides"] = {
                k: deserialize_td(v) for k, v in d["table_threshold_overrides"].items()
            }
        if d.get("table_latency_threshold_overrides"):
            d["table_latency_threshold_overrides"] = {
                k: deserialize_td(v) for k, v in d["table_latency_threshold_overrides"].items()
            }
        if d.get("static_table_threshold_override") is not None:
            d["static_table_threshold_override"] = deserialize_td(
                d["static_table_threshold_override"]
            )
        # Extract shared configs if needed
        return cls(**d)


class CompletenessConfig(MetricConfig):
    """Configuration for CompletenessChecker, extending MetricConfig."""

    def __init__(
        self,
        table_threshold_overrides: Optional[Dict[str, int]] = None,
        **kwargs,  # Pass shared configs to the base class
    ):
        super().__init__(**kwargs)
        self.table_threshold_overrides = table_threshold_overrides

    def to_dict(self):
        base = super().to_dict()
        base.update(
            {
                "table_threshold_overrides": self.table_threshold_overrides,
            }
        )
        return base

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
