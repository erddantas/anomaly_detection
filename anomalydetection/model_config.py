"""Configuration for the anomaly detection model"""

import dataclasses
import json
import os
from enum import Enum
from typing import Optional, Any


@dataclasses.dataclass
class BaseModelConfig:
    # The number of commit timestamps we log in elk and background activity logs
    num_commit_timestamps_logged: int = 500
    # The max grace period (used for testing, set to -1 to disable)
    max_grace_period_minutes: float = -1
    # Multiplier on grace period
    grace_period_multiplier: float = 1.0
    # The age threshold to use for determining a static table
    static_table_age_threshold_days: int = 30
    # The max number of commits to use for training the model
    max_commit_training_points: int = 1000
    # The max number of commits to use for training the model
    max_backtesting_commit_training_points: int = 200
    # The max number of timestamps to use for training the event model
    max_event_training_points: int = 360
    # The minimum number of commits needed to train the model
    min_commit_training_points: int = 2
    # The maximum lookback period for describe history (2 weeks)
    max_lookback_days: int = 14
    # The minimum duration between event times in seconds
    min_duration_between_event_times_sec: int = 3600  # 1 hour
    # The minimum number of data points required to evaluate completeness a table
    min_completeness_training_data_size: int = 7
    # Whether to send the health checks to our DM backend API
    enable_put_health_checks: bool = False
    # use grace period on predicted upper bound timestamp for commit rather than grace period on predicted timestamp
    use_grace_period_on_predicted_upper_bound: bool = True
    # Whether we should compute blast radius
    enable_blast_radius_computation: bool = False
    # Whether to rename fresh / stale to healthy / unhealthy
    rename_fresh_stale_to_healthy_unhealthy: bool = True
    # Num days to lookback to fetch downstream tables
    downstream_table_lookback_days: int = 90
    # Enable dashboard
    enable_dashboard: bool = False
    # Whether to enable backtesting automatically when job is created
    enable_backtesting: bool = False
    # Whether to grab history based on timestamp rather than number points
    enable_limit_history_by_timestamp: bool = False
    # The number of eval_points before retraining a model from scratch in backtesting
    eval_points_before_retrain: int = 10

    def __post_init__(self):
        """Ensure instance and subclasses gets the latest class-level value."""
        self.enable_put_health_checks = BaseModelConfig.enable_put_health_checks
        self.enable_blast_radius_computation = BaseModelConfig.enable_blast_radius_computation
        self.enable_dashboard = BaseModelConfig.enable_dashboard
        self.enable_backtesting = BaseModelConfig.enable_backtesting
        self.set_enable_limit_history_by_timestamp = (
            BaseModelConfig.enable_limit_history_by_timestamp
        )


@dataclasses.dataclass
class AutoArimaModelConfig(BaseModelConfig):
    # Defaults to 1, which means no seasonality
    seasonal_period: int = 1
    # The order of first-differencing. Defaults to 0, which means no differencing
    differencing_order: int = 0
    # Whether to infer the seasonal period using autocorrelation function.
    # If True, the seasonal_period will be ignored.
    # Autocorrelation measures how strongly a time series is correlated with its
    # past values at different lags, we can use the ACF to find the repeating patterns
    # in the time series data and infer the seasonality from it.
    infer_seasonality_acf: bool = True
    # Number of lags to compute autocorrelation for, defaults to 50.
    acf_max_lag: int = 52
    # The threshold for determining if a pattern found by ACF is significant enough
    # to be considered as seasonality. Defaults to 0.65
    acf_threshold: float = 0.65
    # boolean arg to determine whether to generate jittered synthetic data based on current data to boost ACF seasonality
    boost_seasonal_period_acf: bool = True
    # boolean arg to determine whether to generate jittered synthetic data should be trained upon
    train_on_jittered_data: bool = False
    # number of times we prepend jittered synthetic data to the original data
    jitter_repetitions: int = 1
    # number of training points at which we will not add synthetic data
    sufficient_training_points: int = 250
    # use a fourier transform + ARIMA to capture seasonality
    use_fourier_transform: bool = True
    # use std on residuals for confidence interval
    use_std_residuals: bool = True
    # If using std on residuals for CI, how many std to look away
    num_std_for_residuals: int = 3


@dataclasses.dataclass
class ModelConfig:
    autoarima_model_config: Optional[AutoArimaModelConfig] = None


class ModelType(Enum):
    """Enum for the model to use for anomaly detection"""

    # Uses autoarima + confidence intervals (no seasonality)
    AUTOARIMA = "autoarima"


# Default model type to use for anomaly detection
DEFAULT_MODEL_TYPE = ModelType.AUTOARIMA

# Environment variables to override the model type and config
MODEL_TYPE_ENV_VAR = "LHM_AD_MODEL_TYPE"
MODEL_CONFIG_ENV_VAR = "LHM_AD_MODEL_CONFIG"


def get_model_type() -> ModelType:
    if MODEL_TYPE_ENV_VAR in os.environ:
        return ModelType(os.environ[MODEL_TYPE_ENV_VAR])
    return DEFAULT_MODEL_TYPE


def get_model_config() -> ModelConfig:
    # If environment variable is set, use the value as the model config
    if MODEL_CONFIG_ENV_VAR in os.environ:
        raw_dict = json.loads(os.environ[MODEL_CONFIG_ENV_VAR])
        arima_config_dict = raw_dict["autoarima_model_config"]
        arima_config = AutoArimaModelConfig(**arima_config_dict) if arima_config_dict else None
        return ModelConfig(autoarima_model_config=arima_config)
    else:
        return ModelConfig(autoarima_model_config=AutoArimaModelConfig())


def get_underlying_model_config() -> BaseModelConfig:
    model_type = get_model_type()
    model_config = get_model_config()
    if model_type == ModelType.AUTOARIMA:
        if not model_config.autoarima_model_config:
            return AutoArimaModelConfig()
        return model_config.autoarima_model_config

    raise ValueError(f"Unsupported model type: {model_type}")


@dataclasses.dataclass
class ModelParams:
    # Model object
    model: Optional[Any] = None
    # Transform object
    transform: Optional[Any] = None
    # the seasonal period arg used to train the model
    seasonal_period: Optional[int] = 1
    # the minimum duration between commit timestamps
    min_val: Optional[int] = None
    # last update timestamp in epoch seconds
    last_update_timestamp: Optional[int] = None


def get_num_commit_timestamps_logged() -> int:
    return get_underlying_model_config().num_commit_timestamps_logged


def get_max_grace_period_minutes() -> float:
    return get_underlying_model_config().max_grace_period_minutes


def get_static_table_age_threshold_days() -> int:
    return get_underlying_model_config().static_table_age_threshold_days


def get_max_commit_training_points() -> int:
    return get_underlying_model_config().max_commit_training_points


def get_max_backtesting_commit_training_points() -> int:
    return get_underlying_model_config().max_backtesting_commit_training_points


def get_max_event_training_points() -> int:
    return get_underlying_model_config().max_event_training_points


def get_min_commit_training_points() -> int:
    return get_underlying_model_config().min_commit_training_points


def get_max_lookback_days() -> int:
    return get_underlying_model_config().max_lookback_days


def get_min_duration_between_event_times_sec() -> int:
    return get_underlying_model_config().min_duration_between_event_times_sec


def get_min_completeness_training_data_size() -> int:
    return get_underlying_model_config().min_completeness_training_data_size


def set_enable_put_health_checks(enable_put_health_checks: bool) -> None:
    BaseModelConfig.enable_put_health_checks = enable_put_health_checks


def get_enable_put_health_checks() -> int:
    return get_underlying_model_config().enable_put_health_checks


def get_grace_period_multiplier() -> float:
    return get_underlying_model_config().grace_period_multiplier


def get_use_grace_period_on_predicted_upper_bound() -> bool:
    return get_underlying_model_config().use_grace_period_on_predicted_upper_bound


def set_enable_blast_radius_computation(enable_blast_radius_computation: bool) -> None:
    BaseModelConfig.enable_blast_radius_computation = enable_blast_radius_computation


def get_enable_blast_radius_computation() -> bool:
    return get_underlying_model_config().enable_blast_radius_computation


def get_rename_fresh_stale_to_healthy_unhealthy() -> bool:
    return get_underlying_model_config().rename_fresh_stale_to_healthy_unhealthy


def get_downstream_table_lookback_days() -> int:
    return get_underlying_model_config().downstream_table_lookback_days


def get_enable_dashboard() -> bool:
    return get_underlying_model_config().enable_dashboard


def set_enable_dashboard(enable_dashboard: bool) -> None:
    BaseModelConfig.enable_dashboard = enable_dashboard


def set_enable_backtesting(enable_backtesting: bool) -> None:
    BaseModelConfig.enable_backtesting = enable_backtesting


def get_enable_backtesting() -> bool:
    return get_underlying_model_config().enable_backtesting


def set_enable_limit_history_by_timestamp(enable_limit_history_by_timestamp: bool) -> None:
    BaseModelConfig.enable_limit_history_by_timestamp = enable_limit_history_by_timestamp


def get_enable_limit_history_by_timestamp() -> bool:
    return get_underlying_model_config().enable_limit_history_by_timestamp


def get_eval_points_before_retrain() -> int:
    return get_underlying_model_config().eval_points_before_retrain
