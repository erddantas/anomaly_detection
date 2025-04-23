from enum import Enum
from databricks.data_monitoring.anomalydetection import model_config


class ErrorCode(Enum):
    NO_UPDATES_IN_TABLE_HISTORY = "NO_UPDATES_IN_TABLE_HISTORY"
    FAILED_TO_FIT_MODEL = "FAILED_TO_FIT_MODEL"
    NOT_ENOUGH_UPDATE_OP = "NOT_ENOUGH_UPDATE_OP"
    NOT_ENOUGH_UPDATE_OP_BACKTESTING = "NOT_ENOUGH_UPDATE_OP_BACKTESTING"
    USER_CONFIGURED_SKIP = "USER_CONFIGURED_SKIP"
    FAILED_TO_PREDICT = "FAILED_TO_PREDICT"
    NOT_ENOUGH_TABLE_HISTORY = "NOT_ENOUGH_TABLE_HISTORY"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    INTERNAL_ERROR = "INTERNAL_ERROR"  # Error code that captures all error messages not defined below or anticipated
    USER_ERROR = "USER_ERROR"  # No specific error message for USER_ERROR, since it is a catch-all for user-related error messages we throw
    BLAST_RADIUS_COMPUTATION_ERROR = "BLAST_RADIUS_COMPUTATION_ERROR"


ERROR_CODE_TO_MESSAGE = {
    ErrorCode.NO_UPDATES_IN_TABLE_HISTORY: "The table does not have any update operations in the table history.",
    ErrorCode.FAILED_TO_FIT_MODEL: "Failed to fit model on the table history.",
    ErrorCode.NOT_ENOUGH_UPDATE_OP_BACKTESTING: f"Not enough update operations in table history (<= {model_config.get_min_commit_training_points()} entries) for backtesting.",
    ErrorCode.NOT_ENOUGH_UPDATE_OP: f"Not enough update operations in table history (< {model_config.get_min_commit_training_points()} entries) for a valid forecast.",
    ErrorCode.USER_CONFIGURED_SKIP: "User-specified override to skip table.",
    ErrorCode.FAILED_TO_PREDICT: "Failed to generate prediction from the model.",
    ErrorCode.NOT_ENOUGH_TABLE_HISTORY: f"At least {model_config.get_min_completeness_training_data_size()} "
    f"days of table history is required to evaluate completeness.",
    ErrorCode.PERMISSION_DENIED: "Unable to retrieve table history. Please ensure you have SELECT access to the table.",
    ErrorCode.INTERNAL_ERROR: "An internal error has occurred. Please contact support.",
    ErrorCode.BLAST_RADIUS_COMPUTATION_ERROR: "Failed to compute blast radius.",
}

# Reverse mapping (Message to ErrorCode) for fast lookup
MESSAGE_TO_ERROR_CODE = {v: k for k, v in ERROR_CODE_TO_MESSAGE.items()}


# Match error message to its corresponding code. If no matching error code, return default error code
def match_error_message_to_code(
    error_message: str, default_error_code: ErrorCode = ErrorCode.INTERNAL_ERROR
) -> ErrorCode:
    matched_error_code = MESSAGE_TO_ERROR_CODE.get(error_message, default_error_code)
    return matched_error_code
