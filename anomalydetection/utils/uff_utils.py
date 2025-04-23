"""Utilities to install UFF"""

import subprocess
import importlib.resources


def install_uff():
    try:
        import uff  # pylint: disable=unused-import
    except (ImportError, ModuleNotFoundError):
        # Hack to install UFF
        with importlib.resources.path(
            "databricks.data_monitoring.anomalydetection.resources",
            "uff-0.4.2-py3-none-any.whl",
        ) as wheel_path:
            subprocess.check_call(
                [
                    "pip",
                    "install",
                    str(wheel_path),
                    "--quiet",
                    "--disable-pip-version-check",
                ]
            )
