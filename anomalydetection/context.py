"""
Introduces main Context class and the framework to specify different specialized
contexts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable
from IPython import get_ipython

from pyspark import sql

from databricks.sdk import WorkspaceClient


class ContextMeta(ABC.__class__):
    """
    Metaclass for Context to allow setting the current context instance.
    """

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls._current = None

    @property
    def active(cls) -> bool:
        return cls._current is not None

    @property
    def current(cls) -> Any:
        if not cls.active:
            raise Exception("No available context")
        return cls._current

    @current.setter
    def current(cls, context) -> None:
        if context is not None and cls.active:
            raise Exception("Context already set")
        cls._current = context

    @property
    def spark(cls) -> sql.SparkSession:
        return cls.current.get_spark()

    @property
    def workspace_client(cls) -> WorkspaceClient:
        return cls.current.get_workspace_client()

    @property
    def dbutils(cls):
        return cls.current.get_dbutils()

    # Hack to be able to use the pretty display() from databricks notebooks
    @property
    def display(self) -> Callable[[Any], None]:
        ipy = get_ipython()
        if ipy:
            return ipy.user_ns["display"]
        else:
            return lambda x: print(x)

    def clear(cls) -> None:
        cls.current.clear_context()
        cls.current = None


class Context(ABC, metaclass=ContextMeta):
    """
    Abstract class for Data Monitoring execution context.
    """

    @abstractmethod
    def display_html(self, html: str) -> None:
        """
        Display HTML in the output.
        """
        pass

    @abstractmethod
    def get_spark(self) -> sql.SparkSession:
        pass

    @abstractmethod
    def get_workspace_client(self) -> WorkspaceClient:
        pass

    @abstractmethod
    def get_dbutils(self):
        pass
