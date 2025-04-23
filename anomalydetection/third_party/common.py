import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
from pandas import Timestamp
from pyspark import cloudpickle
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from uff.base import Transformer
from uff.tstypes import TimeIndex, TimeIndexedData

logger = logging.getLogger(__name__)


class UnivariateOutlierDetectionAlgorithm(ABC):
    @abstractmethod
    def check_is_anomaly(
        self, timestamp: Union[datetime, Sequence[datetime]], value: Union[float, Sequence[float]]
    ):
        pass

    @abstractmethod
    def train(self, data: TimeIndexedData, **kwargs):
        pass


class MultivariateOutlierDetectionAlgorithm(ABC):
    @abstractmethod
    def check_is_anomaly(
        self,
        timestamp: Union[datetime, Sequence[datetime]],
        value: Union[Dict[Hashable, float], Sequence[Dict[Hashable, float]]],
    ):
        pass

    @abstractmethod
    def train(self, data: TimeIndexedData, **kwargs):
        pass


class ApplyTransformersMixin:
    def __init__(self, transformers=Iterable[Transformer], **kwargs) -> None:
        self.transformers = transformers

    def apply_transforms(self, data: TimeIndexedData) -> TimeIndexedData:
        for t in self.transformers:
            data = t.transform(data).out
        return data

    def apply_fit_transforms(self, data: TimeIndexedData) -> TimeIndexedData:
        for t in self.transformers:
            t.fit(data)
            data = t.transform(data).out
        return data


def entity_transform(entity: List[str]) -> Union[str, Tuple[str]]:
    """Transforms a spark array entity into a tuple, or returns a singleton if no tuple is needed"""
    if len(entity) == 1:
        return entity[0]
    return tuple(entity)


class MultivariateOutlierDetectionAdapter(MultivariateOutlierDetectionAlgorithm):
    """
    This adapter class allows us to replicate multiple instances of a UnivariateOutlierDetectionAlgorithms
    across a multivariate TimeIndexedData object

    The advantage is that this class allows us to have a unified interface for multivariate and univariate
    outlier detection algorithms
    """

    def __init__(
        self,
        univariate_algo_class: Type[UnivariateOutlierDetectionAlgorithm],
        algo_arguments: Optional[Dict[str, Any]] = None,
        is_distributed: bool = False,
    ) -> None:
        self.univariate_algo_class = univariate_algo_class
        self.algo_arguments = algo_arguments or {}
        self.algo_instances = {}
        self.is_distributed = is_distributed

    def train(self, data: DataFrame, time_index: TimeIndex):
        """Train multiple univariate models, one for each entity in `data`.

        Parameters
        ----------
            data (DataFrame):
                Expects three columns:
                1. entity: A spark array identifying the dimensional groupings (in order).
                2. timeseries: A spark array of the timestamps.
                3. metricseries: A spark array of the metric values.
                metricseries[i] is the value at timeseries[i].
            time_index (TimeIndex):
                The time index which the metric series will be placed into.
                Missing values will be replaced with zero.
        """
        if self.is_distributed:
            self.train_distributed(data, time_index)
        else:
            self.train_synchronous(data, time_index)

    def train_synchronous(self, data: DataFrame, time_index: TimeIndex) -> None:
        """Trains synchronously on the driver."""

        @F.udf(returnType="array<double>")
        def f(ts: List[str], ms: List[float]) -> List[float]:
            """Constructs a sequence of metric values in accordance with time_index.

            Parameters
            ----------
                ts (List[str]):
                    List of timestamps.
                ms (List[float]):
                    List of metric values. Each index corresponds to a time in ts.

            Returns
            -------
            List[float]
                The metric values or 0, in order of time_index.
            """
            # dict from timestamp to metric value
            d = {Timestamp(t): m for t, m in zip(ts, ms)}

            # Align with time_index
            return [float(d.get(t, 0)) for t in time_index.timestamp_values]

        df = data.select("entity", f("timeseries", "metricseries").alias("values")).collect()
        values = np.array([r.values for r in df])

        data = TimeIndexedData.from_time_index(
            time_index, values.T, [entity_transform(r.entity) for r in df]
        )

        if len(data.shape) > 2:
            raise ValueError(
                "expected 2-dimenional TimeIndexedData instance, general tensor-like TimeIndexedData "
                "instances are currently not supported"
            )

        for c in data.column_names:
            self.algo_instances[c] = self.univariate_algo_class(**deepcopy(self.algo_arguments))
            try:
                self.algo_instances[c].train(
                    TimeIndexedData.from_time_index(
                        index=data.time_index,
                        values=data[c],
                        column_names=c,
                    )
                )
            except Exception as e:
                logger.error(
                    f"error training model for column {c}, discarding algorithm instance: {e}"
                )
                del self.algo_instances[c]

    def train_distributed(self, data: DataFrame, time_index: TimeIndex):
        """Trains distributed across the spark cluster"""
        univariate_algo_class = self.univariate_algo_class
        algo_arguments = self.algo_arguments

        @F.udf(returnType="binary")
        def f(ts: List[str], ms: List[float]) -> bytes:
            """Spark udf which performs the training.

            Parameters
            ----------
                ts (List[str]):
                    List of timestamps.
                ms (List[float]):
                    List of metric values. Each index corresponds to a time in ts.

            Returns
            -------
            bytes
                The serialized, trained model.
            """
            # univariate_algo_class and algo_arguments are serialized into the closure
            model = univariate_algo_class(**deepcopy(algo_arguments))

            # Mapping from timestamp to metric value
            ttm = {Timestamp(ts[i]): ms[i] for i in range(len(ts))}
            # The existing metric series ms, extended with 0s to the entire time index
            metric_series = [ttm[t] if t in ttm else 0 for t in time_index.timestamp_values]
            timeseries = TimeIndexedData.from_time_index(
                index=time_index,
                values=metric_series,
            )

            model.train(timeseries)

            return cloudpickle.dumps(model)

        # Added to avoid pushdown duplication
        f = f.asNondeterministic()

        raw_models = (
            data.withColumn("model", f("timeseries", "metricseries"))
            .select("entity", "model")
            .collect()
        )

        self.algo_instances = {
            entity_transform(r.entity): cloudpickle.loads(r.model) for r in raw_models
        }

    def check_is_anomaly(self, df: DataFrame) -> List[Tuple[Hashable, str, float]]:
        """Checks if the values from a multivariate time series contains anomalies, at a given timestamp
        Parameters
        ----------
            df (DataFrame):
                Expects three columns:
                1. entity: A spark array identifying the dimensional groupings (in order).
                2. timeseries: A spark array of the timestamps.
                3. metricseries: A spark array of the metric values.
                metricseries[i] is the value at timeseries[i].

        Returns
        -------
        List[Tuple[Any, str, float]]
        List of anomaly records.
        Each record is (entity, timestamp, metric value) pair.
        """
        if self.is_distributed:
            return self.check_is_anomaly_distributed(df)
        else:
            return self.check_is_anomaly_synchronous(df)

    def check_is_anomaly_synchronous(
        self,
        df: DataFrame,
    ) -> List[Tuple[Hashable, str, float]]:
        """Runs detection synchronously on the driver."""
        data = df.select("entity", "timeseries", "metricseries").collect()
        output = []
        for row in data:
            e, ts, ms = entity_transform(row.entity), row.timeseries, row.metricseries
            m = self.algo_instances.get(e)
            if m is None:
                logger.warn(
                    f"trained model for column {e} not found, no anomalies will be labeled for this segment"
                )
            else:
                pairs = sorted(zip(ts, ms))
                ts = [t for t, _ in pairs]
                ms = [m for _, m in pairs]

                anomalies = m.check_is_anomaly(ts, ms)
                output.extend(
                    (e, time, metric)
                    for time, metric, is_anomaly in zip(ts, ms, anomalies)
                    if is_anomaly
                )
        return output

    def check_is_anomaly_distributed(self, df: DataFrame) -> List[Tuple[Hashable, str, float]]:
        """Runs detection distributed on executors."""
        logger.info("Starting distributed inference run.")

        algo_instances = self.algo_instances
        count = 0
        # TODO: Replace this with loading individual model instances from mlflow in the udf
        for algo in algo_instances.values():
            # Remove init data to prevent size overflow when broadcasted for udf
            if algo.__class__.__name__ == "SPOT" and hasattr(algo, "init_data"):
                del algo.init_data
                count += 1

        logger.info(f"Removed {count} instances of init_data from SPOT algorithms.")

        @F.udf(returnType="map<string, double>")
        def f(e, ts: List[str], ms: List[float]) -> bytes:
            """Spark udf which performs the detection on a particular entity.

            Parameters
            ----------
                e (Any):
                    The entity.
                ts (List[str]):
                    List of timestamps.
                ms (List[float]):
                    List of metric values. Each index corresponds to a time in ts.

            Returns
            -------
            Dict[str, float]
                Mapping from anomalous timestamps to the corresponding anomalous metric value.
            """
            entity = entity_transform(e)
            m = algo_instances.get(entity)

            if m is None:
                return []

            pairs = sorted(zip(ts, ms))
            ts = [t for t, _ in pairs]
            ms = [m for _, m in pairs]

            anomalies = m.check_is_anomaly(ts, ms)
            return {str(t): float(m) for t, m, isanomaly in zip(ts, ms, anomalies) if isanomaly}

        logger.info("Created inference udf.")

        # Added to avoid pushdown duplication
        f = f.asNondeterministic()

        logger.info("Set udf to non-deterministic.")

        out = (
            df.withColumn("output", f("entity", "timeseries", "metricseries"))
            .select("entity", F.explode("output"))
            .select("entity", F.col("key").alias("time"), F.col("value").alias("metric"))
            .collect()
        )

        logger.info("Distributed inference complete.")

        return [(entity_transform(e), t, m) for e, t, m in out]
