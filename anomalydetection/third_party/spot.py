import logging
from math import log
from typing import Iterable, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from uff.base import Forecaster, Transformer
from uff.tstypes import TimeIndexedData, TimeStamp

from databricks.data_monitoring.anomalydetection.third_party.common import (
    ApplyTransformersMixin,
    UnivariateOutlierDetectionAlgorithm,
)

logger = logging.getLogger(__name__)
"""
This code was adapted from the version published under the GNU GPLv3 license @ https://github.com/Amossys-team/SPOT
The SPOT algorithm is described in the paper
    Alban Siffer, Pierre-Alain Fouque, Alexandre Termier, and Christine Largouet. 2017. Anomaly Detection in Streams
    with Extreme Value Theory. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery
    and Data Mining (KDD '17). Association for Computing Machinery, New York, NY, USA, 1067–1075.
    https://doi.org/10.1145/3097983.3098144
"""

# colors for plot
deep_saffron = "#FF9933"
air_force_blue = "#5D8AA8"


def backMean(X: Sequence[float], d: int) -> np.ndarray:
    M = []
    w = X[:d].sum()
    M.append(w / d)
    for i in range(d, len(X)):
        w = w - X[i - d] + X[i]
        M.append(w / d)
    return np.array(M)


class SPOT(UnivariateOutlierDetectionAlgorithm, ApplyTransformersMixin):
    """
    This class allows to run SPOT or DSPOT algorithm on univariate dataset (upper and lower bounds)

    Attributes
    ----------
    transformers : Optional[Iterable[Transformer]]
        Optional set of transformers.  If specified, the transformations will be applied prior to
        checking the SPOT bounds

    model: Optional[Forecaster]
        Optional model.  If specified, SPOT will be applied on the residuals of the fitted model

    proba : float
        Detection level (risk), chosen by the user

    depth : int
        Number of observations to compute the moving average

    direction : Optional[str]
        This argument allows us to apply the SPOT algorithm in a single direction ("up" or "down")
        If left unspecified or an invalid option is passed, this assumes the algorithm will be applied in both direction

    add_trend_component : bool
        if True, run DSPOT instead of SPOT

    extreme_quantile : float
        current threshold (bound between normal and abnormal events)

    init_data : pandas.Series
        initial batch of observations (for the calibration/initialization step)

    init_threshold : float
        initial threshold computed during the calibration step

    peaks : numpy.array
        array of peaks (excesses above the initial threshold)

    n : int
        number of observed values

    Nt : int
        number of observed peaks

    W : numpy.array
        window, only used in DSPOT to calculate moving average
    """

    def __init__(
        self,
        transformers: Optional[Iterable[Transformer]] = None,
        model: Optional[Forecaster] = None,
        q: float = 1e-4,
        depth: int = 10,
        direction: Optional[str] = None,
        add_trend_component: bool = False,
    ):
        super().__init__(transformers or [])
        self.model = model
        self.proba = q
        self.data = None
        self.init_data = None
        self.init_data_size = None
        self.n = 0
        self.depth = depth
        self.add_trend_component = add_trend_component

        possible_directions = ("up", "down")
        self.direction = (direction,) if direction in possible_directions else possible_directions
        nonedict = {d: None for d in self.direction}

        self.extreme_quantile = dict.copy(nonedict)
        self.init_threshold = dict.copy(nonedict)
        self.peaks = dict.copy(nonedict)
        self.gamma = dict.copy(nonedict)
        self.sigma = dict.copy(nonedict)
        self.Nt = {d: 0 for d in self.direction}
        self.W = np.array([])

    def __str__(self):
        s = ""
        s += "Streaming Peaks-Over-Threshold Object\n"
        s += "Detection level q = %s\n" % self.proba
        if self.data is not None:
            s += "Data imported : Yes\n"
            s += "\t initialization  : %s values\n" % self.init_data_size
            s += "\t stream : %s values\n" % self.data.size
        else:
            s += "Data imported : No\n"
            return s

        if self.n == 0:
            s += "Algorithm initialized : No\n"
        else:
            s += "Algorithm initialized : Yes\n"
            s += "\t initial threshold : %s\n" % self.init_threshold

            r = self.n - self.init_data_size
            if r > 0:
                s += "Algorithm run : Yes\n"
                s += "\t number of observations : %s (%.2f %%)\n" % (r, 100 * r / self.n)
                s += "\t triggered alarms : %s (%.2f %%)\n" % (
                    len(self.alarm),
                    100 * len(self.alarm) / self.n,
                )
            else:
                s += "\t number of peaks  : %s\n" % self.Nt
                s += "\t upper extreme quantile : %s\n" % self.extreme_quantile["up"]
                s += "\t lower extreme quantile : %s\n" % self.extreme_quantile["down"]
                s += "Algorithm run : No\n"
        return s

    def initialize(self, data: TimeIndexedData, verbose=True, **kwargs) -> None:
        """
        Run the calibration (initialization) step

        Parameters
        ----------
        init_data : TimeIndexedData
            initial batch of observations (for the calibration/initialization step)

        verbose : bool
            (default=True) If True, gives details about the batch initialization
        """
        data = self.apply_fit_transforms(data)

        if self.model is not None:
            self.model.fit(data)
            data -= self.model.forecast(data=data.time_index).out

        self.data = pd.DataFrame(
            {
                "val": float,
                "upper_thresholds": float,
                "lower_thresholds": float,
                "alarms": bool,
            },
            index=[],
        )
        self.init_data = (
            data.to_pandas(time_col="ts", time_as_pd_timestamps=True)
            .set_index("ts")
            .squeeze("columns")
        )
        if self.init_data.hasnans:
            logger.info("input data contains nans after transformation")
            prev_len = len(self.init_data)
            self.init_data = self.init_data.dropna()
            logger.info(
                f"dropping nans.  prior length {prev_len}, length after dropping nans {len(self.init_data)}"
            )

        self.W = np.array(self.init_data[-self.depth :])
        if self.add_trend_component:
            n_init = self.init_data.size - self.depth
            M = backMean(self.init_data, self.depth)
            T = self.init_data[self.depth :] - M[:-1]
        else:
            n_init = self.init_data.size
            T = self.init_data

        S = np.sort(T)  # we sort T to get the empirical quantile
        eps = 1e-8
        if "up" in self.init_threshold:
            self.init_threshold["up"] = S[int(0.98 * n_init)]  # t is fixed for the whole algorithm
            # initial peaks
            self.peaks["up"] = np.array(
                T[T > self.init_threshold["up"]] - self.init_threshold["up"]
            )
            self.Nt["up"] = self.peaks["up"].size
            # if Nt is 0, this will throw an error in the _rootsFinder subroutine
            # if this happens, let's introduce a peak and move on
            if self.Nt["up"] == 0:
                self.peaks["up"] = np.array(eps)
                self.Nt["up"] += 1
        if "down" in self.init_threshold:
            # same steps as in the "up" direction
            self.init_threshold["down"] = S[
                int(0.02 * n_init)
            ]  # t is fixed for the whole algorithm
            self.peaks["down"] = -(T[T < self.init_threshold["down"]] - self.init_threshold["down"])
            self.Nt["down"] = self.peaks["down"].size
            if self.Nt["down"] == 0:
                self.peaks["down"] = np.array(eps)
                self.Nt["down"] += 1
        self.n = n_init

        if verbose:
            logger.info("Initial threshold : %s" % self.init_threshold)
            logger.info("Number of peaks : %s" % self.Nt)
            logger.info("Grimshaw maximum log-likelihood estimation ... ")

        ell = {d: None for d in self.direction}
        for side in ell.keys():
            g, s, ell[side] = self._grimshaw(side)
            self.extreme_quantile[side] = self._quantile(side, g, s)
            self.gamma[side] = g
            self.sigma[side] = s

        ltab = 20
        form = "\t" + "%20s" + "%20.2f" + "%20.2f"
        if verbose:
            logger.info("[done]")
            logger.info("\t" + "Parameters".rjust(ltab) + "Upper".rjust(ltab) + "Lower".rjust(ltab))
            logger.info("\t" + "-" * ltab * 3)
            logger.info(
                form % (chr(0x03B3), self.gamma.get("up", np.NaN), self.gamma.get("down", np.NaN))
            )
            logger.info(
                form % (chr(0x03C3), self.sigma.get("up", np.NaN), self.sigma.get("down", np.NaN))
            )
            logger.info(form % ("likelihood", ell.get("up", np.NaN), ell.get("down", np.NaN)))
            logger.info(
                form
                % (
                    "Extreme quantile",
                    self.extreme_quantile.get("up", np.NaN),
                    self.extreme_quantile.get("down", np.NaN),
                )
            )
            logger.info("\t" + "-" * ltab * 3)

        self.init_data_size = self.init_data.size
        return

    train = initialize

    def _rootsFinder(self, fun, jac, bounds, npoints, method) -> np.ndarray:
        """
        Find possible roots of a scalar function

        Parameters
        ----------
        fun : function
            scalar function
        jac : function
            first order derivative of the function
        bounds : tuple
            (min,max) interval for the roots search
        npoints : int
            maximum number of roots to output
        method : str
            "regular" : regular sample of the search interval
            "random" : uniform (distribution) sample of the search interval

        Returns
        ----------
        numpy.array
            possible roots of the function
        """
        if method == "regular":
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == "random":
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx**2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(
            lambda X: objFun(X, fun, jac),
            X0,
            method="L-BFGS-B",
            jac=True,
            bounds=[bounds] * len(X0),
        )

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(self, Y: np.ndarray, gamma: float, sigma: float) -> float:
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)

        Parameters
        ----------
        Y : numpy.array
            observations
        gamma : float
            GPD index parameter
        sigma : float
            GPD scale parameter (>0)

        Returns
        ----------
        float
            log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, side: str, epsilon: float = 1e-8, n_points: int = 8):
        """
        Compute the GPD parameters estimation with the Grimshaw"s trick

        Parameters
        ----------
        epsilon : float
            numerical parameter to perform (default : 1e-8)
        n_points : int
            maximum number of candidates for maximum likelihood (default : 10)

        Returns
        ----------
        gamma_best,sigma_best,ll_best
            gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s**2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks[side].min()
        YM = self.peaks[side].max()
        Ymean = self.peaks[side].mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        b = max(2 * (Ymean - Ym) / (Ymean * Ym), epsilon)
        c = max(2 * (Ymean - Ym) / (Ym**2), 2 * epsilon)

        # We look for possible roots
        left_zeros = self._rootsFinder(
            lambda t: w(self.peaks[side], t),
            lambda t: jac_w(self.peaks[side], t),
            (a + epsilon, -epsilon),
            n_points,
            "regular",
        )

        right_zeros = self._rootsFinder(
            lambda t: w(self.peaks[side], t),
            lambda t: jac_w(self.peaks[side], t),
            (b, c),
            n_points,
            "regular",
        )

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = self._log_likelihood(self.peaks[side], gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks[side]) - 1
            sigma = gamma / z
            ll = self._log_likelihood(self.peaks[side], gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, side: str, gamma: float, sigma: float) -> float:
        """
        Compute the quantile at level 1-q for a given side

        Parameters
        ----------
        side : str
            "up" or "down"
        gamma : float
            GPD parameter
        sigma : float
            GPD parameter

        Returns
        ----------
        float
            quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        if side == "up":
            r = self.n * self.proba / self.Nt[side]
            if gamma != 0:
                return self.init_threshold["up"] + (sigma / gamma) * (pow(r, -gamma) - 1)
            else:
                return self.init_threshold["up"] - sigma * log(r)
        elif side == "down":
            r = self.n * self.proba / self.Nt[side]
            if gamma != 0:
                return self.init_threshold["down"] - (sigma / gamma) * (pow(r, -gamma) - 1)
            else:
                return self.init_threshold["down"] + sigma * log(r)
        else:
            logger.error("the side is not right")

    def check_is_anomaly(
        self, timestamp: Union[TimeStamp, Sequence[TimeStamp]], val: Union[float, Sequence[float]]
    ) -> List[bool]:
        if self.data is None:
            raise ValueError(
                "data attribute is None, the algorithm might not have been initialized"
            )

        series = TimeIndexedData(time_array=timestamp, values=val, column_names=["val"])
        series = self.apply_transforms(series)
        if self.model is not None:
            series -= self.model.forecast(data=series).out.set_column_names("val")
        timestamp = series.pd_timestamp_index()
        val = series.values
        return [self.apply_spot_single_point(t, v) for t, v in zip(timestamp, val)]

    def apply_spot_single_point(self, timestamp: pd.Timestamp, val: float) -> bool:
        Mi = self.W.mean() if self.add_trend_component else 0
        Ni = val - Mi

        if Ni > self.extreme_quantile.get("up", np.inf) or Ni < self.extreme_quantile.get(
            "down", -np.inf
        ):
            self.data.loc[timestamp] = {
                "val": val,
                "upper_thresholds": self.extreme_quantile["up"] + Mi
                if "up" in self.direction
                else np.NaN,
                "lower_thresholds": self.extreme_quantile["down"] + Mi
                if "down" in self.direction
                else np.NaN,
                "alarms": True,
            }
            return True
        elif Ni > self.init_threshold.get("up", np.inf) or Ni < self.init_threshold.get(
            "down", -np.inf
        ):
            direction = "up" if Ni > self.init_threshold.get("up", np.inf) else "down"
            self.peaks[direction] = np.append(
                self.peaks[direction],
                Ni - self.init_threshold["up"]
                if direction == "up"
                else -(Ni - self.init_threshold["down"]),
            )
            self.Nt[direction] += 1
            g, s, _ = self._grimshaw(direction)
            self.extreme_quantile[direction] = self._quantile(direction, g, s)

        self.n += 1
        self.W = np.append(self.W[1:], val)
        self.data.loc[timestamp] = {
            "val": val,
            "upper_thresholds": self.extreme_quantile["up"] + Mi
            if "up" in self.direction
            else np.NaN,
            "lower_thresholds": self.extreme_quantile["down"] + Mi
            if "down" in self.direction
            else np.NaN,
            "alarms": False,
        }
        return False

    def plot(self, with_alarm: bool = True) -> None:
        """
        Plot the results given by the run

        Parameters
        ----------
        with_alarm : bool
            (default=True) If True, alarms are plotted.


        Returns
        ----------
        list
            list of the plots

        """
        if self.data is None:
            raise ValueError(
                "data attribute is None, the algorithm might not have been initialized"
            )
        elif self.data.size < 2:
            raise ValueError("insufficient oberservations for a plot")

        self.data.val.plot(color=air_force_blue)
        self.data.upper_thresholds.plot(color=deep_saffron, lw=2, ls="dashed")
        self.data.lower_thresholds.plot(color=deep_saffron, lw=2, ls="dashed")

        alarms = self.data.query("alarms").alarms
        if with_alarm and len(alarms) > 0:
            plt.scatter(alarms.index, self.data.loc[alarms.index, "val"], color="red")
        return
