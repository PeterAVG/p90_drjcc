import enum
import math
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Setup:
    rng: np.random.Generator
    no_ev: int  # number of ev charging stations
    no_days: int  # number of days to simulate
    no_hours_per_day: int  # number of hours in a day
    no_minutes_per_hour: int  # number of minutes in an hour
    # ev charging parameters
    charge_rate: float
    max_charge_rate: float
    avg_charge_time: float
    std_charge_time: float
    # percentage of days where no charging is allowed
    prob_of_no_charging: float

    # total time period for the simulation, e.g., 60 hours
    total_time: int = field(init=False)

    def __post_init__(self) -> None:
        self.total_time = (
            self.no_days * self.no_hours_per_day * self.no_minutes_per_hour
        )


class IS_OOS_Enum(enum.Enum):
    IS = 1
    OOS = 2


class Method_Enum(enum.Enum):
    ALSO_X = "also_x"
    DRJCC = "drjcc"
    CVAR = "cvar"


@dataclass
class CrossValidator:
    # variables related to an instance on a fold
    no_samples: int  # for empirical distribution
    no_ev_samples: int  # on. evs in portfolio
    # in-sample period start and end
    in_sample_start: int
    in_sample_end: int
    # out-of-sample period start and end
    out_sample_start: int
    out_sample_end: int

    def __post_init__(self) -> None:
        assert self.in_sample_start < self.in_sample_end
        assert self.out_sample_start < self.out_sample_end
        assert self.in_sample_end <= self.out_sample_start
        # assert all above 0
        assert self.no_samples > 0
        assert self.no_ev_samples > 0
        assert self.in_sample_start >= 0
        assert self.in_sample_end >= 0
        assert self.out_sample_start >= 0
        assert self.out_sample_end >= 0

    def no_is_days(self, setup: Setup) -> int:
        d = (self.in_sample_end - self.in_sample_start) / (
            setup.no_hours_per_day * setup.no_minutes_per_hour
        )
        assert d % 1 == 0
        return int(d)

    def no_oos_days(self, setup: Setup) -> int:
        d = (self.out_sample_end - self.out_sample_start) / (
            setup.no_hours_per_day * setup.no_minutes_per_hour
        )
        assert d % 1 == 0
        return int(d)


@dataclass
class ALSO_X:
    M: float
    q: float


@dataclass
class DRJCC:
    M: float
    theta_list: List[float] = field(init=False)
    # theta: float

    def __post_init__(self) -> None:
        self.theta_list = [
            0,
            1e-07,
            1e-06,
            1e-05,
            1e-05 * 2,
            1e-05 * 3,
            1e-05 * 4,
            1e-05 * 5,
            1e-04,
            1e-04 * 2,
            1e-04 * 3,
            1e-04 * 4,
            1e-04 * 5,
            1e-03,
            1e-03 * 2,
            1e-03 * 3,
            1e-03 * 4,
            1e-03 * 5,
            1e-02,
            1e-02 * 2,
            1e-02 * 3,
            1e-02 * 4,
            1e-02 * 5,
            1e-01,
            1e-01 * 2,
            1e-01 * 3,
            1e-01 * 4,
            1e-01 * 5,
            1.0,
            1.0 + 0.2,
            1.0 + 0.3,
            1.0 + 0.4,
            1.0 + 0.5,
            2.0,
            2.0 + 0.2,
            2.0 + 0.3,
            2.0 + 0.4,
            2.0 + 0.5,
            10,
            20,
            30,
            40,
            50,
            100,
        ]


@dataclass
class OptimizationInstance:
    setup: Setup
    epsilon: float
    no_samples: int
    prices: np.ndarray
    ev_kw: np.ndarray
    empirical_distribution: np.ndarray
    cv: CrossValidator

    also_x: ALSO_X = field(init=False)
    drjcc: DRJCC = field(init=False)

    def __post_init__(self) -> None:
        assert 0 <= self.cv.no_ev_samples <= self.setup.no_ev
        assert self.prices.shape == (24,)
        assert self.no_samples == self.cv.no_samples
        assert self.empirical_distribution.shape[0] == self.no_samples
        self.also_x = ALSO_X(
            M=self.cv.no_ev_samples * self.setup.max_charge_rate,
            q=math.floor(self.epsilon * self.no_samples * 24 * 60),
        )
        # self.drjcc = DRJCC(M=self.cv.no_ev_samples * self.setup.max_charge_rate)
        # self.drjcc = DRJCC(M=10**6)
        # self.drjcc = DRJCC(M=0)
        # theta = 1 / self.no_samples * self.empirical_distribution.std() * 3
        self.drjcc = DRJCC(
            # M=1 / self.no_samples * self.empirical_distribution.std() * 3,
            # M=1 / self.no_samples * self.empirical_distribution.std() * 3,
            M=np.mean(self.empirical_distribution),
            # M=10**6,
            # theta_list=[theta],
        )

    @staticmethod
    def get_aggregate_kw(ev_kw: np.ndarray) -> np.ndarray:
        return np.nansum(ev_kw, axis=0)

    @property
    def aggregate_kw(self) -> np.ndarray:
        return self.get_aggregate_kw(self.ev_kw)

    @property
    def max_kw(self) -> float:
        return self.cv.no_ev_samples * self.setup.max_charge_rate

    @property
    def average_empirical_distribution(self) -> np.ndarray:
        return np.mean(self.empirical_distribution, axis=0)

    @property
    def max_of_mean_empirical_distribution(self) -> float:
        return self.average_empirical_distribution.max()


@dataclass
class Result:
    p_cap: np.ndarray
    p_realized: np.ndarray
    violation: np.ndarray
    is_oos: IS_OOS_Enum
    method: Method_Enum

    revenue: float
    penalty: float

    # frequency of violations
    freq: float = field(init=False)

    def __post_init__(self) -> None:
        assert self.p_cap.shape[0] % 24 == 0
        assert self.p_realized.shape[0] % 24 == 0
        assert self.p_realized.shape[0] % (24 * 60) == 0
        assert self.violation.shape == self.p_realized.shape

        self.freq = np.sum(self.violation > 0) / np.size(self.violation)  # type:ignore

    def get_mean_realized_kw(self) -> np.ndarray:
        return np.nanmean(self.p_realized.reshape(24, -1), axis=1)

    def get_mean_violation_kw(self) -> np.ndarray:
        return np.nanmean(self.violation.reshape(24, -1), axis=1)

    def get_freq_distribution(self) -> np.ndarray:
        ar = self.violation.reshape(60, -1)
        ar = np.sum(ar > 0, axis=0) / 60
        assert np.isclose(np.mean(ar), self.freq)
        return ar

    def get_violation_distribution(self) -> np.ndarray:
        ar = self.violation[self.violation > 0]
        if ar.shape[0] == 0:
            return np.array([0.0])
        return ar
