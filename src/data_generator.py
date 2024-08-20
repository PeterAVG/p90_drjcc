from typing import Tuple

import numba
import numpy as np

from src.base import CrossValidator, IS_OOS_Enum, Setup
from src.utils import timing


def generate_probability_vector(rng: np.random.Generator) -> np.ndarray:
    # create probability distribution over 24 hours using cosine function which peaks at 00:00
    p = np.cos(np.linspace(0, 2 * np.pi, 24)) + 1
    print(np.argmax(p))
    # assert np.argmax(p) == 17
    p += rng.standard_normal(24)  # add some noise
    p = np.abs(p)  # take absolute value
    p = p / np.sum(p)  # normalize the probabilities to sum to 1
    assert p.shape == (24,)
    return p


def draw_charging_events(
    rng: np.random.Generator, setup: Setup, prob: np.ndarray
) -> np.ndarray:
    # draw no_ev charging times for each day
    size = (setup.no_ev, setup.no_days)  # size of the events array
    events = rng.choice(np.arange(24), size=size, p=prob)
    prob_of_no_charging = setup.prob_of_no_charging
    # for no_charging % of the time, set events to 0
    mask = rng.choice(
        [np.nan, 1], size=size, p=[prob_of_no_charging, 1 - prob_of_no_charging]
    )
    events = events * mask
    assert (events[~np.isnan(events)] >= 0).all()
    return events


@timing()
@numba.njit(
    [
        (
            numba.types.int64,
            numba.types.int64,
            numba.types.int64,
            numba.types.int64,
            numba.types.float64,
            numba.types.float64,
            numba.types.float64,
            numba.types.float64,
            numba.types.float64[:, :],
            numba.types.float64[:, :],
        )
    ],
    cache=True,
    # np.isnan does not work with fastmath=True:
    # https://github.com/numba/numba/issues/6807
    fastmath=True,
)
def compute_ev_consumption(
    no_ev: int,
    no_minutes_per_hour: int,
    no_hours_per_day: int,
    total_time: int,
    charge_rate: float,
    max_charge_rate: float,
    avg_charge_time: float,
    std_charge_time: float,
    events: np.ndarray,
    ev_kw: np.ndarray,
) -> None:
    np.random.seed(1)
    rng = np.random
    for i in range(no_ev):
        # print(i, end="\r")
        for j in range(total_time):
            # j is minute from 1...total_time
            day = j // (no_hours_per_day * no_minutes_per_hour)
            day_of_week = day % 7
            hour = j // no_minutes_per_hour
            hour_of_day = hour % 24

            if events[i][day] == hour_of_day and ev_kw[i][j] == 0:
                # charge_rate_ = charge_rate + rng.normal() * 0.5
                charge_rate_ = charge_rate + rng.standard_cauchy() * 0.5
                charge_rate_ += rng.normal(0, np.sqrt(day))
                charge_rate_ = max(0, min(charge_rate_, max_charge_rate))
                ev_kw[i][j] = charge_rate_
                if day_of_week < 5:  # Weekdays
                    # draw charge time from a normal distribution with higher charge time on weekdays
                    # charge_time = rng.normal(avg_charge_time, std_charge_time)
                    charge_time = rng.standard_t(3) + avg_charge_time
                    charge_time += rng.normal(0, np.sqrt(day))
                else:  # Weekends
                    # draw charge time from a normal distribution
                    # charge_time = rng.normal(
                    #     avg_charge_time * 1.5, std_charge_time * 0.50
                    # )
                    charge_time = rng.standard_t(3) + avg_charge_time * 1.5
                    charge_time += rng.normal(0, np.sqrt(day))
                # force non-stationarity for oos period
                if day >= np.infty:
                    # make charge time shorter
                    charge_time = rng.normal(avg_charge_time * 0.5, std_charge_time)
                # bound from below by 0
                charge_time = max(0, charge_time)
                # bound from above by 24
                charge_time = min(12, charge_time)
                # round to nearest minute
                charge_time = round(charge_time * no_minutes_per_hour)
                # apply charge time to ev_kw
                ix1 = max(0, j - int(charge_time / 2))
                ix2 = min(total_time, j + int(charge_time / 2))
                ev_kw[i][ix1:ix2] = charge_rate_


def create_distribution(
    cv: CrossValidator, setup: Setup, ev_kw: np.ndarray, is_oos: IS_OOS_Enum
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    NOTE: empirical distribution is created using the in-sample period
          so emprical distribution == in-sample distribution
    returns:
        emp: np.ndarray of shape (cv.no_samples, 24); flexiblity offered by EVs in-sample
        up_flex: np.ndarray of shape (cv.no_samples, 24); up flexibility offered by EVs in-sample
        down_flex: np.ndarray of shape (cv.no_samples, 24); down flexibility offered by EVs in-sample
        NOTE: emp = min(up_flex, down_flex)
        NOTE: up_flex corresponds to (in-sample) charging power
              and np.sum(up_flex, axis=0) = (in-sample) average charging power
        NOTE: total max charge rate = down_flex + up_flex

    (same applies for out-of-sample)
    """
    idx = (
        np.arange(cv.in_sample_start, cv.in_sample_end)
        if is_oos == IS_OOS_Enum.IS
        else np.arange(cv.out_sample_start, cv.out_sample_end)
    )
    shape_assert = (
        cv.in_sample_end - cv.in_sample_start
        if is_oos == IS_OOS_Enum.IS
        else cv.out_sample_end - cv.out_sample_start
    )
    assert idx.shape[0] == shape_assert
    # how to create empirical distribution? Per hourly basis
    hour_of_day_ar = (np.arange(setup.total_time) // setup.no_minutes_per_hour) % 24
    assert ev_kw.shape[1] == hour_of_day_ar.shape[0]
    min_of_day_ar = np.arange(setup.total_time) % (setup.no_minutes_per_hour * 24)
    assert (
        ev_kw.shape[1]
        == min_of_day_ar.shape[0]
        == setup.total_time
        == hour_of_day_ar.shape[0]
    )

    # get aggregate kw
    aggregate_kw_ = np.nansum(ev_kw[: cv.no_ev_samples, idx], axis=0)
    assert (
        aggregate_kw_.shape[0] == shape_assert
    ), f"{aggregate_kw_.shape[0]} != {shape_assert}"
    # get no. of EVs plugged in for hour i
    no_plugged_in = np.nansum(ev_kw[: cv.no_ev_samples, idx] > 0, axis=0)
    assert (
        no_plugged_in.shape[0] == shape_assert
    ), f"{no_plugged_in.shape[0]} != {shape_assert}"
    assert (no_plugged_in <= cv.no_ev_samples).all()

    assert cv.no_ev_samples <= ev_kw.shape[0]
    emp = np.zeros((cv.no_samples, 24 * 60))
    up_flex = np.zeros((cv.no_samples, 24 * 60))
    down_flex = np.zeros((cv.no_samples, 24 * 60))
    # now sample uniformly cv.no_samples values for each hour (and each minute in that hour)
    for i in range(24 * 60):
        # get all values for minute i
        # values = aggregate_kw_[hour_of_day_ar[idx] == i]
        # values2 = no_plugged_in[hour_of_day_ar[idx] == i]
        values = aggregate_kw_[min_of_day_ar[idx] == i]
        values2 = no_plugged_in[min_of_day_ar[idx] == i]
        assert values.shape[0] == values2.shape[0]
        # sample cv.no_samples values
        ix = np.arange(len(values))
        replace = True if len(values) < cv.no_samples else False
        # fmt: off
        if replace and i == 0: print("Sampling with replacement for empirical distribution")
        # fmt: on
        sample_ix = setup.rng.choice(
            ix, size=cv.no_samples, replace=replace, shuffle=False
        )
        # calculate up and down flexibility
        up_flex_ = values[sample_ix]
        down_flex_ = setup.max_charge_rate * values2[sample_ix] - values[sample_ix]
        # add to emp (which is assumed to be symmetric arround flexiblity offered)
        assert (up_flex_ >= 0).all()
        assert (down_flex_ >= 0).all()
        assert down_flex_.shape[0] == cv.no_samples
        assert down_flex_.shape == up_flex_.shape
        # take minimum since we are offering symmetric capacity in, e.g., FCR
        # h_rng = np.arange(i * 60, (i + 1) * 60)
        emp[:, i] = np.minimum(up_flex_, down_flex_)  # .reshape(cv.no_samples, 60)
        up_flex[:, i] = up_flex_  # .reshape(cv.no_samples, 60)
        down_flex[:, i] = down_flex_  # .reshape(cv.no_samples, 60)

    assert (setup.max_charge_rate * cv.no_ev_samples >= emp).all()

    return emp, up_flex, down_flex


@timing()
def create_emprical_distribution(
    cv: CrossValidator, setup: Setup, ev_kw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return create_distribution(cv, setup, ev_kw, IS_OOS_Enum.IS)


@timing()
def create_is_distribution(
    cv: CrossValidator, setup: Setup, ev_kw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return create_emprical_distribution(cv, setup, ev_kw)


@timing()
def create_oos_distribution(
    cv: CrossValidator, setup: Setup, ev_kw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return create_distribution(cv, setup, ev_kw, IS_OOS_Enum.OOS)
