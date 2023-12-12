#%% # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from also_x import run_also_x
from base import CrossValidator, IS_OOS_Enum, Method_Enum, OptimizationInstance, Setup
from cvar import run_cvar
from data_generator import (
    compute_ev_consumption,
    create_emprical_distribution,
    create_oos_distribution,
    draw_charging_events,
    generate_probability_vector,
)
from drjcc import run_drjcc
from evaluate import evaluate

P90_EPSILON = 0.10

setup = Setup(
    rng=np.random.default_rng(0),  # random number generator
    # simulation parameters
    no_ev=200,
    no_days=90,
    no_hours_per_day=24,
    no_minutes_per_hour=60,
    charge_rate=6.0,  # kw
    max_charge_rate=11.0,  # kw
    avg_charge_time=8.0,  # hours --> 4 hours to charge 44 kwh at 11 kw
    std_charge_time=2.0,  # hours, big variation in charge time
    prob_of_no_charging=0.10,  # percentage of days where no charging is allowed
)

prob = generate_probability_vector(setup.rng)
charging_events = draw_charging_events(setup.rng, setup, prob).astype(np.float64)

ev_kw = np.zeros((setup.no_ev, setup.total_time), dtype=np.float64)
compute_ev_consumption(
    setup.no_ev,
    setup.no_minutes_per_hour,
    setup.no_hours_per_day,
    setup.total_time,
    setup.charge_rate,
    setup.max_charge_rate,
    setup.avg_charge_time,
    setup.std_charge_time,
    charging_events,
    ev_kw,
)
assert (ev_kw[~np.isnan(ev_kw)] >= 0).all()
assert OptimizationInstance.get_aggregate_kw(ev_kw).shape[0] == setup.total_time

s = 15  # days
cv = CrossValidator(
    no_samples=30,  # TODO: maybe use literature to get this number
    no_ev_samples=20,  # no. of EVs in portfolio
    in_sample_start=1 * 24 * 60,  # 1 day
    in_sample_end=s * 24 * 60,  # s days
    out_sample_start=s * 24 * 60,
    out_sample_end=setup.total_time,
)

emp, up_flex, down_flex = create_emprical_distribution(cv, setup, ev_kw)
emp_oos, up_flex_oos, down_flex_oos = create_oos_distribution(cv, setup, ev_kw)

opt_instance = OptimizationInstance(
    setup=setup,
    epsilon=P90_EPSILON,
    no_samples=cv.no_samples,
    prices=np.ones(24),
    ev_kw=ev_kw,
    empirical_distribution=emp,
    cv=cv,
)

if True:
    ### ALSO-X ###
    p_cap_opt, y_opt = run_also_x(opt_instance)
    is_result = evaluate(
        opt_instance, p_cap_opt, is_oos=IS_OOS_Enum.IS, method=Method_Enum.ALSO_X
    )
    oos_result = evaluate(
        opt_instance, p_cap_opt, is_oos=IS_OOS_Enum.OOS, method=Method_Enum.ALSO_X
    )
    print(f"Violation frequency OOS: {round(oos_result.freq, 2)*100}%")

    ### DRJCC ###
    p_cap_opt2 = run_drjcc(opt_instance)
    is_result2 = evaluate(
        opt_instance, p_cap_opt2, is_oos=IS_OOS_Enum.IS, method=Method_Enum.DRJCC
    )
    oos_result2 = evaluate(
        opt_instance, p_cap_opt2, is_oos=IS_OOS_Enum.OOS, method=Method_Enum.DRJCC
    )
    print(f"Violation frequency OOS: {round(oos_result2.freq, 2)*100}%")

    ### CVaR ###
    p_cap_opt3 = run_cvar(opt_instance)
    is_result3 = evaluate(
        opt_instance, p_cap_opt3, is_oos=IS_OOS_Enum.IS, method=Method_Enum.CVAR
    )
    oos_result3 = evaluate(
        opt_instance, p_cap_opt3, is_oos=IS_OOS_Enum.OOS, method=Method_Enum.CVAR
    )
    print(f"Violation frequency OOS: {round(oos_result3.freq, 2)*100}%")

#%% # noqa

# Plotting
x = np.arange(setup.total_time) / (setup.no_hours_per_day * setup.no_minutes_per_hour)
# x = np.arange(total_time) / (no_minutes_per_hour)
plt.figure(figsize=(10, 6))
plt.plot(x, opt_instance.aggregate_kw)
# highligt in-sample and out-of-sample periods
plt.axvspan(
    0,
    cv.in_sample_start / (setup.no_hours_per_day * setup.no_minutes_per_hour),
    alpha=0.2,
    color="grey",
    label="burn-in",
)
plt.axvspan(
    cv.in_sample_start / (setup.no_hours_per_day * setup.no_minutes_per_hour),
    cv.in_sample_end / (setup.no_hours_per_day * setup.no_minutes_per_hour),
    alpha=0.2,
    color="green",
    label="in-sample",
)
plt.axvspan(
    cv.out_sample_start / (setup.no_hours_per_day * setup.no_minutes_per_hour),
    cv.out_sample_end / (setup.no_hours_per_day * setup.no_minutes_per_hour),
    alpha=0.2,
    color="orange",
    label="out-of-sample",
)
plt.xlabel("Days")
plt.ylabel("kW")
plt.legend()
plt.title("EV Charging Simulation")
plt.xlim(0, setup.no_days)
plt.grid(True)
plt.show()

# create a plot that shows the average consumption for each hour (LINEPLOT)
f, ax = plt.subplots(1, 2, figsize=(12, 6))
for i, _up_flex, _ in zip(range(2), [up_flex, up_flex_oos], [down_flex, down_flex_oos]):
    x = np.arange(_up_flex.shape[1]) / 60
    prfx = "IS" if i == 0 else "OOS"
    ax[i].plot(x, np.nanmean(_up_flex, axis=0), label="mean")
    # plot area between 10% and 90%
    ax[i].fill_between(
        x=x,
        y1=np.nanpercentile(_up_flex, 10, axis=0),
        y2=np.nanpercentile(_up_flex, 90, axis=0),
        alpha=0.2,
        label=f"{prfx} 10-90%",
    )
    ax[i].set_xlabel("Hour of Day")
    ax[i].set_ylabel("kW")
    ax[i].set_title(f"{prfx} distribution of EV Charging")
    ax[i].grid(True)
    ax[i].set_xlim(0, 23)
    ax[i].legend()
ylim = max(
    max(np.nanpercentile(up_flex, 90, axis=0)),
    max(np.nanpercentile(up_flex_oos, 90, axis=0)),
)
ax[0].set_ylim(0, ylim * 1.1)
ax[1].set_ylim(0, ylim * 1.1)

#%% # noqa

# create a plot that shows the mean empirical distribution for each hour with 10/90 percentiles
f, ax = plt.subplots(1, 2, figsize=(12, 6))
for i, _emp in zip(range(2), [emp, emp_oos]):
    prfx = "IS" if i == 0 else "OOS"
    x = np.arange(_emp.shape[1]) / 60
    ax[i].plot(x, np.nanmean(_emp, axis=0), label="mean")
    # plot area between 10% and 90%
    ax[i].fill_between(
        x=x,
        y1=np.nanpercentile(_emp, 10, axis=0),
        y2=np.nanpercentile(_emp, 90, axis=0),
        alpha=0.2,
        label=f"{prfx} 10-90%",
    )
    # plot p_cap_opt
    ax[i].plot(p_cap_opt, label="p_cap_opt ALSO-X", color="green", linestyle="--")
    ax[i].plot(p_cap_opt2, label="p_cap_opt DRJCC", color="blue", linestyle="--")
    ax[i].plot(p_cap_opt3, label="p_cap_opt CVaR", color="orange", linestyle="--")
    ax[i].set_xlabel("Hour of Day")
    ax[i].set_ylabel("kW")
    ax[i].set_title(f"{prfx} distribution of flexibility")
    ax[i].grid(True)
    ax[i].set_xlim(0, 23)
    ax[i].legend()
ylim = max(
    max(np.nanpercentile(up_flex, 90, axis=0)),
    max(np.nanpercentile(up_flex_oos, 90, axis=0)),
)
ax[0].set_ylim(0, ylim * 1.1)
ax[1].set_ylim(0, ylim * 1.1)


# create (3,4) table using Pandas showing revenue and penalty for each method, IS and OOS
is_days = cv.no_is_days(setup)
oos_days = cv.no_oos_days(setup)
print(
    pd.DataFrame(
        [
            {
                "method": result.method.value,
                "is": "Yes" if result.is_oos == IS_OOS_Enum.IS else "No",
                "oos": "Yes" if result.is_oos == IS_OOS_Enum.OOS else "No",
                "revenue [DKK/day]": round(result.revenue / is_days, 2)
                if result.is_oos == IS_OOS_Enum.IS
                else round(result.revenue / oos_days, 2),
                "penalty [DKK/day]": round(result.penalty / is_days, 2)
                if result.is_oos == IS_OOS_Enum.IS
                else round(result.penalty / oos_days, 2),
            }
            for result in [
                is_result,
                oos_result,
                is_result2,
                oos_result2,
                is_result3,
                oos_result3,
            ]
        ]
    )
)

# create a histogram that shows the distribution of OOS violation frequencies
f, ax = plt.subplots(3, 2, figsize=(12, 15))
ax = ax.flatten()
for i, result in enumerate([oos_result, oos_result2, oos_result3]):
    match i:
        case 0:
            method = "ALSO-X"
            i1 = 0
            color = "green"
        case 1:
            method = "DRJCC"
            i1 = 2
            color = "blue"
        case 2:
            method = "CVaR"
            i1 = 4
            color = "orange"
        case _:
            raise ValueError(f"Invalid value for i: {i}")
    assert i >= 0 and i <= 2
    counts, bins, patches = ax[i1].hist(
        result.get_freq_distribution(), bins=20, edgecolor="black"
    )
    freq_oos = np.mean(result.get_freq_distribution())
    ax[i1].vlines(
        freq_oos,
        ymin=0,
        ymax=max(counts),
        color=color,
        label=f"{method}: {round(freq_oos*100,2)}%",
        linestyle="--",
    )
    ax[i1].vlines(
        P90_EPSILON,
        ymin=0,
        ymax=max(counts),
        color="black",
        label="P90: 10%",
        linestyle="--",
        alpha=0.5,
    )
    ax[i1].vlines(
        P90_EPSILON + 0.05,
        ymin=0,
        ymax=max(counts),
        color="red",
        label="P90+buffer: 15%",
        linestyle="--",
    )
    ax[i1].set_xlabel("Within-hour violation frequency")
    ax[i1].set_ylabel("Frequency")
    # ax[i1].set_title("Frequency distribution of violations")
    ax[i1].legend()
    mean_vio = np.mean(result.get_violation_distribution())
    counts, bins, patches = ax[i1 + 1].hist(
        result.get_violation_distribution(), bins=30, edgecolor="black"
    )
    ax[i1 + 1].vlines(
        mean_vio,
        ymin=0,
        ymax=max(counts),
        color=color,
        label=f"{method}: {round(mean_vio,2)} kW",
        linestyle="--",
    )
    ax[i1 + 1].legend()
    ax[i1 + 1].set_xlabel("Magnitude of violations [kW]")
    ax[i1 + 1].set_ylabel("Minutes")
    # ax[i1 + 1].set_title("Magnitude distribution of violations")

ul = max(
    np.max(oos_result.get_freq_distribution()),
    np.max(oos_result2.get_freq_distribution()),
)
ul = max(ul, np.max(oos_result3.get_freq_distribution()))
ax[0].set_xlim(0, ul)
ax[2].set_xlim(0, ul)
ax[4].set_xlim(0, ul)
ul = max(
    np.max(oos_result.get_violation_distribution()),
    np.max(oos_result2.get_violation_distribution()),
)
ul = max(ul, np.max(oos_result3.get_violation_distribution()))
ax[1].set_xlim(0, ul)
ax[3].set_xlim(0, ul)
ax[5].set_xlim(0, ul)
plt.show()
