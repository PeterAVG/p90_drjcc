#%% # noqa
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from also_x import run_also_x
from base import CrossValidator, IS_OOS_Enum, Method_Enum, OptimizationInstance, Setup
from bi_level import EPSILON, run_bi_level
from cvar import run_cvar
from data_generator import (
    compute_ev_consumption,
    create_emprical_distribution,
    create_oos_distribution,
    draw_charging_events,
    generate_probability_vector,
)
from drjcc import run_drjcc, run_three_thetas
from evaluate import evaluate
from utils import _set_font_size

SAVE_PLOT = True
P90_EPSILON = 0.10

RUN_ALL_THREE = False
RUN_THREE_THETA_RESULTS = False
RUN_BI_LEVEL = True

p_cap_opt, p_cap_opt2, p_cap_opt3, p_cap_opt4, p_cap_opt5 = None, None, None, None, None  # type: ignore
is_result, is_result2, is_result3, is_result4, is_result5 = None, None, None, None, None  # type: ignore
oos_result, oos_result2, oos_result3, oos_result4, oos_result5 = None, None, None, None, None  # type: ignore
grid_result = None  # type: ignore

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

if RUN_ALL_THREE:
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


if RUN_BI_LEVEL:
    ### Bi-level optimizatino problem using DRJCC ###
    p_cap_opt4, grid_result, grid_result_oos, _ = run_bi_level(opt_instance)
    # is_result4 = evaluate(
    #     opt_instance, p_cap_opt4, is_oos=IS_OOS_Enum.IS, method=Method_Enum.CVAR
    # )
    # oos_result4 = evaluate(
    #     opt_instance, p_cap_opt4, is_oos=IS_OOS_Enum.OOS, method=Method_Enum.CVAR
    # )
    # print(f"Violation frequency OOS: {round(oos_result4.freq, 2)*100}%")

if RUN_THREE_THETA_RESULTS:
    ### DRJCC for three thetas to show as a result ###
    p_cap_opt5, grid_result = run_three_thetas(opt_instance)

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
# plt.title("EV Charging Simulation")
plt.xlim(0, setup.no_days)
plt.grid(True)
ax = plt.gca()
_set_font_size(ax, misc=16, legend=16)
plt.tight_layout()
# NTOE: tikz takes up way too much memory with this plot...
# tikzplotlib.save("tex/figures/drjcc_raw.tikz")
if SAVE_PLOT:
    plt.savefig("tex/figures/drjcc_raw.png", dpi=300)

# create a plot that shows the average consumption for each hour (LINEPLOT)
f, ax = plt.subplots(1, 2, figsize=(12, 4))
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
        label=f"{prfx} 10-90\%",  # type: ignore # noqa
    )
    ax[i].set_xlabel("Hour of Day")
    if i == 0:
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

_set_font_size(ax, misc=16, legend=14)
plt.tight_layout()

if SAVE_PLOT:
    tikzplotlib.save("tex/figures/drjcc_is_oos_flex.tikz", axis_width="0.49\\textwidth")
    # save as png
    plt.savefig("tex/figures/drjcc_is_oos_flex.png", dpi=300)
plt.show()

#%% # noqa

# create a plot that shows the mean empirical distribution for each hour with 10/90 percentiles
f, ax = plt.subplots(1, 2, figsize=(12, 6))
for i, _emp in zip(range(2), [emp, emp_oos]):
    prfx = "IS" if i == 0 else "OOS"
    x = np.arange(_emp.shape[1]) / 60
    ax[i].plot(x, np.nanmean(_emp, axis=0))
    # plot area between 10% and 90%
    ax[i].fill_between(
        x=x,
        y1=np.nanpercentile(_emp, 10, axis=0),
        y2=np.nanpercentile(_emp, 90, axis=0),
        alpha=0.2,
        # label=f"{prfx} 10-90\%",
    )
    # plot p_cap_opt
    # write p_cap_opt as latex math
    tx = r"$p^{\text{cap}}$"
    if RUN_ALL_THREE:
        ax[i].plot(p_cap_opt, label="ALSO-X", color="green", linestyle="--")
        ax[i].plot(p_cap_opt2, label="DRJCC", color="blue", linestyle="--")
        ax[i].plot(p_cap_opt3, label="CVaR", color="orange", linestyle="--")
    if RUN_BI_LEVEL and False:
        ax[i].plot(p_cap_opt4, label="Bi-level", color="red", linestyle="--")
    ax[i].set_xlabel("Hour of Day")
    if i == 0:
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

_set_font_size(ax, misc=16, legend=14)
plt.tight_layout()

if SAVE_PLOT:
    tikzplotlib.save("tex/figures/drjcc_bids.tikz", axis_width="0.49\\textwidth")
    # also save as png
    plt.savefig("tex/figures/drjcc_bids.png", dpi=300)

#%% # noqa

# create a plot that shows the mean empirical distribution for each hour with 10/90 percentiles
f, ax = plt.subplots(1, 2, figsize=(12, 6))
for i, _emp in zip(range(2), [emp, emp_oos]):
    prfx = "IS" if i == 0 else "OOS"
    x = np.arange(_emp.shape[1]) / 60
    ax[i].plot(x, np.nanmean(_emp, axis=0))
    # plot area between 10% and 90%
    ax[i].fill_between(
        x=x,
        y1=np.nanpercentile(_emp, 10, axis=0),
        y2=np.nanpercentile(_emp, 90, axis=0),
        alpha=0.2,
        # label=f"{prfx} 10-90\%",
    )
    # plot p_cap_opt
    # write p_cap_opt as latex math
    tx = r"$p^{\text{cap}}$"
    if RUN_THREE_THETA_RESULTS:
        for q, (theta, val) in enumerate(grid_result.items()):  # type: ignore
            theta_tex = r"$\theta$" + "=" + str(theta)
            p_cap_opt = p_cap_opt5[q, :]  # type: ignore
            ax[i].plot(p_cap_opt, label=theta_tex, linestyle="--")
    ax[i].set_xlabel("Hour of Day")
    if i == 0:
        ax[i].set_ylabel("kW")
    # ax[i].set_title(f"{prfx} distribution of flexibility")
    ax[i].grid(True)
    ax[i].set_xlim(0, 23)
    ax[i].legend()
ylim = max(
    max(np.nanpercentile(up_flex, 90, axis=0)),
    max(np.nanpercentile(up_flex_oos, 90, axis=0)),
)
ax[0].set_ylim(0, ylim * 1.1)
ax[1].set_ylim(0, ylim * 1.1)

_set_font_size(ax, misc=16, legend=16)
plt.tight_layout()

if SAVE_PLOT:
    tikzplotlib.save("tex/figures/drjcc_bids_paper.tikz", axis_width="0.49\\textwidth")
    # also save as png
    plt.savefig("tex/figures/drjcc_bids_paper.png", dpi=300)

#%% # noqa
# create (3,4) table using Pandas showing revenue and penalty for each method, IS and OOS
# is_days = cv.no_is_days(setup)
# oos_days = cv.no_oos_days(setup)
# print(
#     pd.DataFrame(
#         [
#             {
#                 "method": result.method.value,
#                 "is": "Yes" if result.is_oos == IS_OOS_Enum.IS else "No",
#                 "oos": "Yes" if result.is_oos == IS_OOS_Enum.OOS else "No",
#                 "revenue [DKK/day]": round(result.revenue / is_days, 2)
#                 if result.is_oos == IS_OOS_Enum.IS
#                 else round(result.revenue / oos_days, 2),
#                 "penalty [DKK/day]": round(result.penalty / is_days, 2)
#                 if result.is_oos == IS_OOS_Enum.IS
#                 else round(result.penalty / oos_days, 2),
#             }
#             for result in [
#                 is_result,
#                 oos_result,
#                 is_result2,
#                 oos_result2,
#                 is_result3,
#                 oos_result3,
#             ]
#         ]
#     )
# )

plt.close()
if True:
    import tikzplotlib

assert oos_result is not None
assert oos_result2 is not None
assert oos_result3 is not None

ul1 = max(
    np.max(oos_result.get_freq_distribution()),
    np.max(oos_result2.get_freq_distribution()),
)
ul1 = max(ul1, np.max(oos_result3.get_freq_distribution()))
ul2 = max(
    np.max(oos_result.get_violation_distribution()),
    np.max(oos_result2.get_violation_distribution()),
)
ul2 = max(ul2, np.max(oos_result3.get_violation_distribution()))

# create a histogram that shows the distribution of OOS violation frequencies
for i, result in enumerate([oos_result, oos_result2, oos_result3]):
    f, ax = plt.subplots(1, 2, figsize=(12, 4))
    # ax = ax.flatten()
    match i:
        case 0:
            method = "ALSO-X"
            i1 = 0
            color = "green"
        case 1:
            method = "DRJCC"
            i1 = 0
            color = "blue"
        case 2:
            method = "CVaR"
            i1 = 0
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
        label=f"{method}: {round(freq_oos*100,2)}" + "%",
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
    ax[i1 + 1].set_xlabel("Magnitude of violations [kW]")
    ax[i1 + 1].set_ylabel("Minutes")
    # ax[i1 + 1].set_title("Magnitude distribution of violations")

    ax[i1].set_xlim(0, ul1)
    ax[i1].set_xlim(0, ul1)
    ax[i1].set_xlim(0, ul1)

    ax[i1 + 1].set_xlim(0, ul2)
    ax[i1 + 1].set_xlim(0, ul2)
    ax[i1 + 1].set_xlim(0, ul2)

    # Position the y-axis label on the right
    ax[i1 + 1].yaxis.set_label_position("right")
    # Move the y-axis ticks to the right
    ax[i1 + 1].yaxis.tick_right()

    ax[i1].legend()
    ax[i1 + 1].legend()

    # rotate xticks
    for ax_ in [ax[i1], ax[i1 + 1]]:
        plt.sca(ax_)
        plt.xticks(rotation=45)

    _set_font_size(ax[i1], misc=16, legend=14)
    _set_font_size(ax[i1 + 1], misc=16, legend=14)

    # tight
    plt.tight_layout()

    if SAVE_PLOT:
        # NOTE: tikzplotlib can't convert legend as \draw commands does not have \addlegendentry
        # tikzplotlib.save(
        #     f"tex/figures/drjcc_oos_histograms_{method}.tikz",
        #     axis_width="0.49\\textwidth",
        # )
        plt.savefig(f"tex/figures/drjcc_oos_histograms_{method}.png", dpi=300)

plt.show()

#%% # noqa
assert grid_result is not None
assert grid_result_oos is not None


epsilon_length = len(EPSILON)
theta_length = len(opt_instance.drjcc.theta_list)
matrix = np.full((epsilon_length, theta_length), np.nan)
# for ele in grid_result_oos:
for ele in grid_result:
    ep = ele["epsilon"]
    th = ele["theta"]
    assert ep in EPSILON
    assert th in opt_instance.drjcc.theta_list
    # get index of epsilon in EPSILON
    ep_ix = np.where(EPSILON == ep)[0][0]
    # get index of theta in theta_list
    th_ix = opt_instance.drjcc.theta_list.index(th)
    # save result in matrix
    matrix[ep_ix, th_ix] = ele["outer_obj"]

print(matrix.shape)
print(np.nanmax(matrix))
# get row index and col index for max value
row_ix, col_ix = np.where(matrix == np.nanmax(matrix))
assert np.isclose(np.nanmax(matrix), matrix[row_ix, col_ix])
print(f"Optimal epsilon: {EPSILON[row_ix[0]]}")
print(f"Optimal theta: {opt_instance.drjcc.theta_list[col_ix[0]]}")

# plot a heatmap with epsilon on x-axis, theta on y-axis and outer_objective on z-axis
f, ax = plt.subplots(1, 1, figsize=(6, 6))
# use a colormap where dark is higher and light is lower
cmap = plt.cm.get_cmap("viridis_r")  # type: ignore
ax.imshow(matrix, cmap=cmap, interpolation="nearest")
# colorbar
cbar = ax.figure.colorbar(ax.imshow(matrix, cmap=cmap, interpolation="nearest"))
cbar.ax.set_ylabel("Available flexibility [kW]", rotation=-90, va="bottom")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$\epsilon$")
ax.set_xticks(np.arange(theta_length))
ax.set_yticks(np.arange(epsilon_length))
ax.set_xticklabels(opt_instance.drjcc.theta_list)
ax.set_yticklabels(EPSILON)
for item in [
    ax.xaxis.label,
    ax.yaxis.label,
]:  # + ax.get_xticklabels() + ax.get_yticklabels()
    item.set_fontsize(14)
# rotate xticks
plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
plt.tight_layout()

# Interpretation of heatmap:
# Low theta and high epsilon gives very low avaialble flexibility (obj value for TSO)
# due to exponential penalty function in bi_level.py

if SAVE_PLOT:
    # tikz conversion does not work for some reason
    # tikzplotlib.save(
    #     "tex/figures/heatmap.tikz",
    #     axis_width="0.49\\textwidth",
    # )
    plt.savefig("tex/figures/heatmap.png", dpi=300, bbox_inches="tight")

# plt.show()
