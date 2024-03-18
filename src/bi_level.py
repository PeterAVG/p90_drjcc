import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, cast

import numpy as np
from base import IS_OOS_Enum, Method_Enum, OptimizationInstance
from evaluate import evaluate
from gurobipy import GRB, Model
from utils import timing

EPSILON = list(round(x, 2) for x in np.arange(0.01, 0.31, 0.01))
# EPSILON = list(round(x, 2) for x in np.arange(0.05, 0.16, 0.01))
# EPSILON = list(round(x, 2) for x in np.arange(0.01, 0.11, 0.01))


class InfeasibleError(Exception):
    pass


@dataclass
class LocalGurobiObject:

    best_epsilon: float
    best_theta: float

    model: Model | None = None

    # Energinet's objective (for the outer problem)
    best_outer_obj: float = -np.infty
    # Aggregator's objective (for the inner problem)
    best_inner_obj: float = -np.infty
    best_p_cap_opt: np.ndarray = field(init=False)

    best_t: float | None = None
    best_s: np.ndarray | None = None
    best_q: np.ndarray | None = None
    best_nu: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.best_p_cap_opt = np.zeros(24)

    def get_var_values(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_vars = self.model.getVars()  # type: ignore
        values = self.model.getAttr("X", all_vars)  # type: ignore
        names = self.model.getAttr("VarName", all_vars)  # type: ignore
        p_cap_opt = np.array(
            [val for name, val in zip(names, values) if name.startswith("p_cap")]
        )
        nu = np.array(
            [val for name, val in zip(names, values) if name.startswith("nu")]
        )
        # get non-negative values of nu
        nu = np.maximum(nu, 0)
        q = np.array([val for name, val in zip(names, values) if name.startswith("q")])
        return p_cap_opt, nu, q

    def update_s_t_q_nu(self, no_samples: int, no_minutes_per_hour: int) -> None:
        t = self.model.getVarByName("t").x  # type: ignore
        s = np.array([self.model.getVarByName(f"s[{w}]").x for w in range(no_samples)])  # type: ignore
        q = np.array([self.model.getVarByName(f"q[{w}]").x for w in range(no_samples)])  # type: ignore
        nu = np.array([self.model.getVarByName(f"nu[{w},{i}]").x for w in range(no_samples) for i in range(24 * no_minutes_per_hour)])  # type: ignore
        nu = np.maximum(nu, 0)
        self.best_t = t
        self.best_s = s
        self.best_q = q
        self.best_nu = nu


@timing()
def run_optimization(
    inst: OptimizationInstance,
    local: LocalGurobiObject,
    theta: float,
    show_output: bool = False,
) -> None:

    no_samples = inst.cv.no_samples
    no_minutes_per_hour = inst.setup.no_minutes_per_hour
    no_ev_samples = inst.cv.no_ev_samples
    max_charge_rate = inst.setup.max_charge_rate
    empirical_distribution = inst.empirical_distribution
    prices = inst.prices
    epsilon = inst.epsilon
    M = inst.drjcc.M

    def add_theta0_constraint(model: Model, theta: float) -> None:
        """
        Add an additional constraint to the model for the case that theta=0
        in order to get the right result.
        """
        if theta == 0 and False:
            model.addConstr(  # type: ignore
                sum(model.getVarByName(f"q[{w}]") for w in range(no_samples))
                <= math.floor(epsilon * no_samples),
                "theta0",
            )

    if local.model is None:
        # Create a new model
        print("Creating model...")
        m = Model("drjcc")
        m = cast(Model, m)
        # Suppress output
        m.setParam("OutputFlag", 0)  # type: ignore

        # Variables: Number capacity for each hour
        print("Adding variables...")
        m.addVars(  # type: ignore
            24,
            name="p_cap",
            # lb=0,
            lb=-GRB.INFINITY,
            ub=no_ev_samples * max_charge_rate,
            vtype=GRB.CONTINUOUS,
        )
        m.addVars(  # type: ignore
            no_samples,
            name="q",
            lb=0,
            ub=1,
            # vtype=GRB.BINARY,
            vtype=GRB.CONTINUOUS,
        )
        m.addVars(  # type: ignore
            no_samples,
            name="s",
            lb=0,
            vtype=GRB.CONTINUOUS,
        )
        m.addVar(  # type: ignore
            lb=-GRB.INFINITY,
            name="t",
            vtype=GRB.CONTINUOUS,
        )
        m.addVars(  # type: ignore
            no_samples,
            24 * no_minutes_per_hour,
            lb=-GRB.INFINITY,
            name="nu",
            vtype=GRB.CONTINUOUS,
        )
        m.update()

        # Inner objective: Maximize earnings - expected violation penalty (which is just paying back the TSO)
        m.setObjective(
            sum(m.getVarByName(f"p_cap[{h}]") * prices[h] for h in range(24)),
            GRB.MAXIMIZE,
        )

        # Constraints
        print("Adding constraints...")
        m.addConstr(  # type: ignore
            epsilon * no_samples * m.getVarByName("t")
            - sum(m.getVarByName(f"s[{w}]") for w in range(no_samples))
            >= theta * no_samples,
            "jcc0",
        )
        m.addConstrs(  # type: ignore
            (
                empirical_distribution[w, i]
                - m.getVarByName(f"p_cap[{i//60}]")
                + M * m.getVarByName(f"q[{w}]")
                >= m.getVarByName("t") - m.getVarByName(f"s[{w}]")
                for i in range(no_minutes_per_hour * 24)
                for w in range(no_samples)
            ),
            "jcc1",
        )
        m.addConstrs(  # type: ignore
            (
                M * (1 - m.getVarByName(f"q[{w}]"))
                >= m.getVarByName("t") - m.getVarByName(f"s[{w}]")
                for w in range(no_samples)
            ),
            "jcc2",
        )
        m.addConstrs(  # type: ignore
            (
                m.getVarByName(f"nu[{w},{i}]")
                == m.getVarByName(f"p_cap[{i//60}]") - empirical_distribution[w, i]
                for w in range(no_samples)
                for i in range(24 * no_minutes_per_hour)
            ),
            "nu_constraint",
        )
        local.model = m
        add_theta0_constraint(local.model, theta)
        local.model.update()  # type: ignore
    else:
        assert isinstance(local.model, Model)
        # only adjust constraint related to a new theta and epsilon
        local.model.remove(local.model.getConstrByName("jcc0"))  # type: ignore
        local.model.addConstr(  # type: ignore
            epsilon * no_samples * local.model.getVarByName("t")
            - sum(local.model.getVarByName(f"s[{w}]") for w in range(no_samples))
            >= theta * no_samples,
            "jcc0",
        )
        add_theta0_constraint(local.model, theta)
        local.model.update()  # type: ignore

    if show_output:
        # Show output
        local.model.setParam("OutputFlag", 1)  # type: ignore

    # Optimize the model
    print("Optimizing...")
    local.model.optimize()  # type: ignore
    print("Done!")

    # assert optimization status
    try:
        assert local.model.status == GRB.OPTIMAL
    except AssertionError:
        # print error message
        print("Optimization status: ", end="")
        if local.model.status == GRB.INFEASIBLE:
            print("INFEASIBLE")
            raise InfeasibleError
        elif local.model.status == GRB.INF_OR_UNBD:
            print("INF_OR_UNBD")
        elif local.model.status == GRB.UNBOUNDED:
            print("UNBOUNDED")
        else:
            print("UNKNOWN")
            raise AssertionError

    # print objective value
    print(f"Objective value: {round(local.model.objVal, 1)}")  # type: ignore


def exponential_function(x: float) -> float:
    return 100 * (2**x - 1)


@timing()
def run_bi_level(
    opt_instance: OptimizationInstance,
) -> Tuple[
    np.ndarray, List[Dict[str, float]], List[Dict[str, float]], LocalGurobiObject
]:

    theta_list = opt_instance.drjcc.theta_list
    no_samples = opt_instance.cv.no_samples
    no_minutes_per_hour = opt_instance.setup.no_minutes_per_hour
    local = LocalGurobiObject(
        best_theta=theta_list[0],
        best_epsilon=EPSILON[0],
    )

    grid_result_is = []
    grid_result_oos = []

    # perform grid-search over all thetas and epsilons
    for epsilon in EPSILON:

        opt_instance.epsilon = epsilon
        assert opt_instance.epsilon == epsilon

        for theta in theta_list:
            print(f"epsilon: {epsilon}, theta: {theta}")

            try:
                run_optimization(opt_instance, local, theta, show_output=False)
            except InfeasibleError:
                continue

            p_cap_opt_, nu, q = local.get_var_values()
            nu_sum = nu.reshape(no_samples, 24, -1).sum(axis=2).mean(axis=0).sum() / 60
            penalty = exponential_function(sum(q) / no_samples)
            outer_obj = sum(p_cap_opt_) - nu_sum * penalty
            print(f"nu: {nu_sum}, outer obj: {outer_obj}")

            # save result if it's not bogus
            if (
                not any(
                    p_cap_opt_
                    >= opt_instance.setup.max_charge_rate
                    * opt_instance.cv.no_ev_samples
                    - 1
                )
                and not (p_cap_opt_ <= 1).all()
                and not (p_cap_opt_ < 0).any()
            ):
                grid_result_is.append(
                    {
                        "theta": theta,
                        "epsilon": epsilon,
                        "outer_obj": outer_obj,
                        "p_cap_opt": p_cap_opt_,
                        "nu": nu_sum,
                        "penalty": penalty,
                    }
                )
                # get corresponding OOS result
                oos_result = evaluate(
                    opt_instance,
                    p_cap_opt_,
                    is_oos=IS_OOS_Enum.OOS,
                    method=Method_Enum.ALSO_X,
                )
                nu_sum_oos = (
                    oos_result.violation.reshape(-1, 24, 60)
                    .sum(axis=2)
                    .mean(axis=0)
                    .sum()
                    / 60
                )
                violation_frequency = oos_result.freq
                penalty_oos = exponential_function(violation_frequency)
                outer_obj_oos = sum(p_cap_opt_) - nu_sum_oos * penalty_oos
                outer_obj_oos = max(outer_obj_oos, 0)
                grid_result_oos.append(
                    {
                        "theta": theta,
                        "epsilon": epsilon,
                        "outer_obj": outer_obj_oos,
                        "p_cap_opt": p_cap_opt_,
                        "nu": nu_sum_oos,
                        "penalty": penalty_oos,
                    }
                )

            # check if this is the best solution AND it's not bogus...
            if (
                outer_obj > local.best_outer_obj
                and not any(
                    p_cap_opt_
                    >= opt_instance.setup.max_charge_rate
                    * opt_instance.cv.no_ev_samples
                    - 1
                )
                and not (p_cap_opt_ <= 1).all()
                and not (p_cap_opt_ < 0).any()
            ):
                print("New best solution found!")
                local.best_p_cap_opt = p_cap_opt_
                local.best_theta = theta
                local.best_epsilon = epsilon
                local.best_outer_obj = outer_obj
                local.best_inner_obj = local.model.objVal  # type: ignore
                local.update_s_t_q_nu(no_samples, no_minutes_per_hour)

    if local.best_outer_obj == -np.infty:
        print("No solution found for any theta for DRJCC!")

    assert local.model is not None
    print("\n\n")
    print(f"Best theta: {local.best_theta}")
    print(f"Best epsilon: {local.best_epsilon}")
    print(f"Best outer objective value: {local.best_outer_obj}")
    print(f"Best inner objective value: {local.best_inner_obj}")
    print(f"Best p_cap_opt: {local.best_p_cap_opt}")

    print(f"Best t: {local.best_t}")  # type: ignore
    print(f"Best s: {local.best_s}")  # type: ignore
    print(f"Best q: {local.best_q}")  # type: ignore
    print(f"Best nu: {round(sum(local.best_nu) / 60 / no_samples,1)}")  # type: ignore

    return local.best_p_cap_opt, grid_result_is, grid_result_oos, local
