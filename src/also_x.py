from dataclasses import dataclass
from typing import Tuple, cast

import numpy as np
from base import OptimizationInstance
from gurobipy import GRB, Model
from utils import timing


@dataclass
class LocalGurobiObject:
    model: Model | None = None

    def get_var_values(self) -> Tuple[np.ndarray, np.ndarray]:
        all_vars = self.model.getVars()  # type: ignore
        values = self.model.getAttr("X", all_vars)  # type: ignore
        names = self.model.getAttr("VarName", all_vars)  # type: ignore
        p_cap_opt = np.array(
            [val for name, val in zip(names, values) if name.startswith("p_cap")]
        )
        y_opt = np.array(
            [val for name, val in zip(names, values) if name.startswith("y")]
        )
        return p_cap_opt, y_opt


@timing()
def run_optimization(
    inst: OptimizationInstance,
    local: LocalGurobiObject,
    q: float,
    show_output: bool = False,
) -> None:

    no_samples = inst.cv.no_samples
    no_minutes_per_hour = inst.setup.no_minutes_per_hour
    no_ev_samples = inst.cv.no_ev_samples
    max_charge_rate = inst.setup.max_charge_rate
    empirical_distribution = inst.empirical_distribution
    prices = inst.prices
    M = inst.also_x.M

    if local.model is None:
        # Create a new model
        print("Creating model...")
        m = Model("also_x")
        m = cast(Model, m)
        # Suppress output
        m.setParam("OutputFlag", 0)  # type: ignore

        # Variables: Number capacity for each hour
        print("Adding variables...")
        m.addVars(  # type: ignore
            24,
            name="p_cap",
            lb=0,
            ub=no_ev_samples * max_charge_rate,
            vtype=GRB.CONTINUOUS,
        )
        m.addVars(  # type: ignore
            no_samples,
            no_minutes_per_hour * 24,
            name="y",
            lb=0,
            ub=1,
            vtype=GRB.CONTINUOUS,
        )
        m.update()

        # Objective: Maximize earnings
        m.setObjective(sum(m.getVarByName(f"p_cap[{h}]") * prices[h] for h in range(24)), GRB.MAXIMIZE)  # type: ignore

        # Constraints
        print("Adding constraints...")
        m.addConstrs(  # type: ignore
            (
                -(1 - m.getVarByName(f"y[{w},{i}]")) * M
                <= m.getVarByName(f"p_cap[{i//60}]") - empirical_distribution[w, i]
                for i in range(no_minutes_per_hour * 24)
                for w in range(no_samples)
            ),
            "jcc1",
        )
        m.addConstrs(  # type: ignore
            (
                m.getVarByName(f"p_cap[{i//60}]") - empirical_distribution[w, i]
                <= m.getVarByName(f"y[{w},{i}]") * M
                for i in range(no_minutes_per_hour * 24)
                for w in range(no_samples)
            ),
            "jcc2",
        )
        m.addConstr(  # type: ignore
            sum(
                m.getVarByName(f"y[{w},{i}]")
                for i in range(no_minutes_per_hour * 24)
                for w in range(no_samples)
            )
            <= q,
            "q_ax",
        )
        local.model = m
    else:
        assert isinstance(local.model, Model)
        # only adjust constraint related to a new q
        local.model.remove(local.model.getConstrByName("q_ax"))  # type: ignore
        local.model.addConstr(  # type: ignore
            sum(
                local.model.getVarByName(f"y[{w},{i}]")
                for i in range(no_minutes_per_hour * 24)
                for w in range(no_samples)
            )
            <= q,
            "q_ax",
        )
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
        elif local.model.status == GRB.INF_OR_UNBD:
            print("INF_OR_UNBD")
        elif local.model.status == GRB.UNBOUNDED:
            print("UNBOUNDED")
        else:
            print("UNKNOWN")
            raise AssertionError

    # print objective value
    print(f"Objective value: {round(local.model.objVal, 1)}")  # type: ignore


@timing()
def run_also_x(opt_instance: OptimizationInstance) -> Tuple[np.ndarray, np.ndarray]:

    # tolerance for the difference between the upper and lower bound
    tol = 1
    # set q_lower and q_upper
    q_lower = 0
    q_upper = opt_instance.also_x.q
    iter = 0
    q: float
    p_cap_opt: np.ndarray
    y_opt: np.ndarray
    local = LocalGurobiObject()

    while abs(q_upper - q_lower) > tol:

        print(f"iteration: {iter}")

        # set q
        q = (q_lower + q_upper) / 2

        print(f"\n\nq_lower: {q_lower}, q_upper: {q_upper}, q: {q}\n\n")

        # run relaxation of MILP
        if iter == 0:
            run_optimization(opt_instance, local, q)
        else:
            run_optimization(opt_instance, local, q)

        p_cap_opt, y_opt = local.get_var_values()

        # get percentage of y_opy that are 0
        y_opt_0 = np.sum(y_opt == 0) / np.size(y_opt)
        print(f"Percentage of y_opt that are 0: {round(y_opt_0, 2)*100}%")
        print(f"P_cap_opt: {p_cap_opt}")

        # adjust q accordingly
        if y_opt_0 >= 1 - opt_instance.epsilon:
            q_lower = q  # type: ignore
        else:
            q_upper = q  # type: ignore

        iter += 1

    assert local.model is not None and iter > 0
    print(f"Final q: {q}")  # type: ignore

    p_cap_opt, y_opt = local.get_var_values()
    return p_cap_opt, y_opt  # type: ignore
