from dataclasses import dataclass
from typing import cast

import numpy as np
from base import OptimizationInstance
from gurobipy import GRB, Model
from utils import timing


@dataclass
class LocalGurobiObject:
    model: Model | None = None

    def get_var_values(self) -> np.ndarray:
        all_vars = self.model.getVars()  # type: ignore
        values = self.model.getAttr("X", all_vars)  # type: ignore
        names = self.model.getAttr("VarName", all_vars)  # type: ignore
        p_cap_opt = np.array(
            [val for name, val in zip(names, values) if name.startswith("p_cap")]
        )
        return p_cap_opt


@timing()
def run_optimization(
    inst: OptimizationInstance,
    local: LocalGurobiObject,
    show_output: bool = False,
) -> None:

    no_samples = inst.cv.no_samples
    no_minutes_per_hour = inst.setup.no_minutes_per_hour
    no_ev_samples = inst.cv.no_ev_samples
    max_charge_rate = inst.setup.max_charge_rate
    empirical_distribution = inst.empirical_distribution
    prices = inst.prices
    epsilon = inst.epsilon

    if local.model is None:
        # Create a new model
        print("Creating model...")
        m = Model("cvar")
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
            lb=-GRB.INFINITY,
            name="zeta",
            vtype=GRB.CONTINUOUS,
        )
        m.addVar(  # type: ignore
            name="beta",
            lb=-GRB.INFINITY,
            ub=0,
            vtype=GRB.CONTINUOUS,
        )
        m.update()

        # Objective: Maximize earnings
        m.setObjective(sum(m.getVarByName(f"p_cap[{h}]") * prices[h] for h in range(24)), GRB.MAXIMIZE)  # type: ignore

        # Constraints
        m.addConstrs(  # type: ignore
            (
                m.getVarByName(f"p_cap[{i//60}]") - empirical_distribution[w, i]
                <= m.getVarByName(f"zeta[{w},{i}]")
                for i in range(no_minutes_per_hour * 24)
                for w in range(no_samples)
            ),
            "jcc1",
        )
        m.addConstr(  # type: ignore
            (
                sum(
                    m.getVarByName(f"zeta[{w},{i}]")
                    for w in range(no_samples)
                    for i in range(no_minutes_per_hour * 24)
                )
                / (no_samples * no_minutes_per_hour * 24)
                - (1 - epsilon) * m.getVarByName("beta")
                <= 0
            ),
            "jcc2",
        )
        m.addConstrs(  # type: ignore
            (
                m.getVarByName("beta") <= m.getVarByName(f"zeta[{w},{i}]")
                for i in range(no_minutes_per_hour * 24)
                for w in range(no_samples)
            ),
            "jcc3",
        )
        local.model = m
    else:
        raise ValueError("Model already exists")

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
def run_cvar(opt_instance: OptimizationInstance) -> np.ndarray:

    local = LocalGurobiObject()
    run_optimization(opt_instance, local)

    p_cap_opt = local.get_var_values()

    # get percentage of y_opy that are 0
    print(f"P_cap_opt: {p_cap_opt}")
    assert local.model is not None

    p_cap_opt = local.get_var_values()
    return p_cap_opt  # type: ignore
