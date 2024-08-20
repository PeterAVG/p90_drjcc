import math
from dataclasses import dataclass, field
from typing import Tuple, cast

import numpy as np
from gurobipy import GRB, Model

from src.base import IS_OOS_Enum, Method_Enum, OptimizationInstance
from src.evaluate import evaluate
from src.utils import timing


class InfeasibleError(Exception):
    pass


@dataclass
class LocalGurobiObject:

    best_theta: float

    model: Model | None = None

    best_obj: float = np.infty
    best_p_cap_opt: np.ndarray = field(init=False)

    best_t: float | None = None
    best_s: np.ndarray | None = None
    best_q: np.ndarray | None = None
    # best_nu: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.best_p_cap_opt = np.zeros(24)

    def get_var_values(self) -> np.ndarray:
        all_vars = self.model.getVars()  # type: ignore
        values = self.model.getAttr("X", all_vars)  # type: ignore
        names = self.model.getAttr("VarName", all_vars)  # type: ignore
        p_cap_opt = np.array(
            [val for name, val in zip(names, values) if name.startswith("p_cap")]
        )
        return p_cap_opt

    def update_s_t_q(self, no_samples: int) -> None:
        t = self.model.getVarByName("t").x  # type: ignore
        s = np.array([self.model.getVarByName(f"s[{w}]").x for w in range(no_samples)])  # type: ignore
        q = np.array([self.model.getVarByName(f"q[{w}]").x for w in range(no_samples)])  # type: ignore
        # nu = np.array([self.model.getVarByName(f"nu[{w},{i}]").x for w in range(no_samples) for i in range(24 * 60)])  # type: ignore
        self.best_t = t
        self.best_s = s
        self.best_q = q


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
    # M2 = inst.drjcc.M2
    # assert len(M2) == 24

    def add_theta0_constraint(model: Model, theta: float) -> None:
        """
        Add an additional constraint to the model for the case that theta=0
        in order to get the right result.
        """
        if theta == 0 and False:
            print("Adding theta0 constraint...")
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
        m.update()

        # Objective: Maximize earnings
        m.setObjective(sum(m.getVarByName(f"p_cap[{h}]") * prices[h] for h in range(24)), GRB.MAXIMIZE)  # type: ignore

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
        local.model = m
        add_theta0_constraint(local.model, theta)
        local.model.update()  # type: ignore
    else:
        assert isinstance(local.model, Model)
        # only adjust constraint related to a new theta
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


@timing()
def run_drjcc(opt_instance: OptimizationInstance) -> np.ndarray:

    theta_list = opt_instance.drjcc.theta_list
    no_samples = opt_instance.cv.no_samples
    local = LocalGurobiObject(best_theta=theta_list[0])

    # perform grid-search over all thetas
    for theta in theta_list:
        print(f"theta: {theta}")
        try:
            run_optimization(opt_instance, local, theta, show_output=False)
        except InfeasibleError:
            continue
        p_cap_opt_ = local.get_var_values()
        print(f"p_cap_opt: {p_cap_opt_}")

        # check if this is the best solution AND it's not bogus...
        if (
            local.model.objVal < local.best_obj  # type: ignore
            and not any(
                p_cap_opt_
                >= opt_instance.setup.max_charge_rate * opt_instance.cv.no_ev_samples
                - 1
            )
            and not (p_cap_opt_ <= 1).all()
            and not (p_cap_opt_ < 0).any()
        ):
            local.best_p_cap_opt = p_cap_opt_
            local.best_theta = theta
            local.best_obj = local.model.objVal  # type: ignore
            local.update_s_t_q(no_samples)

    if local.best_obj == 0.0:
        print("No solution found for any theta for DRJCC!")

    assert local.model is not None
    print("\n\n")
    print(f"Best theta: {local.best_theta}")
    print(f"Best p_cap_opt: {local.best_p_cap_opt}")
    print(f"Best objective value: {local.best_obj}")
    print(f"Best t: {local.best_t}")  # type: ignore
    print(f"Best s: {local.best_s}")  # type: ignore
    print(f"Best q: {local.best_q}")  # type: ignore

    return local.best_p_cap_opt


def run_three_thetas(opt_instance: OptimizationInstance) -> Tuple[np.ndarray, dict]:

    THETAS = [0.01, 0.1, 0.35]
    placeholder = np.empty((len(THETAS), 24))
    grid_result = {}

    # perform grid-search over all thetas
    for theta in THETAS:
        opt_instance.drjcc.theta_list = [theta]
        p_cap_opt = run_drjcc(opt_instance)
        placeholder[THETAS.index(theta)] = p_cap_opt

        is_result = evaluate(
            opt_instance, p_cap_opt, is_oos=IS_OOS_Enum.IS, method=Method_Enum.DRJCC
        )
        oos_result = evaluate(
            opt_instance, p_cap_opt, is_oos=IS_OOS_Enum.OOS, method=Method_Enum.DRJCC
        )
        grid_result[theta] = {"is_result": is_result, "oos_result": oos_result}

    print(f"Theta results: {placeholder}")
    return placeholder, grid_result
