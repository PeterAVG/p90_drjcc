import numpy as np

from src.base import IS_OOS_Enum, Method_Enum, OptimizationInstance, Result


def evaluate(
    inst: OptimizationInstance,
    p_cap_opt: np.ndarray,
    is_oos: IS_OOS_Enum,
    method: Method_Enum,
) -> Result:

    ev_kw = inst.ev_kw
    no_ev_samples = inst.cv.no_ev_samples
    ix1 = (
        inst.cv.out_sample_start
        if is_oos == IS_OOS_Enum.OOS
        else inst.cv.in_sample_start
    )
    ix2 = inst.cv.out_sample_end if is_oos == IS_OOS_Enum.OOS else inst.cv.in_sample_end
    max_charge_rate = inst.setup.max_charge_rate

    # get aggregate kw
    aggregate_kw_ = np.nansum(ev_kw[:no_ev_samples, ix1:ix2], axis=0)
    assert aggregate_kw_.shape[0] == ix2 - ix1

    # get no. of EVs plugged in
    no_plugged_in = np.sum(ev_kw[:no_ev_samples, ix1:ix2] > 0, axis=0)
    assert no_plugged_in.shape[0] == ix2 - ix1
    assert (no_plugged_in <= no_ev_samples).all()

    # calculate up and down flexibility
    up_flex = aggregate_kw_.copy()
    down_flex = max_charge_rate * no_plugged_in - aggregate_kw_
    sym_flex = np.minimum(up_flex, down_flex)
    assert (up_flex >= 0).all()
    assert (down_flex >= 0).all()
    assert down_flex.shape == up_flex.shape

    # get violations
    assert (ix2 - ix1) % 24 == 0
    p_cap = np.tile(np.repeat(p_cap_opt, 60), (ix2 - ix1) // (24 * 60))
    assert p_cap.shape[0] == ix2 - ix1 == sym_flex.shape[0]
    violations = np.maximum(p_cap - sym_flex, 0.0)  # if positive: violation

    # calculate revenue and penalty
    prices_ = np.tile(np.repeat(inst.prices, 60), (ix2 - ix1) // (24 * 60))
    assert prices_.shape[0] == ix2 - ix1 == sym_flex.shape[0] == violations.shape[0]
    penalty = np.sum(violations * prices_ * 2)  # pay back twice the price
    revenue = np.sum(prices_ * p_cap) - penalty

    # create result instance
    result = Result(
        p_cap=p_cap,
        p_realized=aggregate_kw_,
        violation=violations,
        is_oos=is_oos,
        method=method,
        revenue=revenue,
        penalty=penalty,
    )

    print(f"Violation frequency: {round(result.freq, 2)*100}%")
    print(
        f"Violation avg. magnitude: {round(np.mean(result.violation[result.violation > 0]), 2)} kW"
    )

    return result
