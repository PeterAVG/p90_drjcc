# %%

import numpy as np


def calculate_norms(vector, p=None):
    """
    Calculate the 1-norm, p-norm, and infinity norm of a vector.

    Parameters:
    vector (list or numpy array): The input vector.
    p (int, optional): The value of p for the p-norm. Default is None, which calculates the 2-norm.

    Returns:
    float: The 1-norm of the vector.
    float: The p-norm of the vector (default is 2-norm).
    float: The infinity norm of the vector.
    """
    vector = np.array(
        vector
    )  # Convert the input to a numpy array for easy calculations
    one_norm = np.sum(np.abs(vector))

    if p is None:
        p_norm = np.linalg.norm(vector)  # Default to 2-norm if p is not specified
    else:
        p_norm = np.power(np.sum(np.abs(vector) ** p), 1 / p)

    infinity_norm = np.max(np.abs(vector))

    return one_norm, p_norm, infinity_norm


# Example usage:
vector = [1, -2, 3, -4]
vector = [1, 1, 1, 1]
vector = np.zeros(24 * 60)
vector[0] = 1
one_norm, two_norm, infinity_norm = calculate_norms(vector)
_, ten_norm, _ = calculate_norms(vector, 10)
print("1-Norm:", one_norm)
print("2-Norm:", two_norm)
print("10-Norm:", ten_norm)
print("Infinity Norm:", infinity_norm)
