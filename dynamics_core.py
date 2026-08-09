"""
Step 1 & 2: payoff matrix with antisymmetry check, and the interior
fixed point x* solving M @ x* = 0.
"""

import numpy as np
from scipy.linalg import null_space


def payoff_matrix(strategies: np.ndarray) -> np.ndarray:
    k = strategies.shape[0]
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            M[i, j] = np.sum(strategies[i] > strategies[j]) - np.sum(strategies[i] < strategies[j])
    return M


def assert_valid_payoff_matrix(M: np.ndarray, tol: float = 1e-9):
    if not np.allclose(M, -M.T, atol=tol):
        raise AssertionError("Payoff matrix is not antisymmetric.")
    if not np.allclose(np.diag(M), 0, atol=tol):
        raise AssertionError("Diagonal of payoff matrix is not zero.")


def find_interior_fixed_point(M: np.ndarray) -> np.ndarray | None:
    ns = null_space(M)
    if ns.shape[1] != 1:
        print(f"Expected a 1-dimensional null space, got {ns.shape[1]}.")
        return None
    v = ns[:, 0]
    if np.all(v < 0):
        v = -v
    if not np.all(v > 1e-9):
        print("Null vector has non-positive entries -- no interior fixed point exists.")
        return None
    return v / v.sum()


def V(x: np.ndarray, x_star: np.ndarray) -> float:
    """Conserved quantity along replicator trajectories, for zero-sum M."""
    return sum(x_star[i] * np.log(x[i] / x_star[i]) for i in range(len(x_star)))


def replicator_velocity(x: np.ndarray, M: np.ndarray) -> np.ndarray:
    """dx/dt = x * (M @ x), elementwise."""
    return x * (M @ x)


def step_euler(x: np.ndarray, M: np.ndarray, eta: float) -> np.ndarray:
    x_new = x + eta * replicator_velocity(x, M)
    return x_new / x_new.sum()


def step_rk4(x: np.ndarray, M: np.ndarray, eta: float) -> np.ndarray:
    k1 = replicator_velocity(x, M)
    k2 = replicator_velocity(x + 0.5 * eta * k1, M)
    k3 = replicator_velocity(x + 0.5 * eta * k2, M)
    k4 = replicator_velocity(x + eta * k3, M)
    x_new = x + (eta / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_new / x_new.sum()


if __name__ == "__main__":
    # Quick test using 3-strategy rock-paper-scissors, where we KNOW
    # the answer analytically: x_star should come out to (1/3, 1/3, 1/3).
    M_rps = np.array([
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0],
    ], dtype=float)

    assert_valid_payoff_matrix(M_rps)
    print("Antisymmetry check passed.")

    x_star = find_interior_fixed_point(M_rps)
    print("Fixed point found:", x_star)
    expected = np.array([1 / 3, 1 / 3, 1 / 3])
    if x_star is not None and np.allclose(x_star, expected, atol=1e-6):
        print("Matches the known analytical answer (1/3, 1/3, 1/3). Good.")
    else:
        print("Doesn't match -- something's off, debug before moving on.")

    # Compare Euler vs RK4 drift in the conserved quantity V
    rng = np.random.default_rng(0)
    x0 = x_star + rng.normal(0, 0.02, size=3)
    x0 = np.clip(x0, 1e-6, None)
    x0 = x0 / x0.sum()

    eta = 0.05
    steps = 500

    x_e = x0.copy()
    for _ in range(steps):
        x_e = step_euler(x_e, M_rps, eta)

    x_r = x0.copy()
    for _ in range(steps):
        x_r = step_rk4(x_r, M_rps, eta)

    print("\nStarting V:", V(x0, x_star))
    print("V after Euler,", steps, "steps:", V(x_e, x_star))
    print("V after RK4,  ", steps, "steps:", V(x_r, x_star)) 