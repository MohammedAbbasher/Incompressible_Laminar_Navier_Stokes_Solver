import numpy as np


def setup_indices(N):
    """
    Set up index mapping for monolithic system
    """

    nu_u = N * (N - 1)
    nu_v = (N - 1) * N
    nu_p = N * N

    total_unknowns = nu_u + nu_v + nu_p

    u_idx = np.arange(nu_u).reshape(N, N - 1)

    v_idx = nu_u + np.arange(nu_v).reshape(N - 1, N)

    p_idx = nu_u + nu_v + np.arange(nu_p).reshape(N, N)

    return nu_u, nu_v, nu_p, total_unknowns, u_idx, v_idx, p_idx
