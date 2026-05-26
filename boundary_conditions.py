def apply_boundary_conditions_to_rhs(
    rhs,
    N,
    U_lid,
    u_idx,
    v_idx,
    p_idx
):
    """
    Apply boundary conditions to RHS vector
    """

    # =====================================================
    # U-VELOCITY BOUNDARY CONDITIONS
    # =====================================================

    for i in range(N):

        for j in range(N - 1):

            idx = u_idx[i, j]

            # Bottom wall: u = 0
            if i == 0:

                rhs[idx] = 0.0

            # Top wall: u = U_lid
            elif i == N - 1:

                rhs[idx] = U_lid

            # Left wall: u = 0
            elif j == 0 and (0 < i < N - 1):

                rhs[idx] = 0.0

            # Right wall: u = 0
            elif j == N - 2 and (0 < i < N - 1):

                rhs[idx] = 0.0

    # =====================================================
    # V-VELOCITY BOUNDARY CONDITIONS
    # =====================================================

    for i in range(N - 1):

        for j in range(N):

            idx = v_idx[i, j]

            # All walls: v = 0
            if (
                i == 0
                or i == N - 2
                or j == 0
                or j == N - 1
            ):

                rhs[idx] = 0.0

    # =====================================================
    # PRESSURE GAUGE FIX
    # =====================================================

    center_idx = p_idx[N // 2, N // 2]

    rhs[center_idx] = 0.0

    return rhs