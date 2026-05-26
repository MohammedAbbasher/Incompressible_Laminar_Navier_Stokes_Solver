from scipy.sparse import lil_matrix

from jacobian import compute_convective_jacobian


def build_fully_implicit_matrix(
    N,
    h,
    dt,
    nu,
    total_unknowns,
    u_idx,
    v_idx,
    p_idx,
    u_lin,
    v_lin,
    apply_bc=True
):
    """
    Build the FULLY IMPLICIT monolithic system matrix
    """

    # =====================================================
    # INITIALIZE MATRIX
    # =====================================================

    A = lil_matrix((total_unknowns, total_unknowns))

    # =====================================================
    # CONVECTIVE JACOBIAN
    # =====================================================

    J_u, J_v = compute_convective_jacobian(
        N,
        h,
        total_unknowns,
        u_idx,
        v_idx,
        u_lin,
        v_lin
    )

    # =====================================================
    # U-MOMENTUM EQUATION
    # =====================================================

    for i in range(N):

        for j in range(N - 1):

            idx = u_idx[i, j]

            is_boundary = False

            # Bottom wall
            if i == 0 and apply_bc:

                A[idx, idx] = 1.0
                is_boundary = True

            # Top wall
            elif i == N - 1 and apply_bc:

                A[idx, idx] = 1.0
                is_boundary = True

            # Left wall
            elif (j == 0 and (0 < i < N - 1)) and apply_bc:

                A[idx, idx] = 1.0
                is_boundary = True

            # Right wall
            elif (j == N - 2 and (0 < i < N - 1)) and apply_bc:

                A[idx, idx] = 1.0
                is_boundary = True

            # =============================================
            # INTERIOR POINT
            # =============================================

            if not is_boundary:

                # Time derivative

                A[idx, idx] = 1.0 / dt

                # Diffusion

                A[idx, idx] += 4.0 * nu / (h ** 2)

                if i > 0:
                    A[idx, u_idx[i - 1, j]] += -nu / (h ** 2)

                if i < N - 1:
                    A[idx, u_idx[i + 1, j]] += -nu / (h ** 2)

                if j > 0:
                    A[idx, u_idx[i, j - 1]] += -nu / (h ** 2)

                if j < N - 2:
                    A[idx, u_idx[i, j + 1]] += -nu / (h ** 2)

                # Convective Jacobian

                row = J_u.getrow(idx)

                for k in range(len(row.indices)):

                    col_idx = row.indices[k]
                    val = row.data[k]

                    A[idx, col_idx] += val

                # Pressure gradient

                A[idx, p_idx[i, j + 1]] = 1.0 / h

                A[idx, p_idx[i, j]] = -1.0 / h

    # =====================================================
    # V-MOMENTUM EQUATION
    # =====================================================

    for i in range(N - 1):

        for j in range(N):

            idx = v_idx[i, j]

            is_boundary = False

            # Wall BC

            if (
                i == 0
                or i == N - 2
                or j == 0
                or j == N - 1
            ) and apply_bc:

                A[idx, idx] = 1.0
                is_boundary = True

            # =============================================
            # INTERIOR POINT
            # =============================================

            if not is_boundary:

                # Time derivative

                A[idx, idx] = 1.0 / dt

                # Diffusion

                A[idx, idx] += 4.0 * nu / (h ** 2)

                if i > 0:
                    A[idx, v_idx[i - 1, j]] += -nu / (h ** 2)

                if i < N - 2:
                    A[idx, v_idx[i + 1, j]] += -nu / (h ** 2)

                if j > 0:
                    A[idx, v_idx[i, j - 1]] += -nu / (h ** 2)

                if j < N - 1:
                    A[idx, v_idx[i, j + 1]] += -nu / (h ** 2)

                # Convective Jacobian

                row = J_v.getrow(idx)

                for k in range(len(row.indices)):

                    col_idx = row.indices[k]
                    val = row.data[k]

                    A[idx, col_idx] += val

                # Pressure gradient

                A[idx, p_idx[i + 1, j]] = 1.0 / h

                A[idx, p_idx[i, j]] = -1.0 / h

    # =====================================================
    # CONTINUITY EQUATION
    # =====================================================

    for i in range(N):

        for j in range(N):

            idx = p_idx[i, j]

            if 0 < i < N - 1 and 0 < j < N - 1:

                # du/dx

                if j < N - 1:
                    A[idx, u_idx[i, j]] = 1.0 / h

                if j > 0:
                    A[idx, u_idx[i, j - 1]] = -1.0 / h

                # dv/dy

                if i < N - 1:
                    A[idx, v_idx[i, j]] = 1.0 / h

                if i > 0:
                    A[idx, v_idx[i - 1, j]] = -1.0 / h

            else:

                A[idx, idx] = 1.0

    # =====================================================
    # PRESSURE GAUGE FIX
    # =====================================================

    center_idx = p_idx[N // 2, N // 2]

    A[center_idx, :] = 0.0

    A[center_idx, center_idx] = 1.0

    return A.tocsr()
