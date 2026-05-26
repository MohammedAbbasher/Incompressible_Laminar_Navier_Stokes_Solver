import numpy as np

from scipy.sparse.linalg import (
    spsolve,
    bicgstab,
    LinearOperator
)

from grid import setup_indices
from matrix_assembly import build_fully_implicit_matrix
from boundary_conditions import apply_boundary_conditions_to_rhs
from residuals import compute_convective_residual


def solve_fully_implicit(
    Re,
    U_lid,
    N,
    dt,
    final_time,
    h,
    nu,
    method,
    max_nonlinear_iter,
    tol,
    use_direct
):
    
    
    # =====================================================
    # STAGGERED GRID VARIABLES
    # =====================================================

    p = np.zeros((N, N))

    u = np.zeros((N, N - 1))

    v = np.zeros((N - 1, N))

    # =====================================================
    # INDEX SETUP
    # =====================================================

    (
        nu_u,
        nu_v,
        nu_p,
        total_unknowns,
        u_idx,
        v_idx,
        p_idx
    ) = setup_indices(N)

    # =====================================================
    # TIME SETUP
    # =====================================================

    num_steps = int(final_time / dt)

    time_history = []
    divergence_history = []
    u_max_history = []

    print(f"Starting FULLY IMPLICIT time stepping with {num_steps} steps...")
    print(f"Method: {method}, Re={Re}, dt={dt}")

    # =====================================================
    # TIME LOOP
    # =====================================================

    for step in range(num_steps):

        u_old = u.copy()

        v_old = v.copy()

        # =================================================
        # NONLINEAR ITERATIONS
        # =================================================

        for nl_iter in range(max_nonlinear_iter):

            # =============================================
            # Compute nonlinear residuals
            # =============================================

            if method == 'newton' and nl_iter > 0:

                conv_u, conv_v = compute_convective_residual(
                    N,
                    h,
                    u,
                    v
                )

            # =============================================
            # RHS VECTOR
            # =============================================

            rhs = np.zeros(total_unknowns)

            # ---------------------------------------------
            # U-equation
            # ---------------------------------------------

            for i in range(N):

                for j in range(N - 1):

                    idx = u_idx[i, j]

                    rhs[idx] = u_old[i, j] / dt

                    if method == 'newton' and nl_iter > 0:

                        rhs[idx] -= conv_u[i, j]

            # ---------------------------------------------
            # V-equation
            # ---------------------------------------------

            for i in range(N - 1):

                for j in range(N):

                    idx = v_idx[i, j]

                    rhs[idx] = v_old[i, j] / dt

                    if method == 'newton' and nl_iter > 0:

                        rhs[idx] -= conv_v[i, j]

            # =============================================
            # APPLY BC
            # =============================================

            rhs = apply_boundary_conditions_to_rhs(
                rhs,
                N,
                U_lid,
                u_idx,
                v_idx,
                p_idx
            )

            # =============================================
            # BUILD MATRIX
            # =============================================

            A = build_fully_implicit_matrix(
                N,
                h,
                dt,
                nu,
                total_unknowns,
                u_idx,
                v_idx,
                p_idx,
                u,
                v,
                apply_bc=True
            )

            # =============================================
            # SOLVE SYSTEM
            # =============================================

            if use_direct:

                x = spsolve(A, rhs)

            else:

                M = LinearOperator(
                    (total_unknowns, total_unknowns),
                    matvec=lambda x: x / A.diagonal()
                )

                x, info = bicgstab(
                    A,
                    rhs,
                    tol=1e-8,
                    maxiter=2000,
                    M=M
                )

                if info != 0:

                    print("Warning: BiCGStab did not converge")

            # =============================================
            # EXTRACT SOLUTION
            # =============================================

            u_new = x[u_idx].reshape(N, N - 1)

            v_new = x[v_idx].reshape(N - 1, N)

            p_new = x[p_idx].reshape(N, N)

            # =============================================
            # NONLINEAR CONVERGENCE
            # =============================================

            if nl_iter > 0:

                du_norm = (
                    np.linalg.norm(u_new - u)
                    /
                    (np.linalg.norm(u) + 1e-10)
                )

                dv_norm = (
                    np.linalg.norm(v_new - v)
                    /
                    (np.linalg.norm(v) + 1e-10)
                )

                if du_norm < tol and dv_norm < tol:

                    print(
                        f"Step {step}: "
                        f"Converged in {nl_iter + 1} iterations"
                    )

                    break

            # =============================================
            # UPDATE SOLUTION
            # =============================================

            u = u_new

            v = v_new

            p = p_new

        # =================================================
        # DIAGNOSTICS
        # =================================================

        div_max = 0.0

        for i in range(1, N - 1):

            for j in range(1, N - 1):

                div = (
                    (u[i, j] - u[i, j - 1]) / h
                    +
                    (v[i, j] - v[i - 1, j]) / h
                )

                div_max = max(div_max, abs(div))

        u_max = np.max(np.abs(u))

        v_max = np.max(np.abs(v))

        # =================================================
        # MONITOR
        # =================================================

        if step % 50 == 0 or step == num_steps - 1:

            print(
                f"Step {step}/{num_steps}, "
                f"Time = {step * dt:.3f}, "
                f"Max div = {div_max:.2e}, "
                f"Max u = {u_max:.4f}, "
                f"Max v = {v_max:.4f}"
            )

            time_history.append(step * dt)

            divergence_history.append(div_max)

            u_max_history.append(u_max)

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print("\nFinal statistics:")

    print(f"Maximum u-velocity: {np.max(np.abs(u)):.6f}")

    print(f"Maximum v-velocity: {np.max(np.abs(v)):.6f}")

    print(f"Maximum divergence: {div_max:.2e}")

    return (
        u,
        v,
        p,
        h,
        U_lid,
        time_history,
        divergence_history,
        u_max_history
    )
