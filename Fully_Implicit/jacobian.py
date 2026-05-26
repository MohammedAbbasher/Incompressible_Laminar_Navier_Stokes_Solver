from scipy.sparse import lil_matrix


def compute_convective_jacobian(
    N,
    h,
    total_unknowns,
    u_idx,
    v_idx,
    u,
    v
):
        """
        Compute the Jacobian matrix for convective terms (u·∇)u
        This linearizes the convective terms around the current solution
        
        Returns two sparse matrices: J_u (for u-equation) and J_v (for v-equation)
        """

        
        # Initialize sparse matrices for convective Jacobian
        J_u = lil_matrix((total_unknowns, total_unknowns))
        J_v = lil_matrix((total_unknowns, total_unknowns))
        
        # Fill convective Jacobian for u-equation
        for i in range(1, N-1):  # Interior points
            for j in range(1, N-2):  # Interior u-points
                idx = u_idx[i, j]
                
                # Linearization of u * ∂u/∂x
                # ∂/∂u_{i,j} (u * ∂u/∂x) = u_ij * (∂/∂u_{i,j})(∂u/∂x) + (∂u/∂x) * (∂/∂u_{i,j})(u)
                # But simpler: treat as u^{n+1} * (∂u/∂x)^n + u^n * (∂u/∂x)^{n+1}
                # For Picard iteration (fixed point): use u^n for coefficients
                
                # Coefficient for u^{n+1}_{i,j} from ∂u/∂x term
                J_u[idx, idx] += u[i, j] * (1/(2*h))  # From forward difference part
                J_u[idx, idx] += u[i, j] * (-1/(2*h)) # From backward difference part (net 0)
                
                # Actually, let's use a simpler approach: Picard linearization
                # (u·∇)u ≈ (u^n·∇)u^{n+1}
                # So the Jacobian contains derivatives of the convective operator w.r.t u^{n+1}
                # with u^n fixed
                
                # For u * ∂u/∂x term:
                # Contribution to u_{i,j-1}
                if j > 0:
                    J_u[idx, u_idx[i, j-1]] += -u[i, j] / (2*h)
                
                # Contribution to u_{i,j+1}
                if j < N-2:
                    J_u[idx, u_idx[i, j+1]] += u[i, j] / (2*h)
                
                # For v * ∂u/∂y term (need v at u-point):
                if i > 0 and i < N-1 and j < N-2:
                    v_avg = 0.25 * (v[i-1, j] + v[i-1, j+1] + v[i, j] + v[i, j+1])
                    
                    # Contribution to u_{i-1,j}
                    J_u[idx, u_idx[i-1, j]] += -v_avg / (2*h)
                    
                    # Contribution to u_{i+1,j}
                    J_u[idx, u_idx[i+1, j]] += v_avg / (2*h)
        
        # Fill convective Jacobian for v-equation
        for i in range(1, N-2):  # Interior v-points
            for j in range(1, N-1):  # Interior points
                idx = v_idx[i, j]
                
                # For u * ∂v/∂x term:
                if i < N-2 and j > 0 and j < N-1:
                    u_avg = 0.25 * (u[i, j-1] + u[i, j] + u[i+1, j-1] + u[i+1, j])
                    
                    # Contribution to v_{i,j-1}
                    J_v[idx, v_idx[i, j-1]] += -u_avg / (2*h)
                    
                    # Contribution to v_{i,j+1}
                    J_v[idx, v_idx[i, j+1]] += u_avg / (2*h)
                
                # For v * ∂v/∂y term:
                # Contribution to v_{i-1,j}
                if i > 0:
                    J_v[idx, v_idx[i-1, j]] += -v[i, j] / (2*h)
                
                # Contribution to v_{i+1,j}
                if i < N-2:
                    J_v[idx, v_idx[i+1, j]] += v[i, j] / (2*h)
        
        return J_u.tocsr(), J_v.tocsr()
