import numpy as np


def compute_convective_residual(
    N,
    h,
    u,
    v
):
        """
        Compute convective residual (u·∇)u
        This is the nonlinear part that goes into RHS for Newton/Picard iterations
        """
        N = self.N
        h = self.h
        
        conv_u = np.zeros_like(u)
        conv_v = np.zeros_like(v)
        
        # Compute convective term for u-equation: u * ∂u/∂x + v * ∂u/∂y
        for i in range(1, N-1):
            for j in range(1, N-2):
                # u at current point
                u_ij = u[i, j]
                
                # ∂u/∂x (central difference)
                du_dx = (u[i, j+1] - u[i, j-1]) / (2*h)
                
                # Interpolate v to u location
                if i > 0 and i < N-1 and j < N-2:
                    v_avg = 0.25 * (v[i-1, j] + v[i-1, j+1] + v[i, j] + v[i, j+1])
                else:
                    v_avg = 0.0
                
                # ∂u/∂y
                u_top = 0.5 * (u[i, j] + u[i, j+1]) if j < N-2 else u[i, j]
                u_bottom = 0.5 * (u[i, j-1] + u[i, j]) if j > 0 else u[i, j]
                du_dy = (u_top - u_bottom) / h
                
                conv_u[i, j] = u_ij * du_dx + v_avg * du_dy
        
        # Compute convective term for v-equation: u * ∂v/∂x + v * ∂v/∂y
        for i in range(1, N-2):
            for j in range(1, N-1):
                # v at current point
                v_ij = v[i, j]
                
                # ∂v/∂y
                dv_dy = (v[i+1, j] - v[i-1, j]) / (2*h) if i > 0 and i < N-2 else 0.0
                
                # Interpolate u to v location
                if i < N-2 and j > 0 and j < N-1:
                    u_avg = 0.25 * (u[i, j-1] + u[i, j] + u[i+1, j-1] + u[i+1, j])
                else:
                    u_avg = 0.0
                
                # ∂v/∂x
                v_right = 0.5 * (v[i, j] + v[i+1, j]) if i < N-2 else v[i, j]
                v_left = 0.5 * (v[i-1, j] + v[i, j]) if i > 0 else v[i, j]
                dv_dx = (v_right - v_left) / h
                
                conv_v[i, j] = u_avg * dv_dx + v_ij * dv_dy
        
        return conv_u, conv_v