import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata

def interpolate_to_center(
    u,
    v,
    N,
    U_lid
    ):
        """Interpolate staggered velocities to cell centers"""
       
        u_center = np.zeros((N, N))
        v_center = np.zeros((N, N))
        
        # Interpolate u to center
        for i in range(N):
            for j in range(1, N-1):
                u_center[i, j] = 0.5 * (u[i, j-1] + u[i, j])
        
        # Interpolate v to center
        for j in range(N):
            for i in range(1, N-1):
                v_center[i, j] = 0.5 * (v[i-1, j] + v[i, j])
        
        # Apply boundary conditions
        u_center[-1, :] = U_lid
        v_center[-1, :] = 0.0
        u_center[0, :] = 0.0
        v_center[0, :] = 0.0
        u_center[:, 0] = 0.0
        v_center[:, 0] = 0.0
        u_center[:, -1] = 0.0
        v_center[:, -1] = 0.0
        
        return u_center, v_center
    
def compute_streamfunction(self):
        """Compute streamfunction"""
        N = self.N
        h = self.h
        
        u_center, v_center = self.interpolate_to_center()
        
        # Compute vorticity
        omega = np.zeros((N, N))
        for i in range(1, N-1):
            for j in range(1, N-1):
                dv_dx = (v_center[i, j+1] - v_center[i, j-1]) / (2*h)
                du_dy = (u_center[i+1, j] - u_center[i-1, j]) / (2*h)
                omega[i, j] = dv_dx - du_dy
        
        # Solve Poisson equation
        psi = np.zeros((N, N))
        max_iter = 10000
        tolerance = 1e-6
        
        for iter in range(max_iter):
            psi_old = psi.copy()
            for i in range(1, N-1):
                for j in range(1, N-1):
                    psi[i, j] = 0.25 * (psi_old[i+1, j] + psi_old[i-1, j] +
                                        psi_old[i, j+1] + psi_old[i, j-1] +
                                        h**2 * omega[i, j])
            
            if np.max(np.abs(psi - psi_old)) < tolerance:
                break
        
        # Boundaries
        psi[0, :] = 0.0
        psi[-1, :] = 0.0
        psi[:, 0] = 0.0
        psi[:, -1] = 0.0
        
        return psi
    
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import griddata


def Ghia(Re):

    x = [
        0.0000, 0.0625, 0.0703, 0.0781, 0.0938,
        0.1563, 0.2266, 0.2344, 0.5000, 0.8047,
        0.8594, 0.9063, 0.9453, 0.9531, 0.9609,
        0.9688, 1.0000
    ]

    y = [
        0.0000, 0.0547, 0.0625, 0.0703, 0.1016,
        0.1719, 0.2813, 0.4531, 0.5000, 0.6172,
        0.7344, 0.8516, 0.9531, 0.9609, 0.9688,
        0.9766, 1.0000
    ]

    reference_data = {

        100: {

            'u': [
                0.00000, -0.03717, -0.04192, -0.04775,
                -0.06434, -0.10150, -0.15662, -0.21090,
                -0.20581, -0.13641, 0.00332, 0.23151,
                0.68717, 0.73722, 0.78871, 0.84123,
                1.00000
            ],

            'v': [
                0.00000, 0.09233, 0.10091, 0.10890,
                0.12317, 0.16077, 0.17507, 0.17527,
                0.05454, -0.24533, -0.22445, -0.16914,
                -0.10313, -0.08864, -0.07391, -0.05906,
                0.00000
            ]
        }
    }

    data = reference_data[Re]

    return {
        'x': x,
        'y': y,
        'u': data['u'],
        'v': data['v']
    }


def plot_results(
    u,
    v,
    p,
    N,
    L,
    Re,
    U_lid,
    time_history,
    divergence_history
):

    # ============================================
    # GRID
    # ============================================

    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)

    X, Y = np.meshgrid(x, y)

    # ============================================
    # INTERPOLATE TO CELL CENTERS
    # ============================================

    u_center = np.zeros((N, N))
    v_center = np.zeros((N, N))

    u_center[:, 1:-1] = 0.5 * (
        u[:, :-1] + u[:, 1:]
    )

    u_center[:, 0] = 0.0
    u_center[:, -1] = 0.0

    u_center[-1, :] = U_lid

    v_center[1:-1, :] = 0.5 * (
        v[:-1, :] + v[1:, :]
    )

    vel_mag = np.sqrt(
        u_center**2 + v_center**2
    )

    # ============================================
    # FIGURE
    # ============================================

    fig, axes = plt.subplots(3, 2,
         figsize=(15, 15)
    )
    # ============================================
    # VELOCITY MAGNITUDE
    # ============================================

    im1 = axes[0, 0].contourf(
        X,
        Y,
        vel_mag,
        levels=20,
        cmap='jet'
    )

    axes[0, 0].set_title(
        f'Velocity Magnitude (Re={Re})'
    )

    axes[0, 0].set_aspect('equal')

    plt.colorbar(im1, ax=axes[0, 0])

    # ============================================
    # STREAMLINES
    # ============================================

    axes[0, 1].streamplot(
        X,
        Y,
        u_center,
        v_center,
        density=2
    )

    axes[0, 1].set_title('Streamlines')

    axes[0, 1].set_aspect('equal')

    # ============================================
    # GHIA COMPARISON (U)
    # ============================================

    ref = Ghia(Re)

    ref_y = np.array(ref['y'])

    ref_u = np.array(ref['u'])

    points = np.column_stack(
        (
            X.ravel(),
            Y.ravel()
        )
    )

    u_interp = griddata(
        points,
        u_center.ravel(),
        (0.5, ref_y),
        method='linear'
    )

    y_line = np.linspace(0, L, 200)

    u_line = griddata(
        points,
        u_center.ravel(),
        (0.5 * np.ones_like(y_line), y_line),
        method='linear'
    )
    
    axes[1, 0].plot(
        y_line,
        u_line,
        'b-',
        label='Simulation'
    )

    axes[1, 0].plot(
        ref_y,
        u_interp,
        'bs',
        label='Interpolated'
    )

    axes[1, 0].plot(
        ref_y,
        ref_u,
        'ko',
        label='Ghia et al.'
    )

    axes[1, 0].set_title(
        'Centerline u-velocity'
    )

    axes[1, 0].legend()

    axes[1, 0].grid(True)

    # ============================================
    # GHIA COMPARISON (V)
    # ============================================

    ref_x = np.array(ref['x'])

    ref_v = np.array(ref['v'])

    v_interp = griddata(
        points,
        v_center.ravel(),
        (ref_x, 0.5),
        method='linear'
    )

    x_line = np.linspace(0, L, 200)

    v_line = griddata(
        points,
        v_center.ravel(),
        (x_line, 0.5 * np.ones_like(x_line)),
        method='linear'
    )
    
    axes[1, 1].plot(
        x_line,
        v_line,
        'b-',
        label='Simulation'
    )

    axes[1, 1].plot(
        ref_x,
        v_interp,
        'bs',
        label='Interpolated'
    )

    axes[1, 1].plot(
        ref_x,
        ref_v,
        'ko',
        label='Ghia et al.'
    )

    axes[1, 1].set_title(
        'Centerline v-velocity'
    )

    axes[1, 1].legend()

    axes[1, 1].grid(True)

    # ============================================
    # DIVERGENCE HISTORY
    # ============================================
    
    axes[2, 0].semilogy(
        time_history,
        divergence_history,
        'bo-'
    )
    
    axes[2, 0].set_title(
        'Divergence History'
    )
    
    axes[2, 0].set_xlabel('Time')
    
    axes[2, 0].set_ylabel('Maximum Divergence')
    
    axes[2, 0].grid(True)
    
   # ============================================
    # ERROR VS GHIA
    # ============================================
    
    u_error = np.abs(u_interp - ref_u)
    
    v_error = np.abs(v_interp - ref_v)
    
    axes[2, 1].semilogy(
        ref_y,
        u_error,
        'bo-',
        label='u error'
    )
    
    axes[2, 1].semilogy(
        ref_x,
        v_error,
        'rs-',
        label='v error'
    )
    
    axes[2, 1].set_title(
        'Error vs Ghia Reference'
    )
    
    axes[2, 1].set_xlabel('Position')
    
    axes[2, 1].set_ylabel('Absolute Error')
    
    axes[2, 1].legend()
    
    axes[2, 1].grid(True)
    
    # plt.xlabel('Position')
    
    # plt.ylabel('Absolute Error')
    
    # plt.title('Error vs Ghia Reference')
    
    # plt.legend()
    
    # plt.grid(True)
    
    # plt.tight_layout()
    
    # plt.show()

    plt.tight_layout()

    plt.show()