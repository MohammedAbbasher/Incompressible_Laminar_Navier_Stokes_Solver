# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:57:17 2026

@author: mohammed.abbasher
"""
import numpy as np

from grid import setup_grid
from metrics import compute_metrics
from interpolation import initialize_quick_coefficients


def initialize_solver(
    nx,
    ny,
    Re,
    dt,
    Lx=1.0,
    Ly=1.0,
    curvilinear=False,
    alpha=0.1
):
    """
    Initialize all variables and arrays for the
    semi-implicit curvilinear Navier-Stokes solver.
    """

    # =====================================================
    # GRID SPACING IN COMPUTATIONAL SPACE
    # =====================================================

    dxi = 1.0 / (nx - 1)
    deta = 1.0 / (ny - 1)

    # =====================================================
    # FLUID PROPERTIES
    # =====================================================

    nu = 1.0 / Re

    # =====================================================
    # PRIMARY FLOW VARIABLES
    # =====================================================

    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    p = np.zeros((nx, ny))
    phi = np.zeros((nx, ny))

    # =====================================================
    # VOLUME FLUXES
    # =====================================================

    U = np.zeros((nx + 1, ny))
    V = np.zeros((nx, ny + 1))

    # =====================================================
    # INTERMEDIATE VELOCITIES
    # =====================================================

    u_star = np.zeros((nx, ny))
    v_star = np.zeros((nx, ny))

    # =====================================================
    # INTERMEDIATE FLUXES
    # =====================================================

    U_star = np.zeros((nx + 1, ny))
    V_star = np.zeros((nx, ny + 1))

    # =====================================================
    # ADAMS-BASHFORTH STORAGE
    # =====================================================

    C_u_old = np.zeros((nx, ny))
    C_v_old = np.zeros((nx, ny))

    D_E_u_old = np.zeros((nx, ny))
    D_E_v_old = np.zeros((nx, ny))

    # =====================================================
    # COMPUTATIONAL GRID
    # =====================================================

    xi = np.linspace(0.0, 1.0, nx)
    eta = np.linspace(0.0, 1.0, ny)

    XI, ETA = np.meshgrid(
        xi,
        eta,
        indexing='ij'
    )

    # =====================================================
    # PHYSICAL GRID
    # =====================================================

    x = np.zeros((nx, ny))
    y = np.zeros((nx, ny))

    # =====================================================
    # METRIC TERMS
    # =====================================================

    Jinv = np.zeros((nx, ny))

    dxi_dx = np.ones((nx, ny))
    dxi_dy = np.zeros((nx, ny))

    deta_dx = np.zeros((nx, ny))
    deta_dy = np.ones((nx, ny))

    # =====================================================
    # MESH SKEWNESS TENSOR
    # =====================================================

    G11 = np.ones((nx, ny))
    G12 = np.zeros((nx, ny))
    G22 = np.ones((nx, ny))

    # =====================================================
    # SETUP GRID
    # =====================================================

    x, y = setup_grid(
        nx,
        ny,
        XI,
        ETA,
        Lx,
        Ly,
        curvilinear,
        alpha
    )

    # =====================================================
    # COMPUTE METRICS
    # =====================================================

    (
        Jinv,
        dxi_dx,
        dxi_dy,
        deta_dx,
        deta_dy,
        G11,
        G12,
        G22,
        dx_min,
        dy_min
    ) = compute_metrics(
        nx,
        ny,
        x,
        y,
        dxi,
        deta,
        curvilinear
    )

    # =====================================================
    # QUICK COEFFICIENTS
    # =====================================================

    quick_weights = initialize_quick_coefficients()

    # =====================================================
    # RETURN EVERYTHING
    # =====================================================

    return (
        nx,
        ny,
        Re,
        dt,
        Lx,
        Ly,
        curvilinear,
        alpha,
        nu,
        dxi,
        deta,
        xi,
        eta,
        XI,
        ETA,
        x,
        y,
        u,
        v,
        p,
        phi,
        U,
        V,
        u_star,
        v_star,
        U_star,
        V_star,
        C_u_old,
        C_v_old,
        D_E_u_old,
        D_E_v_old,
        Jinv,
        dxi_dx,
        dxi_dy,
        deta_dx,
        deta_dy,
        G11,
        G12,
        G22,
        dx_min,
        dy_min,
        quick_weights
    )
    
def setup_grid(self):
    """Set up the physical grid coordinates."""
    if not curvilinear:
        # Cartesian grid
        x = XI * Lx
        y = ETA * Ly
    else:
        # Simple sheared grid (smoother transformation)
        for i in range(nx):
            for j in range(ny):
                xi = XI[i, j]
                eta = ETA[i, j]
                
                # Smoother transformation
                x[i, j] = xi * Lx + alpha * Lx * eta * np.sin(np.pi * xi)
                y[i, j] = eta * Ly + alpha * Ly * xi * np.cos(np.pi * eta)

def compute_metrics(self):
    """Compute metric terms for curvilinear transformation."""
    if not curvilinear:
        # Cartesian metrics are trivial
        Jinv = np.ones((nx, ny))
        # All other metrics remain as initialized (identity-like)
    else:
        # Compute derivatives using second-order central differences
        dx_dxi = np.zeros((nx, ny))
        dx_deta = np.zeros((nx, ny))
        dy_dxi = np.zeros((nx, ny))
        dy_deta = np.zeros((nx, ny))
        
        # Interior points
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                dx_dxi[i, j] = (x[i+1, j] - x[i-1, j]) / (2 * dxi)
                dx_deta[i, j] = (x[i, j+1] - x[i, j-1]) / (2 * deta)
                dy_dxi[i, j] = (y[i+1, j] - y[i-1, j]) / (2 * dxi)
                dy_deta[i, j] = (y[i, j+1] - y[i, j-1]) / (2 * deta)
        
        # Boundaries (one-sided differences)
        # Left boundary (i=0)
        i = 0
        for j in range(1, ny-1):
            dx_dxi[i, j] = (x[i+1, j] - x[i, j]) / dxi
            dx_deta[i, j] = (x[i, j+1] - x[i, j-1]) / (2 * deta)
            dy_dxi[i, j] = (y[i+1, j] - y[i, j]) / dxi
            dy_deta[i, j] = (y[i, j+1] - y[i, j-1]) / (2 * deta)
        
        # Right boundary (i=nx-1)
        i = nx-1
        for j in range(1, ny-1):
            dx_dxi[i, j] = (x[i, j] - x[i-1, j]) / dxi
            dx_deta[i, j] = (x[i, j+1] - x[i, j-1]) / (2 * deta)
            dy_dxi[i, j] = (y[i, j] - y[i-1, j]) / dxi
            dy_deta[i, j] = (y[i, j+1] - y[i, j-1]) / (2 * deta)
        
        # Bottom boundary (j=0)
        j = 0
        for i in range(1, nx-1):
            dx_dxi[i, j] = (x[i+1, j] - x[i-1, j]) / (2 * dxi)
            dx_deta[i, j] = (x[i, j+1] - x[i, j]) / deta
            dy_dxi[i, j] = (y[i+1, j] - y[i-1, j]) / (2 * dxi)
            dy_deta[i, j] = (y[i, j+1] - y[i, j]) / deta
        
        # Top boundary (j=ny-1)
        j = ny-1
        for i in range(1, nx-1):
            dx_dxi[i, j] = (x[i+1, j] - x[i-1, j]) / (2 * dxi)
            dx_deta[i, j] = (x[i, j] - x[i, j-1]) / deta
            dy_dxi[i, j] = (y[i+1, j] - y[i-1, j]) / (2 * dxi)
            dy_deta[i, j] = (y[i, j] - y[i, j-1]) / deta
        
        # Corners (simple averages)
        for i in [0, nx-1]:
            for j in [0, ny-1]:
                dx_dxi[i, j] = 0.0
                dx_deta[i, j] = 0.0
                dy_dxi[i, j] = 0.0
                dy_deta[i, j] = 0.0
        
        # Compute Jacobian and metrics
        for i in range(nx):
            for j in range(ny):
                # Jacobian determinant
                J = dx_dxi[i, j] * dy_deta[i, j] - dx_deta[i, j] * dy_dxi[i, j]
                
                # Avoid division by zero
                if abs(J) < 1e-10:
                    J = 1.0
                
                Jinv[i, j] = J
                
                # Inverse metrics
                dxi_dx[i, j] = dy_deta[i, j] / J
                dxi_dy[i, j] = -dx_deta[i, j] / J
                deta_dx[i, j] = -dy_dxi[i, j] / J
                deta_dy[i, j] = dx_dxi[i, j] / J
                
                # Mesh skewness tensor (Equation 6)
                G11[i, j] = Jinv[i, j] * (
                    dxi_dx[i, j]**2 + dxi_dy[i, j]**2)
                G22[i, j] = Jinv[i, j] * (
                    deta_dx[i, j]**2 + deta_dy[i, j]**2)
                G12[i, j] = Jinv[i, j] * (
                    dxi_dx[i, j] * deta_dx[i, j] + 
                    dxi_dy[i, j] * deta_dy[i, j])
    
    # Store minimum physical grid spacing for CFL
    dx_min = np.min(np.sqrt(np.diff(x, axis=0)**2 + np.diff(y, axis=0)**2))
    dy_min = np.min(np.sqrt(np.diff(x, axis=1)**2 + np.diff(y, axis=1)**2))

def initialize_quick_coefficients(self):
    """Initialize coefficients for QUICK scheme."""
    quick_weights = {
        'positive': np.array([-1/8, 6/8, 3/8]),
        'negative': np.array([3/8, 6/8, -1/8])
    }

def apply_boundary_conditions(self, u=None, v=None):
    """Apply boundary conditions for lid-driven cavity."""
    if u is None:
        u = u
    if v is None:
        v = v
    
    # No-slip conditions on walls
    u[0, :] = 0.0      # left wall
    u[-1, :] = 0.0     # right wall
    u[:, 0] = 0.0      # bottom wall
    u[:, -1] = 1.0     # top wall (lid moves with u=1)
    
    v[0, :] = 0.0      # left wall
    v[-1, :] = 0.0     # right wall
    v[:, 0] = 0.0      # bottom wall
    v[:, -1] = 0.0     # top wall
    
    return u, v

def compute_convective_term_simple(self, u, v):
    """Simplified convective term for stability."""
    C_u = np.zeros_like(u)
    C_v = np.zeros_like(v)
    
    # Simple second-order central difference for now
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # ξ-direction derivatives
            du_dxi = (u[i+1, j] - u[i-1, j]) / (2 * dxi)
            dv_dxi = (v[i+1, j] - v[i-1, j]) / (2 * dxi)
            
            # η-direction derivatives
            du_deta = (u[i, j+1] - u[i, j-1]) / (2 * deta)
            dv_deta = (v[i, j+1] - v[i, j-1]) / (2 * deta)
            
            # Transform to physical derivatives
            du_dx = dxi_dx[i, j] * du_dxi + deta_dx[i, j] * du_deta
            du_dy = dxi_dy[i, j] * du_dxi + deta_dy[i, j] * du_deta
            dv_dx = dxi_dx[i, j] * dv_dxi + deta_dx[i, j] * dv_deta
            dv_dy = dxi_dy[i, j] * dv_dxi + deta_dy[i, j] * dv_deta
            
            # Convective terms (non-conservative form for stability)
            C_u[i, j] = -(u[i, j] * du_dx + v[i, j] * du_dy)
            C_v[i, j] = -(u[i, j] * dv_dx + v[i, j] * dv_dy)
    
    return C_u, C_v

def compute_viscous_terms(self, u, v):
    """Compute viscous terms."""
    D_I_u = np.zeros_like(u)
    D_E_u = np.zeros_like(u)
    D_I_v = np.zeros_like(v)
    D_E_v = np.zeros_like(v)
    
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Diagonal viscous terms (implicit)
            # ξ-direction
            dudxi_e = (u[i+1, j] - u[i, j]) / dxi
            dudxi_w = (u[i, j] - u[i-1, j]) / dxi
            G11_e = 0.5 * (G11[i+1, j] + G11[i, j])
            G11_w = 0.5 * (G11[i, j] + G11[i-1, j])
            term_xi = (G11_e * dudxi_e - G11_w * dudxi_w) / dxi
            
            # η-direction
            dudeta_n = (u[i, j+1] - u[i, j]) / deta
            dudeta_s = (u[i, j] - u[i, j-1]) / deta
            G22_n = 0.5 * (G22[i, j+1] + G22[i, j])
            G22_s = 0.5 * (G22[i, j] + G22[i, j-1])
            term_eta = (G22_n * dudeta_n - G22_s * dudeta_s) / deta
            
            D_I_u[i, j] = nu * (term_xi + term_eta)
            
            # Similar for v
            dvdxi_e = (v[i+1, j] - v[i, j]) / dxi
            dvdxi_w = (v[i, j] - v[i-1, j]) / dxi
            dvdeta_n = (v[i, j+1] - v[i, j]) / deta
            dvdeta_s = (v[i, j] - v[i, j-1]) / deta
            
            term_xi_v = (G11_e * dvdxi_e - G11_w * dvdxi_w) / dxi
            term_eta_v = (G22_n * dvdeta_n - G22_s * dvdeta_s) / deta
            
            D_I_v[i, j] = nu * (term_xi_v + term_eta_v)
            
            # Off-diagonal terms (explicit) - simplified
            if curvilinear:
                # Cross derivatives for u
                dudeta_e = (u[i+1, j+1] + u[i, j+1] - u[i+1, j-1] - u[i, j-1]) / (4 * deta)
                dudeta_w = (u[i, j+1] + u[i-1, j+1] - u[i, j-1] - u[i-1, j-1]) / (4 * deta)
                dudxi_n = (u[i+1, j+1] + u[i+1, j] - u[i-1, j+1] - u[i-1, j]) / (4 * dxi)
                dudxi_s = (u[i+1, j] + u[i+1, j-1] - u[i-1, j] - u[i-1, j-1]) / (4 * dxi)
                
                G12_e = 0.5 * (G12[i+1, j] + G12[i, j])
                G12_w = 0.5 * (G12[i, j] + G12[i-1, j])
                G12_n = 0.5 * (G12[i, j+1] + G12[i, j])
                G12_s = 0.5 * (G12[i, j] + G12[i, j-1])
                
                term_cross_xi = (G12_e * dudeta_e - G12_w * dudeta_w) / dxi
                term_cross_eta = (G12_n * dudxi_n - G12_s * dudxi_s) / deta
                
                D_E_u[i, j] = nu * (term_cross_xi + term_cross_eta)
                
                # Similar for v
                dvdeta_e = (v[i+1, j+1] + v[i, j+1] - v[i+1, j-1] - v[i, j-1]) / (4 * deta)
                dvdeta_w = (v[i, j+1] + v[i-1, j+1] - v[i, j-1] - v[i-1, j-1]) / (4 * deta)
                dvdxi_n = (v[i+1, j+1] + v[i+1, j] - v[i-1, j+1] - v[i-1, j]) / (4 * dxi)
                dvdxi_s = (v[i+1, j] + v[i+1, j-1] - v[i-1, j] - v[i-1, j-1]) / (4 * dxi)
                
                term_cross_xi_v = (G12_e * dvdeta_e - G12_w * dvdeta_w) / dxi
                term_cross_eta_v = (G12_n * dvdxi_n - G12_s * dvdxi_s) / deta
                
                D_E_v[i, j] = nu * (term_cross_xi_v + term_cross_eta_v)
    
    return D_I_u, D_E_u, D_I_v, D_E_v

def step1_solve_intermediate_velocity(self):
    """Solve for intermediate velocity u_j*."""
    # Apply boundary conditions
    u, v = apply_boundary_conditions(u, v)
    
    # Compute terms at time level n
    C_u_n, C_v_n = compute_convective_term_simple(u, v)
    D_I_u_n, D_E_u_n, D_I_v_n, D_E_v_n = compute_viscous_terms(u, v)
    
    # Adams-Bashforth for explicit terms
    if np.all(C_u_old == 0):
        # First time step: use Euler forward
        AB_u = C_u_n + D_E_u_n
        AB_v = C_v_n + D_E_v_n
    else:
        # Second-order Adams-Bashforth
        AB_u = 1.5 * (C_u_n + D_E_u_n) - 0.5 * (C_u_old + D_E_u_old)
        AB_v = 1.5 * (C_v_n + D_E_v_n) - 0.5 * (C_v_old + D_E_v_old)
    
    # Store for next time step
    C_u_old = C_u_n.copy()
    C_v_old = C_v_n.copy()
    D_E_u_old = D_E_u_n.copy()
    D_E_v_old = D_E_v_n.copy()
    
    # Compute RHS
    RHS_u = np.zeros_like(u)
    RHS_v = np.zeros_like(v)
    
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Equation (20) RHS
            RHS_u[i, j] = (dt / Jinv[i, j]) * (AB_u[i, j] + D_I_u_n[i, j])
            RHS_v[i, j] = (dt / Jinv[i, j]) * (AB_v[i, j] + D_I_v_n[i, j])
    
    # Solve using ADI method (simplified - for stability use explicit for now)
    u_star = u + RHS_u
    v_star = v + RHS_v
    
    # Apply boundary conditions to intermediate velocities
    u_star, v_star = apply_boundary_conditions(u_star, v_star)

def step2_interpolate_and_compute_rhs(self):
    """Interpolate u_j* to faces and compute RHS of pressure equation."""
    # Compute U* at ξ faces (i+1/2, j) - simple linear interpolation
    for i in range(1, nx):
        for j in range(1, ny-1):
            # Average metrics to face
            J_face = 0.5 * (Jinv[i-1, j] + Jinv[i, j])
            dxi_dx_face = 0.5 * (dxi_dx[i-1, j] + dxi_dx[i, j])
            dxi_dy_face = 0.5 * (dxi_dy[i-1, j] + dxi_dy[i, j])
            
            # Average velocities to face
            u_face = 0.5 * (u_star[i-1, j] + u_star[i, j])
            v_face = 0.5 * (v_star[i-1, j] + v_star[i, j])
            
            # Compute intermediate volume flux
            U_star[i, j] = J_face * (dxi_dx_face * u_face + dxi_dy_face * v_face)
    
    # Compute V* at η faces (i, j+1/2)
    for i in range(1, nx-1):
        for j in range(1, ny):
            J_face = 0.5 * (Jinv[i, j-1] + Jinv[i, j])
            deta_dx_face = 0.5 * (deta_dx[i, j-1] + deta_dx[i, j])
            deta_dy_face = 0.5 * (deta_dy[i, j-1] + deta_dy[i, j])
            
            u_face = 0.5 * (u_star[i, j-1] + u_star[i, j])
            v_face = 0.5 * (v_star[i, j-1] + v_star[i, j])
            
            V_star[i, j] = J_face * (deta_dx_face * u_face + deta_dy_face * v_face)
    
    # Boundary conditions
    U_star[0, :] = 0.0
    U_star[-1, :] = 0.0
    V_star[:, 0] = 0.0
    V_star[:, -1] = 0.0

def step3_solve_pressure_poisson(self):
    """Solve pressure Poisson equation."""
    # Compute RHS: (1/Δt) ∂U_m*/∂ξ_m
    rhs = np.zeros((nx, ny))
    
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            div_U_star = (U_star[i+1, j] - U_star[i, j]) / dxi + \
                        (V_star[i, j+1] - V_star[i, j]) / deta
            rhs[i, j] = div_U_star / dt
    
    # Solve using simple Poisson solver (for stability)
    phi = solve_simple_poisson(rhs)

def solve_simple_poisson(self, rhs, max_iter=100, tol=1e-6):
    """Simple Poisson solver for stability."""
    phi = np.zeros((nx, ny))
    
    for iteration in range(max_iter):
        phi_old = phi.copy()
        
        # Gauss-Seidel iteration
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                # Simplified Laplacian (Cartesian for stability)
                phi[i, j] = 0.25 * (phi[i+1, j] + phi[i-1, j] + 
                                   phi[i, j+1] + phi[i, j-1] - 
                                   rhs[i, j] * dxi**2)
        
        # Neumann boundary conditions
        phi[0, :] = phi[1, :]
        phi[-1, :] = phi[-2, :]
        phi[:, 0] = phi[:, 1]
        phi[:, -1] = phi[:, -2]
        
        # Check convergence
        residual = np.max(np.abs(phi - phi_old))
        if residual < tol:
            break
    
    return phi

def step4_correct_velocities_and_fluxes(self):
    """Correct velocities and update fluxes."""
    # Correct Cartesian velocities
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Compute pressure gradient
            dphi_dxi = (phi[i+1, j] - phi[i-1, j]) / (2 * dxi)
            dphi_deta = (phi[i, j+1] - phi[i, j-1]) / (2 * deta)
            
            # Transform to physical gradient
            dp_dx = (dphi_dxi * dxi_dx[i, j] + dphi_deta * deta_dx[i, j])
            dp_dy = (dphi_dxi * dxi_dy[i, j] + dphi_deta * deta_dy[i, j])
            
            # Correct velocities (Equation 16)
            u[i, j] = u_star[i, j] - dt * dp_dx
            v[i, j] = v_star[i, j] - dt * dp_dy
    
    # Apply boundary conditions
    u, v = apply_boundary_conditions(u, v)
    
    # Update volume fluxes (simplified)
    for i in range(1, nx):
        for j in range(1, ny-1):
            U[i, j] = U_star[i, j]
    
    for i in range(1, nx-1):
        for j in range(1, ny):
            V[i, j] = V_star[i, j]
    
    # Boundary conditions
    U[0, :] = 0.0
    U[-1, :] = 0.0
    V[:, 0] = 0.0
    V[:, -1] = 0.0
