import numpy as np
import matplotlib.pyplot as plt
import time

from other import (
    step1_solve_intermediate_velocity,
    step2_interpolate_and_compute_rhs,
    step3_solve_pressure_poisson,
    step4_correct_velocities_and_fluxes,
    apply_boundary_conditions
    )


def time_step(
    u,
    v,
    phi,
    dt,
    dx_min,
    dy_min,
    nx,
    ny,
    dxi,
    deta,
    dxi_dx,
    dxi_dy,
    deta_dx,
    deta_dy
):
    """Perform one time step."""
    # Step 1: Solve for intermediate velocity
    step1_solve_intermediate_velocity()
    
    # Step 2: Interpolate to faces and compute RHS
    step2_interpolate_and_compute_rhs()
    
    # Step 3: Solve pressure Poisson equation
    step3_solve_pressure_poisson()
    
    # Step 4: Correct velocities and fluxes
    step4_correct_velocities_and_fluxes()
    
    # Compute CFL
    cfl = compute_cfl()
    
    return cfl

def compute_cfl(
    u,
    v,
    dt,
    dx_min,
    dy_min
):
    """Compute CFL number."""
    max_vel = np.max(np.sqrt(u**2 + v**2))
    
    if max_vel > 0:
        # Use minimum grid spacing
        min_spacing = min(dx_min, dy_min)
        if min_spacing > 0:
            cfl = max_vel * dt / min_spacing
        else:
            cfl = 0.0
    else:
        cfl = 0.0
    
    return cfl

def compute_divergence(
    u,
    v,
    nx,
    ny,
    dxi,
    deta,
    dxi_dx,
    dxi_dy,
    deta_dx,
    deta_dy
):
    """Compute divergence."""
    div = np.zeros((nx, ny))
    
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Compute ∂u/∂x + ∂v/∂y
            du_dxi = (u[i+1, j] - u[i-1, j]) / (2 * dxi)
            du_deta = (u[i, j+1] - u[i, j-1]) / (2 * deta)
            dv_dxi = (v[i+1, j] - v[i-1, j]) / (2 * dxi)
            dv_deta = (v[i, j+1] - v[i, j-1]) / (2 * deta)
            
            du_dx = dxi_dx[i, j] * du_dxi + deta_dx[i, j] * du_deta
            dv_dy = dxi_dy[i, j] * dv_dxi + deta_dy[i, j] * dv_deta
            
            div[i, j] = du_dx + dv_dy
    
    return div

def run_simulation(
    nx,
    ny,
    Re,
    dt,
    curvilinear,
    alpha,
    phi,
    num_steps,   
    save_interval=10
):
    """Run simulation."""
    grid_type = "curvilinear" if curvilinear else "Cartesian"
    print(f"\nStarting {grid_type} grid simulation")
    print(f"Re={Re}, dt={dt}, grid={nx}x{ny}")
    
    # Initialize with zero velocity
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    u, v = apply_boundary_conditions(u, v)
    
    # Results storage
    results = {
        'u': [], 'v': [], 'phi': [],
        'time': [], 'cfl': [], 'divergence': []
    }
    
    start_time = time.time()
    
    for step in range(num_steps):
        cfl = time_step()
        
        # Store results
        if step % save_interval == 0 or step == num_steps-1:
            results['u'].append(u.copy())
            results['v'].append(v.copy())
            results['phi'].append(phi.copy())
            results['time'].append(step * dt)
            results['cfl'].append(cfl)
            
            div = compute_divergence()
            max_div = np.max(np.abs(div))
            results['divergence'].append(max_div)
            
            print(f"Step {step:4d}: Time={step*dt:7.3f}, CFL={cfl:6.3f}, "
                  f"Max div={max_div:8.2e}")
    
    end_time = time.time()
    print(f"\nSimulation completed in {end_time - start_time:.2f} seconds")
    
    return results



