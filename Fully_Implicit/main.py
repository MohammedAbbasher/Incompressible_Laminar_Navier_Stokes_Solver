from solver import solve_fully_implicit
from postprocessing import plot_results

"""

=====================================================================
FULLY IMPLICIT MONOLITHIC FINITE DIFFERENCE NAVIER–STOKES SOLVER
=====================================================================

This project implements a fully implicit monolithic finite difference
solver for the incompressible Navier–Stokes equations applied to the
2D lid-driven cavity problem.

The solver uses a staggered grid arrangement and solves the continuity
and momentum equations simultaneously within a single coupled linear
system.

Unlike segregated methods such as SIMPLE or projection methods,
this approach solves:

    [ u-momentum ]
    [ v-momentum ]
    [ continuity ]

all together in one monolithic matrix system.

This improves pressure–velocity coupling and provides enhanced
numerical stability for implicit time integration.


Fully_Implicit_Lid_Driven_Cavity/

main.py
│
├── 1. Define simulation parameters
│       • Reynolds number (Re)
│       • domain length (L)
│       • grid resolution (N)
│       • lid velocity (U_lid)
│       • time step (dt)
│       • final simulation time (T)
│
├── 2. Compute derived quantities
│       • grid spacing (h)
│       • kinematic viscosity (ν)
│
├── 3. Call:
│
│       solve_fully_implicit(...)
│
│
▼
solver.py
│
├── 1. Generate staggered-grid storage
│       • pressure field
│       • u-velocity field
│       • v-velocity field
│
├── 2. Call:
│
│       setup_indices(...)
│
│
▼
grid.py
│
├── Generate global indexing system
│
├── Map:
│       • u-velocity unknowns
│       • v-velocity unknowns
│       • pressure unknowns
│
└── Return global sparse matrix indices
│
│
▼
solver.py
│
├── Start fully implicit time loop
│
├── For each nonlinear iteration:
│
│       • Build RHS vector
│       • Apply boundary conditions
│       • Assemble monolithic matrix
│
├── Call:
│
│       build_fully_implicit_matrix(...)
│
│
▼
matrix_assembly.py
│
├── Construct monolithic sparse matrix
│
├── Include:
│       • time derivative terms
│       • viscous diffusion terms
│       • pressure gradient terms
│       • continuity equation
│       • convective Jacobian terms
│
├── Call:
│
│       compute_convective_jacobian(...)
│
│
▼
jacobian.py
│
├── Linearize nonlinear convective terms
│
├── Construct:
│       • J_u matrix
│       • J_v matrix
│
└── Return sparse Jacobian operators
│
│
▼
matrix_assembly.py
│
├── Complete coupled monolithic matrix
│
└── Return global sparse system
│
│
▼
solver.py
│
├── Apply boundary conditions to RHS
│
├── Call:
│
│       apply_boundary_conditions_to_rhs(...)
│
│
▼
boundary_conditions.py
│
├── Impose lid-driven cavity conditions
│
├── Top wall:
│       • u = U_lid
│       • v = 0
│
├── Remaining walls:
│       • u = 0
│       • v = 0
│
└── Fix pressure gauge point
│
│
▼
solver.py
│
├── Optional Newton nonlinear correction
│
├── Call:
│
│       compute_convective_residual(...)
│
│
▼
residuals.py
│
├── Compute nonlinear convective residuals
│
├── Evaluate:
│       • (u · ∇)u
│       • (u · ∇)v
│
├── Used for:
│       • Picard iteration
│       • Newton linearization
│
└── Return convective correction fields
│
│
▼
solver.py
│
├── Solve monolithic sparse linear system
│
├── Simultaneously solve:
│       • continuity equation
│       • u-momentum equation
│       • v-momentum equation
│
├── Update:
│       • velocity field
│       • pressure field
│
├── Compute diagnostics
│       • divergence history
│       • velocity history
│       • convergence information
│
└── Repeat until final time reached
│
│
▼
main.py
│
├── Call:
│
│       plot_results(...)
│
│
▼
postprocessing.py
│
├── Interpolate staggered velocities
│       to cell centers
│
├── Compute:
│       • velocity magnitude
│       • streamlines
│       • divergence history
│       • Ghia benchmark error
│
├── Compare numerical solution against:
│
│       Ghia et al. (1982)
│
└── Plot:
        • velocity contours
        • streamlines
        • centerline velocity profiles
        • benchmark comparisons
        • divergence convergence
        • numerical error history



---------------------------------------------------------------------
NUMERICAL FEATURES
---------------------------------------------------------------------

• Finite Difference discretization
• Fully implicit time integration
• Monolithic coupled formulation
• Staggered Cartesian grid
• Pressure–velocity coupling solved simultaneously
• Picard / Newton nonlinear iterations
• Sparse matrix assembly
• Direct or iterative sparse linear solvers
• Comparison against Ghia et al. (1982) benchmark data

---------------------------------------------------------------------
GOVERNING EQUATIONS
---------------------------------------------------------------------

The incompressible Navier–Stokes equations are solved:

Continuity equation:

    ∇ · u = 0

Momentum equation:

    ∂u/∂t + (u · ∇)u
    =
    -∇p + ν∇²u

where:

    u  → velocity vector
    p  → pressure
    ν  → kinematic viscosity

---------------------------------------------------------------------
GRID ARRANGEMENT
---------------------------------------------------------------------

A staggered grid is used:

• Pressure stored at cell centers
• u-velocity stored at vertical faces
• v-velocity stored at horizontal faces

This avoids checkerboard pressure oscillations and improves stability.

---------------------------------------------------------------------
MONOLITHIC FORMULATION
---------------------------------------------------------------------

The solver constructs one global sparse matrix containing:

• Time derivative terms
• Convective terms
• Diffusion terms
• Pressure gradient terms
• Continuity equation

The resulting system has the structure:

    [ Auu  Auv  Gx ] [u]   [Ru]
    [ Avu  Avv  Gy ] [v] = [Rv]
    [ Dx   Dy    0 ] [p]   [Rc]

where:

    G  → pressure gradient operators
    D  → divergence operators

This monolithic strategy strongly couples pressure and velocity.

---------------------------------------------------------------------
NONLINEAR TREATMENT
---------------------------------------------------------------------

The convective terms are nonlinear.

Two nonlinear approaches are supported:

1. Picard iteration
   • Linearizes convection using previous iteration velocities

2. Newton linearization
   • Uses a convective Jacobian matrix

---------------------------------------------------------------------
BOUNDARY CONDITIONS
---------------------------------------------------------------------

Lid-driven cavity boundary conditions:

Top wall:
    u = U_lid
    v = 0

Other walls:
    u = 0
    v = 0

Pressure:
    One pressure node is fixed to remove singularity.

---------------------------------------------------------------------
POSTPROCESSING
---------------------------------------------------------------------

The code generates:

• Velocity magnitude contours
• Streamlines
• Centerline velocity comparisons
• Comparison with Ghia benchmark data
• Divergence history
• Numerical error plots

---------------------------------------------------------------------
MAIN INPUT PARAMETERS
---------------------------------------------------------------------

Defined inside main.py:

    Re      → Reynolds number
    N       → Number of grid points
    L       → Domain length
    U_lid   → Lid velocity
    dt      → Time step
    final_time       → Final simulation time

Derived quantities:

    h       → Grid spacing
    ν       → Kinematic viscosity

---------------------------------------------------------------------
CODE STRUCTURE
---------------------------------------------------------------------

main.py
    Driver script and simulation setup

solver.py
    Fully implicit monolithic solver

grid.py
    Index mapping for staggered variables

matrix_assembly.py
    Sparse matrix construction

jacobian.py
    Convective Jacobian assembly

residuals.py
    Convective residual computation

boundary_conditions.py
    Boundary condition enforcement

postprocessing.py
    Visualization and benchmark comparison

---------------------------------------------------------------------
REFERENCE
---------------------------------------------------------------------

Ghia, U., Ghia, K. N., & Shin, C. T. (1982)

"High-Re solutions for incompressible flow using the
Navier–Stokes equations and a multigrid method"

Journal of Computational Physics, 48, 387–411.
"""

if __name__ == "__main__":

    print("=" * 60)
    print("FULLY IMPLICIT MONOLITHIC SOLVER TEST")
    print("=" * 60)
    
    # ============================================
    # PHYSICAL PARAMETERS
    # ============================================
    Re=100
    L = 1.0
    U_lid = 1.0
    
    N=64
    dt=0.1
    final_time=60
       
    h = L / (N - 1)
    nu = U_lid*L / Re
      
    # ============================================
    # SOLVER PARAMETERS
    # ============================================
    method='picard'
    max_nonlinear_iter=3
    tol=1e-4
    use_direct=True
    
    # ============================================
    # SOLVER
    # ============================================
   
    (
        u,
        v,
        p,
        h,
        U_lid,
        time_history,
        divergence_history,
        u_max_history
    ) = solve_fully_implicit(
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
    )
        
    plot_results(
        u,
        v,
        p,
        N,
        L,
        Re,
        U_lid,
        time_history,
        divergence_history
    )
