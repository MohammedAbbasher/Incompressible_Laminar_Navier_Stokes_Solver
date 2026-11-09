from solver import run_simulation
from postprocessing import plot_simple_results


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":

    # =====================================================
    # PARAMETERS
    # =====================================================

    Re = 400.0

    nx = 64
    ny = 64

    dt = 0.001

    alpha = 0.05

    # =====================================================
    # CARTESIAN GRID
    # =====================================================

    print("Testing Cartesian grid...")

    results_cart = run_simulation(
        nx,
        ny,
        Re,
        dt,
        False,     # curvilinear
        alpha,
        phi,
        num_steps=1,
        save_interval=1,        
    )

    # =====================================================
    # CURVILINEAR GRID
    # =====================================================

    print("\nTesting curvilinear grid...")

    results_curv = run_simulation(
        nx,
        ny,
        Re,
        dt,
        True,      # curvilinear
        alpha,
        num_steps=100000,
        save_interval=1000
    )

    # =====================================================
    # PLOT
    # =====================================================

    plot_simple_results(
        results_cart,
        results_curv
    )
