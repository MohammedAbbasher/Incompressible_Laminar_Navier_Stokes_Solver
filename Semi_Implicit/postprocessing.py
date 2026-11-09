"""
Reference solutions for lid-driven cavity flow validation.
Contains benchmark data from Ghia et al. (1982).
"""

def Ghia(Re):
    """Reference solution from Ghia et al. 1982."""
    x = [0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
         0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000]
    y = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
         0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000]

    reference_data = {
        100: {
            'u': [0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662,
                  -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722,
                  0.78871, 0.84123, 1.00000],
            'v': [0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507,
                  0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864,
                  -0.07391, -0.05906, 0.00000]
        },
        400: {
            'u': [0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726,
                  -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 0.55892, 0.61756,
                  0.68439, 0.75837, 1.00000],
            'v': [0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124, 0.30203,
                  0.30174, 0.05186, -0.38598, -0.44993, -0.23827, -0.22847, -0.19254,
                  -0.15663, -0.12146, 0.00000]
        },
        1000: {
            'u': [0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805,
                  -0.10648, -0.06080, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117,
                  0.57492, 0.65928, 1.00000],
            'v': [0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075,
                  0.32235, 0.02526, -0.31966, -0.42665, -0.51500, -0.39188, -0.33714,
                  -0.27669, -0.21388, 0.00000]
        },
        3200: {
            'u': [0.00000, -0.32407, -0.35344, -0.37827, -0.41933, -0.34323, -0.24427,
                  -0.86636, -0.04272, 0.07156, 0.19791, 0.34682, 0.46101, 0.46547,
                  0.48296, 0.53236, 1.00000],
            'v': [0.00000, 0.39560, 0.40917, 0.41906, 0.42768, 0.37119, 0.29030,
                  0.28188, 0.00999, -0.31184, -0.37401, -0.44307, -0.54053, -0.52357,
                  -0.47425, -0.39017, 0.00000]
        },
        5000: {
            'u': [0.00000, -0.41165, -0.42901, -0.43643, -0.40435, -0.33050, -0.22855,
                  -0.07404, -0.03039, 0.08183, 0.20087, 0.33556, 0.46036, 0.45992,
                  0.46120, 0.48223, 1.00000],
            'v': [0.00000, 0.42447, 0.43329, 0.43648, 0.42951, 0.35368, 0.28066,
                  0.27280, 0.00945, -0.30018, -0.36214, -0.41442, -0.52876, -0.55408,
                  -0.55069, -0.49774, 0.00000]
        },
        7500: {
            'u': [0.00000, -0.43154, -0.43590, -0.43025, -0.38324, -0.32393, -0.23176,
                  -0.07503, -0.03800, 0.08342, 0.20591, 0.34228, 0.47167, 0.47323,
                  0.47048, 0.47244, 1.00000],
            'v': [0.00000, 0.43979, 0.44030, 0.43564, 0.41824, 0.35060, 0.28117,
                  0.27348, 0.00824, -0.30448, -0.36213, -0.41050, -0.48590, -0.52347,
                  -0.55216, -0.53858, 0.00000]
        },
        10000: {
            'u': [0.00000, -0.42735, -0.42537, -0.41657, -0.38000, -0.32709, -0.23186,
                  -0.07540, 0.03111, 0.08344, 0.20673, 0.34635, 0.47804, 0.48070,
                  0.47783, 0.47221, 1.00000],
            'v': [0.00000, 0.43983, 0.43733, 0.43124, 0.41487, 0.35070, 0.28003,
                  0.27224, 0.00831, -0.30719, -0.36737, -0.41496, -0.45863, -0.49099,
                  -0.52987, -0.54302, 0.00000]
        }
    }

    data = reference_data.get(Re, {'u': None, 'v': None})
    return {'x': x, 'y': y, 'u': data['u'], 'v': data['v']}


from scipy.interpolate import griddata   # add at the top of the file (or inside the function)
"""
def plot_simple_results(solver_cart, solver_curv, results_cart, results_curv):
    Plot simple results with Ghia reference data and interpolated simulation points.
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Get latest results
    u_cart = results_cart['u'][-1]
    v_cart = results_cart['v'][-1]
    u_curv = results_curv['u'][-1]
    v_curv = results_curv['v'][-1]
    
    # Velocity magnitudes
    vel_mag_cart = np.sqrt(u_cart**2 + v_cart**2)
    vel_mag_curv = np.sqrt(u_curv**2 + v_curv**2)
    
    # Plot 1: Cartesian velocity magnitude
    im1 = axes[0, 0].contourf(solver_cart.x, solver_cart.y, vel_mag_cart, levels=20, cmap='jet')
    axes[0, 0].set_title('Cartesian: Velocity Magnitude')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Curvilinear velocity magnitude
    im2 = axes[0, 1].contourf(solver_curv.x, solver_curv.y, vel_mag_curv, levels=20, cmap='jet')
    axes[0, 1].set_title('Curvilinear: Velocity Magnitude')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Curvilinear grid
    axes[0, 2].plot(solver_curv.x, solver_curv.y, 'k-', alpha=0.5, linewidth=0.5)
    axes[0, 2].plot(solver_curv.x.T, solver_curv.y.T, 'k-', alpha=0.5, linewidth=0.5)
    axes[0, 2].set_title('Curvilinear Grid')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    axes[0, 2].set_aspect('equal')
    
    # Plot 4: Centerline u-velocity with Ghia reference data
    mid_x = solver_cart.nx // 2
    axes[1, 0].plot(solver_cart.y[mid_x, :], u_cart[mid_x, :], 'b-', label='Cartesian (full line)')
    axes[1, 0].plot(solver_curv.y[mid_x, :], u_curv[mid_x, :], 'r--', label='Curvilinear (full line)')
    
    # --- Fetch Ghia reference data for the current Reynolds number ---
    ref_data = Ghia(solver_cart.Re)          # Re=100 in the test
    ref_y = np.array(ref_data['y'])           # vertical coordinates for u
    ref_u = np.array(ref_data['u'])           # reference u at those y
    
    # --- Interpolate simulation u to the exact reference points (x=0.5, y_ref) ---
    # For Cartesian grid (uniform, aligned)
    points_cart = np.column_stack((solver_cart.x.ravel(), solver_cart.y.ravel()))
    u_cart_flat = u_cart.ravel()
    u_cart_interp = griddata(points_cart, u_cart_flat, (0.5, ref_y), method='linear')
    
    # For curvilinear grid (non‑uniform)
    points_curv = np.column_stack((solver_curv.x.ravel(), solver_curv.y.ravel()))
    u_curv_flat = u_curv.ravel()
    u_curv_interp = griddata(points_curv, u_curv_flat, (0.5, ref_y), method='linear')
    
    # Plot the interpolated values as markers
    axes[1, 0].plot(ref_y, u_cart_interp, 'bs', label='Cartesian (at ref. y)', markersize=4)
    axes[1, 0].plot(ref_y, u_curv_interp, 'r^', label='Curvilinear (at ref. y)', markersize=4)
    axes[1, 0].plot(ref_y, ref_u, 'ko', label='Ghia et al. (1982)', markersize=4)
    
    axes[1, 0].set_title('Centerline u-velocity at x=L/2')
    axes[1, 0].set_xlabel('y')
    axes[1, 0].set_ylabel('u')
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: CFL history
    axes[1, 1].plot(results_cart['time'], results_cart['cfl'], 'b-', label='Cartesian')
    axes[1, 1].plot(results_curv['time'], results_curv['cfl'], 'r-', label='Curvilinear')
    axes[1, 1].set_title('CFL Number History')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('CFL')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 6: Divergence history
    axes[1, 2].semilogy(results_cart['time'], results_cart['divergence'], 'b-', label='Cartesian')
    axes[1, 2].semilogy(results_curv['time'], results_curv['divergence'], 'r-', label='Curvilinear')
    axes[1, 2].set_title('Maximum Divergence History')
    axes[1, 2].set_xlabel('Time')
    axes[1, 2].set_ylabel('Max |∇·u|')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
"""
"""
#For only Cartesian results
def plot_simple_results(solver_cart, solver_curv, results_cart, results_curv):
     Plot simple results with Ghia reference data and interpolated simulation points.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Get latest results
    u_cart = results_cart['u'][-1]
    v_cart = results_cart['v'][-1]
   # u_curv = results_curv['u'][-1]
    #v_curv = results_curv['v'][-1]
    
    # Velocity magnitudes
    vel_mag_cart = np.sqrt(u_cart**2 + v_cart**2)
   # vel_mag_curv = np.sqrt(u_curv**2 + v_curv**2)
    
    # Plot 1: Cartesian velocity magnitude
    im1 = axes[0, 0].contourf(solver_cart.x, solver_cart.y, vel_mag_cart, levels=20, cmap='jet')
    axes[0, 0].set_title('Cartesian: Velocity Magnitude')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0])
    
   # Plot 2: Streamlines (with transposed arrays for ij → xy conversion)
    axes[0, 1].streamplot(solver_cart.x.T, solver_cart.y.T, u_cart.T, v_cart.T,
                          color='blue', density=2, linewidth=1, arrowsize=1)
    axes[0, 1].set_title('Streamlines')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    
    #Here I should plot the vorticity lines instead 
    # # Plot 2: Curvilinear velocity magnitude
    # im2 = axes[0, 1].contourf(solver_curv.x, solver_curv.y, vel_mag_curv, levels=20, cmap='jet')
    # axes[0, 1].set_title('Curvilinear: Velocity Magnitude')
    # axes[0, 1].set_xlabel('x')
    # axes[0, 1].set_ylabel('y')
    # axes[0, 1].set_aspect('equal')
    # plt.colorbar(im2, ax=axes[0, 1])
    
    # # Plot 3: Curvilinear grid
    # axes[0, 2].plot(solver_curv.x, solver_curv.y, 'k-', alpha=0.5, linewidth=0.5)
    # axes[0, 2].plot(solver_curv.x.T, solver_curv.y.T, 'k-', alpha=0.5, linewidth=0.5)
    # axes[0, 2].set_title('Curvilinear Grid')
    # axes[0, 2].set_xlabel('x')
    # axes[0, 2].set_ylabel('y')
    # axes[0, 2].set_aspect('equal')
    
    # Plot 3: Centerline u-velocity with Ghia reference data
    mid_x = solver_cart.nx // 2
    axes[1, 0].plot(solver_cart.y[mid_x, :], u_cart[mid_x, :], 'b-', label='Cartesian (full line)')
    #axes[1, 0].plot(solver_curv.y[mid_x, :], u_curv[mid_x, :], 'r--', label='Curvilinear (full line)')
    
    # --- Fetch Ghia reference data for the current Reynolds number ---
    ref_data = Ghia(solver_cart.Re)          # Re=100 in the test
    ref_y = np.array(ref_data['y'])           # vertical coordinates for u
    ref_u = np.array(ref_data['u'])           # reference u at those y
    
    # --- Interpolate simulation u to the exact reference points (x=0.5, y_ref) ---
    # For Cartesian grid (uniform, aligned)
    points_cart = np.column_stack((solver_cart.x.ravel(), solver_cart.y.ravel()))
    u_cart_flat = u_cart.ravel()
    u_cart_interp = griddata(points_cart, u_cart_flat, (0.5, ref_y), method='linear')
    
    # # For curvilinear grid (non‑uniform)
    # points_curv = np.column_stack((solver_curv.x.ravel(), solver_curv.y.ravel()))
    # u_curv_flat = u_curv.ravel()
    # u_curv_interp = griddata(points_curv, u_curv_flat, (0.5, ref_y), method='linear')
    
    # Plot the interpolated values as markers
    axes[1, 0].plot(ref_y, u_cart_interp, 'bs', label='Cartesian (at ref. y)', markersize=4)
   # axes[1, 0].plot(ref_y, u_curv_interp, 'r^', label='Curvilinear (at ref. y)', markersize=4)
    axes[1, 0].plot(ref_y, ref_u, 'ko', label='Ghia et al. (1982)', markersize=4)
    
    axes[1, 0].set_title('Centerline u-velocity at x=L/2')
    axes[1, 0].set_xlabel('y')
    axes[1, 0].set_ylabel('u')
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, alpha=0.3)
    
    #Should plot the same here for v velocity
    
    # Plot 4: Centerline v-velocity at y = L/2 with Ghia reference data
    mid_y = solver_cart.ny // 2
    axes[1, 1].plot(solver_cart.x[:, mid_y], v_cart[:, mid_y], 'b-', label='Cartesian (full line)')
    # axes[1, 1].plot(solver_curv.x[:, mid_y], v_curv[:, mid_y], 'r--', label='Curvilinear (full line)')
    
    # --- Fetch Ghia reference data for the current Reynolds number ---
    ref_data = Ghia(solver_cart.Re)          # Re = 100 (or whatever you set)
    ref_x = np.array(ref_data['x'])           # horizontal coordinates for v
    ref_v = np.array(ref_data['v'])           # reference v at those x
    
    # --- Interpolate simulation v to the exact reference points (x_ref, y = 0.5) ---
    # For Cartesian grid (uniform, aligned)
    points_cart = np.column_stack((solver_cart.x.ravel(), solver_cart.y.ravel()))
    v_cart_flat = v_cart.ravel()
    v_cart_interp = griddata(points_cart, v_cart_flat, (ref_x, 0.5), method='linear')
    
    # For curvilinear grid (non‑uniform) – uncomment when ready
    # points_curv = np.column_stack((solver_curv.x.ravel(), solver_curv.y.ravel()))
    # v_curv_flat = v_curv.ravel()
    # v_curv_interp = griddata(points_curv, v_curv_flat, (ref_x, 0.5), method='linear')
    
    # Plot the interpolated values as markers
    axes[1, 1].plot(ref_x, v_cart_interp, 'bs', label='Cartesian (at ref. x)', markersize=4)
    # axes[1, 1].plot(ref_x, v_curv_interp, 'r^', label='Curvilinear (at ref. x)', markersize=4)
    axes[1, 1].plot(ref_x, ref_v, 'ko', label='Ghia et al. (1982)', markersize=4)
    
    axes[1, 1].set_title('Centerline v-velocity at y = L/2')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('v')
    axes[1, 1].legend(loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
"""    
#For only Curvilinear results
def plot_simple_results(solver_cart, solver_curv, results_cart, results_curv):
        """ Plot simple results with Ghia reference data and interpolated simulation points."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Get latest results
        u_curv = results_curv['u'][-1]
        v_curv = results_curv['v'][-1]
        
        # Velocity magnitudes
        vel_mag_curv = np.sqrt(u_curv**2 + v_curv**2)
        
        # Plot 1: Curvilinear velocity magnitude
        im1 = axes[0, 0].contourf(solver_curv.x, solver_curv.y, vel_mag_curv, levels=20, cmap='jet')
        axes[0, 0].set_title('Curvilinear: Velocity Magnitude')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, 0])
        
       # Plot 2: Streamlines (with transposed arrays for ij → xy conversion)
        axes[0, 1].streamplot(solver_cart.x.T, solver_cart.y.T, u_curv.T, v_curv.T,
                              color='blue', density=2, linewidth=1, arrowsize=1)
        axes[0, 1].set_title('Streamlines')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        
        
        # # Plot 3: Curvilinear grid
        # axes[0, 2].plot(solver_curv.x, solver_curv.y, 'k-', alpha=0.5, linewidth=0.5)
        # axes[0, 2].plot(solver_curv.x.T, solver_curv.y.T, 'k-', alpha=0.5, linewidth=0.5)
        # axes[0, 2].set_title('Curvilinear Grid')
        # axes[0, 2].set_xlabel('x')
        # axes[0, 2].set_ylabel('y')
        # axes[0, 2].set_aspect('equal')
        
        # Plot 3: Centerline u-velocity with Ghia reference data
        mid_x = solver_curv.nx // 2
        axes[1, 0].plot(solver_cart.y[mid_x, :], u_curv[mid_x, :], 'r-', label='Curvilinear (full line)')
       
        # --- Fetch Ghia reference data for the current Reynolds number ---
        ref_data = Ghia(solver_cart.Re)          # Re=100 in the test
        ref_y = np.array(ref_data['y'])           # vertical coordinates for u
        ref_u = np.array(ref_data['u'])           # reference u at those y
        
        # --- Interpolate simulation u to the exact reference points (x=0.5, y_ref) ---
              
        # For curvilinear grid (non‑uniform)
        points_curv = np.column_stack((solver_curv.x.ravel(), solver_curv.y.ravel()))
        u_curv_flat = u_curv.ravel()
        u_curv_interp = griddata(points_curv, u_curv_flat, (0.5, ref_y), method='linear')
        
        # Plot the interpolated values as markers
        
        axes[1, 0].plot(ref_y, u_curv_interp, 'r^', label='Curvilinear (at ref. y)', markersize=4)
        axes[1, 0].plot(ref_y, ref_u, 'ko', label='Ghia et al. (1982)', markersize=4)
        
        axes[1, 0].set_title('Centerline u-velocity at x=L/2')
        axes[1, 0].set_xlabel('y')
        axes[1, 0].set_ylabel('u')
        axes[1, 0].legend(loc='best')
        axes[1, 0].grid(True, alpha=0.3)
        
        #Should plot the same here for v velocity
        
        # Plot 4: Centerline v-velocity at y = L/2 with Ghia reference data
        mid_y = solver_curv.ny // 2      
        axes[1, 1].plot(solver_cart.x[:, mid_y], v_curv[:, mid_y], 'r--', label='Curvilinear (full line)')
        
        # --- Fetch Ghia reference data for the current Reynolds number ---
        ref_data = Ghia(solver_cart.Re)          # Re = 100 (or whatever you set)
        ref_x = np.array(ref_data['x'])           # horizontal coordinates for v
        ref_v = np.array(ref_data['v'])           # reference v at those x
        
        # --- Interpolate simulation v to the exact reference points (x_ref, y = 0.5) ---
        
        #For curvilinear grid (non‑uniform) – uncomment when ready
        points_curv = np.column_stack((solver_curv.x.ravel(), solver_curv.y.ravel()))
        v_curv_flat = v_curv.ravel()
        v_curv_interp = griddata(points_curv, v_curv_flat, (ref_x, 0.5), method='linear')
        
        # Plot the interpolated values as markers
        #axes[1, 1].plot(ref_x, v_cart_interp, 'bs', label='Cartesian (at ref. x)', markersize=4)
        axes[1, 1].plot(ref_x, v_curv_interp, 'r^', label='Curvilinear (at ref. x)', markersize=4)
        axes[1, 1].plot(ref_x, ref_v, 'ko', label='Ghia et al. (1982)', markersize=4)
        
        axes[1, 1].set_title('Centerline v-velocity at y = L/2')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('v')
        axes[1, 1].legend(loc='best')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
