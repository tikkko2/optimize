import cupy as cp
import numpy as np
from scipy.optimize import minimize
import time
import sys

# Thomas Algorithm for tridiagonal system (GPU version)
def thomas_algorithm(a, b, c, d):
    n = len(b)
    c_prime = cp.zeros(n-1)
    d_prime = cp.zeros(n)
    
    # Forward elimination
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denom
    
    d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / (b[n-1] - a[n-2] * c_prime[n-2])
    
    # Back substitution
    x = cp.zeros(n)
    x[n-1] = d_prime[n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x

# Test Thomas Algorithm
def test_thomas_algorithm():
    print("=" * 50)
    print("TESTING THOMAS ALGORITHM")
    print("=" * 50)
    
    a = cp.array([1, 1, 1], dtype=cp.float64)
    b = cp.array([4, 4, 4, 4], dtype=cp.float64)
    c = cp.array([1, 1, 1], dtype=cp.float64)
    d = cp.array([5, 6, 6, 5], dtype=cp.float64)
    
    print("Setting up test system...")
    start_time = time.time()
    
    x = thomas_algorithm(a, b, c, d)
    x_np = cp.asnumpy(x)
    expected = np.linalg.solve(
        np.array([
            [4, 1, 0, 0],
            [1, 4, 1, 0],
            [0, 1, 4, 1],
            [0, 0, 1, 4]
        ]), cp.asnumpy(d))
    
    end_time = time.time()
    
    print(f"Thomas algorithm completed in {end_time - start_time:.4f} seconds")
    print("Computed:", x_np)
    print("Expected:", expected)
    error = np.linalg.norm(x_np - expected)
    print(f"Error: {error:.2e}")
    
    success = np.allclose(x_np, expected)
    print(f"Test result: {'PASSED' if success else 'FAILED'}")
    return success

# Initialize time-dependent coefficients
def init_time_dependent_coeffs(a_base, b_base, c_base, t, t_max, verbose=True):
    """
    Returns a(t), b(t), c(t) as 3D arrays of shape (n, m, nt)
    a_base, b_base, c_base: baseline coefficients (n, m)
    t: time array (nt,)
    """
    if verbose:
        print("Initializing time-dependent coefficients...")
    
    nt = len(t)
    n, m = a_base.shape
    a = cp.zeros((n, m, nt))
    b = cp.zeros((n, m, nt))
    c = cp.zeros((n, m, nt))
    
    for k in range(nt):
        if verbose and k % (nt // 10) == 0:
            print(f"  Processing time step {k}/{nt} ({100*k/nt:.1f}%)")
        
        t_k = t[k]
        a[:, :, k] = a_base * (1 + 0.5 * cp.sin(np.pi * t_k / t_max))
        b[:, :, k] = b_base * (1 + 0.5 * cp.cos(np.pi * t_k / t_max))
        c[:, :, k] = c_base * (1 + 0.5 * cp.sin(2 * np.pi * t_k / t_max))
    
    if verbose:
        print(f"  Completed coefficient initialization for {nt} time steps")
    
    return a, b, c

# Progress bar function
def print_progress_bar(iteration, total, prefix='Progress', suffix='Complete', length=40):
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='', flush=True)
    if iteration == total:
        print()  # New line on completion

# Governing equation solver using Lie splitting (GPU version)
def solve_governing_equation(u0, a_base, b_base, c_base, dt, dx, dy, n, m, t_max, verbose=True):
    if verbose:
        print("Starting governing equation solver...")
    
    u = u0.copy()
    nt = int(t_max / dt)
    t = cp.linspace(0, t_max, nt)
    
    if verbose:
        print(f"Time integration: {nt} steps, dt={dt}, t_max={t_max}")
    
    # Initialize time-dependent coefficients
    start_time = time.time()
    a, b, c = init_time_dependent_coeffs(a_base, b_base, c_base, t, t_max, verbose)
    coeff_time = time.time() - start_time
    
    if verbose:
        print(f"Coefficient initialization completed in {coeff_time:.3f} seconds")
        print("Starting time stepping...")
    
    solve_start_time = time.time()
    
    for k in range(nt):
        if verbose and (k % max(1, nt // 20) == 0 or k == nt - 1):
            elapsed = time.time() - solve_start_time
            if k > 0:
                eta = elapsed * (nt - k) / k
                print(f"  Time step {k+1}/{nt} ({100*(k+1)/nt:.1f}%) - "
                      f"Elapsed: {elapsed:.2f}s, ETA: {eta:.2f}s")
            else:
                print(f"  Time step {k+1}/{nt} ({100*(k+1)/nt:.1f}%)")
        
        # Use coefficients at time t[k]
        a_k = a[:, :, k]
        b_k = b[:, :, k]
        c_k = c[:, :, k]
        
        # Precompute coefficients for this time step
        ap = (a_k + cp.abs(a_k)) / 2
        am = (a_k - cp.abs(a_k)) / 2
        bp = (b_k + cp.abs(b_k)) / 2
        bm = (b_k - cp.abs(b_k)) / 2
        
        # Step 1: Solve x-direction
        u_x = u.copy()
        for j in range(1, m-1):
            lower = -dt/dx * ap[1:-1,j]
            diag = 1 + dt/dx * (ap[1:-1,j] - am[1:-1,j])
            upper = dt/dx * am[1:-1,j]
            rhs = u[1:-1,j] + dt * c_k[1:-1,j] * (1 - u[1:-1,j])
            
            u_x[1:-1,j] = thomas_algorithm(lower, diag, upper, rhs)
        
        # Step 2: Solve y-direction
        u_y = u_x.copy()
        for i in range(1, n-1):
            lower = -dt/dy * bp[i,1:-1]
            diag = 1 + dt/dy * (bp[i,1:-1] - bm[i,1:-1])
            upper = dt/dy * bm[i,1:-1]
            rhs = u_x[i,1:-1] + dt * c_k[i,1:-1] * (1 - u_x[i,1:-1])
            
            u_y[i,1:-1] = thomas_algorithm(lower, diag, upper, rhs)
        
        u = u_y.copy()
    
    total_solve_time = time.time() - solve_start_time
    
    if verbose:
        print(f"\nTime stepping completed in {total_solve_time:.3f} seconds")
        print(f"Average time per step: {total_solve_time/nt:.4f} seconds")
    
    return u

# Test Governing Equation
def test_governing_equation():
    print("\n" + "=" * 50)
    print("TESTING GOVERNING EQUATION")
    print("=" * 50)
    
    n, m = 10, 10
    dx, dy = 0.1, 0.1
    dt = 0.001
    t_max = 0.1
    
    print(f"Grid size: {n}x{m}")
    print(f"Spatial resolution: dx={dx}, dy={dy}")
    print(f"Time parameters: dt={dt}, t_max={t_max}")
    
    u0 = cp.zeros((n, m), dtype=cp.float64)
    uT = cp.ones((n, m), dtype=cp.float64) * 0.5
    a_base = cp.ones((n, m), dtype=cp.float64) * 0.1
    b_base = cp.ones((n, m), dtype=cp.float64) * 0.1
    c_base = cp.ones((n, m), dtype=cp.float64) * 0.1
    
    u0[0,:], u0[-1,:], u0[:,0], u0[:,-1] = uT[0,:], uT[-1,:], uT[:,0], uT[:,-1]
    
    print("Starting simulation...")
    start_time = time.time()
    
    u = solve_governing_equation(u0, a_base, b_base, c_base, dt, dx, dy, n, m, t_max, verbose=True)
    
    end_time = time.time()
    
    u_np = cp.asnumpy(u)
    uT_np = cp.asnumpy(uT)
    
    print(f"\nSimulation completed in {end_time - start_time:.3f} seconds")
    print("Solution shape:", u.shape)
    print("Solution min/max:", u_np.min(), u_np.max())
    
    boundary_check = (cp.allclose(u[0,:], uT[0,:]) and 
                     cp.allclose(u[-1,:], uT[-1,:]) and 
                     cp.allclose(u[:,0], uT[:,0]) and 
                     cp.allclose(u[:,-1], uT[:,-1]))
    
    print("Boundary conditions preserved:", boundary_check)
    
    bounds_check = cp.all(cp.logical_and(u >= 0, u <= 1)).get()
    print("Solution within bounds [0,1]:", bounds_check)
    
    success = bounds_check and boundary_check
    print(f"Test result: {'PASSED' if success else 'FAILED'}")
    
    return success

# Global variables for optimization progress tracking
opt_iteration = 0
opt_start_time = 0
best_loss = float('inf')

# Loss function with progress tracking
def loss_function(params, u0, uT, dt, dx, dy, n, m, t_max):
    global opt_iteration, opt_start_time, best_loss
    
    opt_iteration += 1
    
    size = n * m
    a_base = cp.array(params[:size]).reshape(n, m)
    b_base = cp.array(params[size:2*size]).reshape(n, m)
    c_base = cp.array(params[2*size:]).reshape(n, m)
    
    print(f"\n--- Optimization Iteration {opt_iteration} ---")
    iter_start_time = time.time()
    
    u = solve_governing_equation(u0, a_base, b_base, c_base, dt, dx, dy, n, m, t_max, verbose=False)
    loss = cp.sum((u - uT) ** 2).get()
    
    iter_time = time.time() - iter_start_time
    elapsed_total = time.time() - opt_start_time
    
    if loss < best_loss:
        best_loss = loss
        improvement = "↓ NEW BEST"
    else:
        improvement = "↑"
    
    print(f"Loss: {loss:.6e} {improvement}")
    print(f"Iteration time: {iter_time:.2f}s, Total elapsed: {elapsed_total:.1f}s")
    print(f"Avg coefficient values - a: {np.mean(params[:size]):.4f}, "
          f"b: {np.mean(params[size:2*size]):.4f}, c: {np.mean(params[2*size:]):.4f}")
    
    return loss

# Initialization functions for u0 and uT
def init_u0(x, y):
    return 0.5 * (1 + cp.sin(np.pi * x) * cp.cos(np.pi * y))

def init_uT(x, y):
    return 0.5 * (1 + cp.cos(np.pi * x) * cp.sin(np.pi * y))

# Test Optimization with time-dependent coefficients
def test_optimization():
    global opt_iteration, opt_start_time, best_loss
    
    print("\n" + "=" * 50)
    print("TESTING OPTIMIZATION")
    print("=" * 50)
    
    n, m = 7, 7
    dx, dy = 0.2, 0.2
    dt = 0.1
    t_max = 1
    
    print(f"Optimization problem size: {n}x{m} grid")
    print(f"Parameters to optimize: {3*n*m}")
    print(f"Spatial resolution: dx={dx}, dy={dy}")
    print(f"Time parameters: dt={dt}, t_max={t_max}")
    
    # Create spatial grid on GPU
    x = cp.linspace(0, 1, n)
    y = cp.linspace(0, 1, m)
    X, Y = cp.meshgrid(x, y, indexing='ij')
    
    print("Initializing fields...")
    # Initialize u0, uT
    u0 = init_u0(X, Y)
    uT = init_uT(X, Y)
    
    # Initialize baseline coefficients
    true_a_base = cp.ones((n, m)) * 0.1
    true_b_base = cp.ones((n, m)) * 0.1
    true_c_base = cp.ones((n, m)) * 0.0
    
    # Apply boundary conditions
    u0[0,:], u0[-1,:], u0[:,0], u0[:,-1] = uT[0,:], uT[-1,:], uT[:,0], uT[:,-1]
    
    # Verify initialization bounds
    u0_np = cp.asnumpy(u0)
    uT_np = cp.asnumpy(uT)
    print("u0 range:", f"[{u0_np.min():.3f}, {u0_np.max():.3f}]")
    print("uT range:", f"[{uT_np.min():.3f}, {uT_np.max():.3f}]")
    
    # Initial guess on CPU for scipy
    initial_params_np = np.ones(3 * n * m) * 0.05
    print(f"Initial parameter guess: {np.mean(initial_params_np):.4f} ± {np.std(initial_params_np):.4f}")
    
    # Reset optimization tracking
    opt_iteration = 0
    opt_start_time = time.time()
    best_loss = float('inf')
    
    print("\nStarting optimization...")
    print("Method: L-BFGS-B")
    print("Max iterations: 100")
    
    # Optimization with bounds and relaxed tolerances
    bounds = [(0, 1)] * (3 * n * m)
    result = minimize(
        loss_function,
        initial_params_np,
        args=(u0, uT, dt, dx, dy, n, m, t_max),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': True}
    )
    
    total_opt_time = time.time() - opt_start_time
    
    print("\n" + "=" * 50)
    print("OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Final loss: {result.fun:.6e}")
    print(f"Total iterations: {result.nit}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Total optimization time: {total_opt_time:.1f} seconds")
    print(f"Average time per iteration: {total_opt_time/result.nit:.2f} seconds")
    
    # Analyze final parameters
    size = n * m
    final_a = result.x[:size]
    final_b = result.x[size:2*size]
    final_c = result.x[2*size:]
    
    print(f"\nFinal parameter statistics:")
    print(f"a coefficients: mean={np.mean(final_a):.4f}, std={np.std(final_a):.4f}")
    print(f"b coefficients: mean={np.mean(final_b):.4f}, std={np.std(final_b):.4f}")
    print(f"c coefficients: mean={np.mean(final_c):.4f}, std={np.std(final_c):.4f}")
    
    success = result.success or result.fun < 1e-8
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
    
    return success

# Run all tests
def run_tests():
    print("RUNNING COMPREHENSIVE TESTS")
    print("GPU version with time-dependent coefficients")
    print("=" * 60)
    
    overall_start_time = time.time()
    
    # Run tests
    thomas_passed = test_thomas_algorithm()
    gov_eq_passed = test_governing_equation()
    opt_passed = test_optimization()
    
    overall_time = time.time() - overall_start_time
    
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)
    print(f"Thomas Algorithm: {'PASSED' if thomas_passed else 'FAILED'}")
    print(f"Governing Equation: {'PASSED' if gov_eq_passed else 'FAILED'}")
    print(f"Optimization: {'PASSED' if opt_passed else 'FAILED'}")
    print(f"\nTotal execution time: {overall_time:.1f} seconds")
    
    all_passed = thomas_passed and gov_eq_passed and opt_passed
    print(f"\nOVERALL RESULT: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    print("GPU-Accelerated PDE Solver with Optimization")
    print("CuPy version:", cp.__version__)
    print("GPU device:", cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
    print()
    
    run_tests()
    cp.cuda.runtime.deviceSynchronize()
    print("\nExecution completed.")
