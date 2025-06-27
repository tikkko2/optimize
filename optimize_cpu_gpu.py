import cupy as cp
import numpy as np
from scipy.optimize import minimize
import time
import sys

# Thomas Algorithm for tridiagonal system (GPU version)
def thomas_algorithm(a, b, c, d):
    """
    Solves a tridiagonal system Ax = d on GPU using CuPy
    a: lower diagonal (CuPy array)
    b: main diagonal (CuPy array)
    c: upper diagonal (CuPy array)
    d: right-hand side (CuPy array)
    Returns: solution x (CuPy array)
    """
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
    print("Testing Thomas Algorithm...")
    a = cp.array([1, 1, 1], dtype=cp.float64)  # lower diagonal
    b = cp.array([4, 4, 4, 4], dtype=cp.float64)  # main diagonal
    c = cp.array([1, 1, 1], dtype=cp.float64)  # upper diagonal
    d = cp.array([5, 6, 6, 5], dtype=cp.float64)  # RHS
    
    x = thomas_algorithm(a, b, c, d)
    
    # Convert to NumPy for comparison with CPU solution
    x_np = cp.asnumpy(x)
    expected = np.linalg.solve(
        np.array([
            [4, 1, 0, 0],
            [1, 4, 1, 0],
            [0, 1, 4, 1],
            [0, 0, 1, 4]
        ]), cp.asnumpy(d))
    
    print("Thomas Algorithm Test:")
    print("Computed:", x_np)
    print("Expected:", expected)
    print("Error:", np.linalg.norm(x_np - expected))
    return np.allclose(x_np, expected)

# Governing equation solver using Lie splitting (GPU version) with progress tracking
def solve_governing_equation(u0, a, b, c, dt, dx, dy, n, m, t_max, show_progress=True):
    """
    Solves the governing equation using Lie splitting on GPU
    u0, a, b, c: CuPy arrays
    """
    u = u0.copy()  # CuPy array copy
    nt = int(t_max / dt)
    
    # Precompute coefficients on GPU
    ap = (a + cp.abs(a)) / 2
    am = (a - cp.abs(a)) / 2
    bp = (b + cp.abs(b)) / 2
    bm = (b - cp.abs(b)) / 2
    
    if show_progress:
        print(f"Starting time integration: {nt} time steps")
        start_time = time.time()
        
    # Progress tracking variables
    progress_interval = max(1, nt // 20)  # Show progress every 5%
    
    for step in range(nt):
        # Step 1: Solve x-direction
        u_x = u.copy()
        for j in range(1, m-1):
            # Set up tridiagonal system
            lower = -dt/dx * ap[1:-1,j]
            diag = 1 + dt/dx * (ap[1:-1,j] - am[1:-1,j])
            upper = dt/dx * am[1:-1,j]
            rhs = u[1:-1,j] + dt * c[1:-1,j] * (1 - u[1:-1,j])
            
            u_x[1:-1,j] = thomas_algorithm(lower, diag, upper, rhs)
        
        # Step 2: Solve y-direction
        u_y = u_x.copy()
        for i in range(1, n-1):
            lower = -dt/dy * bp[i,1:-1]
            diag = 1 + dt/dy * (bp[i,1:-1] - bm[i,1:-1])
            upper = dt/dy * bm[i,1:-1]
            rhs = u_x[i,1:-1] + dt * c[i,1:-1] * (1 - u_x[i,1:-1])
            
            u_y[i,1:-1] = thomas_algorithm(lower, diag, upper, rhs)
        
        u = u_y.copy()
        
        # Progress reporting
        if show_progress and (step + 1) % progress_interval == 0:
            progress = (step + 1) / nt * 100
            elapsed = time.time() - start_time
            eta = elapsed / (step + 1) * (nt - step - 1)
            print(f"Time integration progress: {progress:.1f}% "
                  f"({step + 1}/{nt}) - "
                  f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
            sys.stdout.flush()
    
    if show_progress:
        total_time = time.time() - start_time
        print(f"Time integration completed in {total_time:.2f}s")
    
    return u

# Test Governing Equation
def test_governing_equation():
    print("\nTesting Governing Equation...")
    n, m = 10, 10
    dx, dy = 0.1, 0.1
    dt = 0.05
    t_max = 0.1
    
    # Initialize test data on GPU
    u0 = cp.zeros((n, m), dtype=cp.float64)
    uT = cp.ones((n, m), dtype=cp.float64) * 0.5
    a = cp.ones((n, m), dtype=cp.float64) * 0.1
    b = cp.ones((n, m), dtype=cp.float64) * 0.1
    c = cp.ones((n, m), dtype=cp.float64) * 0.1
    
    # Apply boundary conditions
    u0[0,:], u0[-1,:], u0[:,0], u0[:,-1] = uT[0,:], uT[-1,:], uT[:,0], uT[:,-1]
    
    u = solve_governing_equation(u0, a, b, c, dt, dx, dy, n, m, t_max, show_progress=True)
    
    # Convert to NumPy for printing
    u_np = cp.asnumpy(u)
    uT_np = cp.asnumpy(uT)
    
    print("Governing Equation Test:")
    print("Solution shape:", u.shape)
    print("Solution min/max:", u_np.min(), u_np.max())
    print("Boundary conditions preserved:", 
          cp.allclose(u[0,:], uT[0,:]) and 
          cp.allclose(u[-1,:], uT[-1,:]) and 
          cp.allclose(u[:,0], uT[:,0]) and 
          cp.allclose(u[:,-1], uT[:,-1]))
    
    return cp.all(cp.logical_and(u >= 0, u <= 1)).get()

# Global variables for optimization progress tracking
opt_iteration = 0
opt_start_time = None
opt_best_loss = float('inf')

# Loss function with progress tracking
def loss_function(params, u0, uT, dt, dx, dy, n, m, t_max):
    """
    Computes loss on GPU, converts to NumPy for scipy
    params: NumPy array (CPU)
    u0, uT: CuPy arrays
    """
    global opt_iteration, opt_start_time, opt_best_loss
    
    if opt_start_time is None:
        opt_start_time = time.time()
    
    opt_iteration += 1
    
    size = n * m
    # Convert params to CuPy
    a = cp.array(params[:size]).reshape(n, m)
    b = cp.array(params[size:2*size]).reshape(n, m)
    c = cp.array(params[2*size:]).reshape(n, m)
    
    # Solve with reduced progress output for optimization
    u = solve_governing_equation(u0, a, b, c, dt, dx, dy, n, m, t_max, show_progress=False)
    
    # Convert to NumPy for loss computation
    loss = cp.sum((u - uT) ** 2).get()  # .get() moves to CPU
    
    # Track best loss and show progress
    if loss < opt_best_loss:
        opt_best_loss = loss
        
    elapsed = time.time() - opt_start_time
    print(f"Optimization iteration {opt_iteration:3d}: "
          f"Loss = {loss:.6e}, Best = {opt_best_loss:.6e}, "
          f"Time = {elapsed:.1f}s")
    sys.stdout.flush()
    
    return loss

# Test Optimization with progress tracking
def test_optimization():
    global opt_iteration, opt_start_time, opt_best_loss
    
    print("\nTesting Optimization...")
    
    # Reset global variables
    opt_iteration = 0
    opt_start_time = None
    opt_best_loss = float('inf')
    
    n, m = 8, 8
    dx, dy = 1, 1
    dt = 0.01
    t_max = 1
    
    print(f"Problem size: {n}x{m} grid, {3*n*m} parameters to optimize")
    
    # Initialize test data on GPU
    u0 = cp.random.rand(n, m)
    uT = cp.random.rand(n, m)
    true_a = cp.ones((n, m)) * 0.1
    true_b = cp.ones((n, m)) * 0.1
    true_c = cp.ones((n, m)) * 0.1
    
    # Apply boundary conditions
    u0[0,:], u0[-1,:], u0[:,0], u0[:,-1] = uT[0,:], uT[-1,:], uT[:,0], uT[:,-1]
    
    # Initial guess on CPU for scipy
    initial_params_np = np.ones(3 * n * m) * 0.05
    
    print("Starting optimization...")
    optimization_start = time.time()
    
    # Optimization with bounds and relaxed tolerances
    bounds = [(0, 1)] * (3 * n * m)
    result = minimize(
        loss_function,
        initial_params_np,
        args=(u0, uT, dt, dx, dy, n, m, t_max),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
    )
    
    optimization_time = time.time() - optimization_start
    
    print(f"\nOptimization completed in {optimization_time:.2f}s")
    print("Optimization Test Results:")
    print("Optimization success:", result.success)
    print("Optimization message:", result.message)
    print("Final loss:", result.fun)
    print("Number of iterations:", result.nit)
    print("Total function evaluations:", opt_iteration)
    
    return result.success or result.fun < 1e-8

# Run all tests with overall progress tracking
def run_tests():
    print("="*60)
    print("Running GPU Numerical Solver Tests...")
    print("="*60)
    
    total_tests = 3
    passed_tests = 0
    overall_start = time.time()
    
    # Test 1: Thomas Algorithm
    print(f"\n[1/{total_tests}] Thomas Algorithm Test")
    print("-" * 40)
    test_start = time.time()
    thomas_passed = test_thomas_algorithm()
    test_time = time.time() - test_start
    if thomas_passed:
        passed_tests += 1
    print(f"Thomas Algorithm: {'PASSED' if thomas_passed else 'FAILED'} ({test_time:.2f}s)")
    
    # Test 2: Governing Equation
    print(f"\n[2/{total_tests}] Governing Equation Test")
    print("-" * 40)
    test_start = time.time()
    gov_eq_passed = test_governing_equation()
    test_time = time.time() - test_start
    if gov_eq_passed:
        passed_tests += 1
    print(f"Governing Equation: {'PASSED' if gov_eq_passed else 'FAILED'} ({test_time:.2f}s)")
    
    # Test 3: Optimization
    print(f"\n[3/{total_tests}] Optimization Test")
    print("-" * 40)
    test_start = time.time()
    opt_passed = test_optimization()
    test_time = time.time() - test_start
    if opt_passed:
        passed_tests += 1
    print(f"Optimization: {'PASSED' if opt_passed else 'FAILED'} ({test_time:.2f}s)")
    
    # Overall summary
    total_time = time.time() - overall_start
    print("\n" + "="*60)
    print("FINAL TEST SUMMARY")
    print("="*60)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    print(f"Total execution time: {total_time:.2f}s")
    print("="*60)
    
    return passed_tests == total_tests

if __name__ == "__main__":
    # Check GPU availability
    print("GPU Information:")
    print(f"CuPy version: {cp.__version__}")
    print(f"CUDA available: {cp.cuda.is_available()}")
    if cp.cuda.is_available():
        print(f"GPU device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        print(f"GPU memory: {cp.cuda.runtime.memGetInfo()[1] / 1024**3:.1f} GB")
    print()
    
    # Run all tests
    success = run_tests()
    
    # Ensure all GPU operations are complete
    cp.cuda.runtime.deviceSynchronize()
    
    print(f"\nProgram {'completed successfully' if success else 'completed with errors'}!")
