import cupy as cp
import numpy as np
from scipy.optimize import minimize

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

# Governing equation solver using Lie splitting (GPU version)
def solve_governing_equation(u0, a, b, c, dt, dx, dy, n, m, t_max):
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
    
    for _ in range(nt):
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
    
    return u

# Test Governing Equation
def test_governing_equation():
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
    
    u = solve_governing_equation(u0, a, b, c, dt, dx, dy, n, m, t_max)
    
    # Convert to NumPy for printing
    u_np = cp.asnumpy(u)
    uT_np = cp.asnumpy(uT)
    
    print("\nGoverning Equation Test:")
    print("Solution shape:", u.shape)
    print("Solution min/max:", u_np.min(), u_np.max())
    print("Boundary conditions preserved:", 
          cp.allclose(u[0,:], uT[0,:]) and 
          cp.allclose(u[-1,:], uT[-1,:]) and 
          cp.allclose(u[:,0], uT[:,0]) and 
          cp.allclose(u[:,-1], uT[:,-1]))
    
    return cp.all(cp.logical_and(u >= 0, u <= 1)).get()

# Loss function
def loss_function(params, u0, uT, dt, dx, dy, n, m, t_max):
    """
    Computes loss on GPU, converts to NumPy for scipy
    params: NumPy array (CPU)
    u0, uT: CuPy arrays
    """
    size = n * m
    # Convert params to CuPy
    a = cp.array(params[:size]).reshape(n, m)
    b = cp.array(params[size:2*size]).reshape(n, m)
    c = cp.array(params[2*size:]).reshape(n, m)  # Fixed indexing
    
    u = solve_governing_equation(u0, a, b, c, dt, dx, dy, n, m, t_max)
    # Convert to NumPy for loss computation
    loss = cp.sum((u - uT) ** 2).get()  # .get() moves to CPU
    return loss

# Test Optimization
def test_optimization():
    n, m = 8, 8
    dx, dy = 1, 1
    dt = 0.01
    t_max = 1
    
    # Initialize test data on GPU
    u0 = cp.random.rand(n, m)  # Fixed: Removed extra parentheses
    uT = cp.random.rand(n, m)  # Fixed: Removed extra parentheses
    true_a = cp.ones((n, m)) * 0.1
    true_b = cp.ones((n, m)) * 0.1
    true_c = cp.ones((n, m)) * 0.1
    
    # Apply boundary conditions
    u0[0,:], u0[-1,:], u0[:,0], u0[:,-1] = uT[0,:], uT[-1,:], uT[:,0], uT[:,-1]
    
    # Initial guess on CPU for scipy
    initial_params_np = np.ones(3 * n * m) * 0.05
    
    # Optimization with bounds and relaxed tolerances
    bounds = [(0, 1)] * (3 * n * m)  # Fixed: Corrected syntax
    result = minimize(
        loss_function,
        initial_params_np,
        args=(u0, uT, dt, dx, dy, n, m, t_max),  # Fixed: dx, dy
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': True}
    )
    
    print("\nOptimization Test:")
    print("Optimization success:", result.success)
    print("Optimization message:", result.message)
    print("Final loss:", result.fun)  # Fixed: fuun -> fun
    print("Number of iterations:", result.nit)
    
    return result.success or result.fun < 1e-8

# Run all tests
def run_tests():
    print("Running Tests (GPU version)...")
    thomas_passed = test_thomas_algorithm()
    gov_eq_passed = test_governing_equation()
    opt_passed = test_optimization()
    
    print("\nTest Summary:")
    print("Thomas Algorithm:", "PASSED" if thomas_passed else "FAILED")
    print("Governing Equation:", "PASSED" if gov_eq_passed else "FAILED")
    print("Optimization:", "PASSED" if opt_passed else "FAILED")
    
    return thomas_passed and gov_eq_passed and opt_passed

if __name__ == "__main__":
    run_tests()
    cp.cuda.runtime.deviceSynchronize()  # Ensure all GPU operations are complete