import cupy as cp
import numpy as np
from scipy.optimize import minimize

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
    a = cp.array([1, 1, 1], dtype=cp.float64)
    b = cp.array([4, 4, 4, 4], dtype=cp.float64)
    c = cp.array([1, 1, 1], dtype=cp.float64)
    d = cp.array([5, 6, 6, 5], dtype=cp.float64)
    
    x = thomas_algorithm(a, b, c, d)
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

# Initialize time-dependent coefficients
def init_time_dependent_coeffs(a_base, b_base, c_base, t, t_max):
    """
    Returns a(t), b(t), c(t) as 3D arrays of shape (n, m, nt)
    a_base, b_base, c_base: baseline coefficients (n, m)
    t: time array (nt,)
    """
    nt = len(t)
    n, m = a_base.shape
    a = cp.zeros((n, m, nt))
    b = cp.zeros((n, m, nt))
    c = cp.zeros((n, m, nt))
    
    for k in range(nt):
        t_k = t[k]
        a[:, :, k] = a_base * (1 + 0.5 * cp.sin(np.pi * t_k / t_max))
        b[:, :, k] = b_base * (1 + 0.5 * cp.cos(np.pi * t_k / t_max))
        c[:, :, k] = c_base * (1 + 0.5 * cp.sin(2 * np.pi * t_k / t_max))
    
    return a, b, c

# Governing equation solver using Lie splitting (GPU version)
def solve_governing_equation(u0, a_base, b_base, c_base, dt, dx, dy, n, m, t_max):
    u = u0.copy()
    nt = int(t_max / dt)
    t = cp.linspace(0, t_max, nt)
    
    # Initialize time-dependent coefficients
    a, b, c = init_time_dependent_coeffs(a_base, b_base, c_base, t, t_max)
    
    for k in range(nt):
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
    
    return u

# Test Governing Equation
def test_governing_equation():
    n, m = 10, 10
    dx, dy = 0.1, 0.1
    dt = 0.001
    t_max = 0.1
    
    u0 = cp.zeros((n, m), dtype=cp.float64)
    uT = cp.ones((n, m), dtype=cp.float64) * 0.5
    a_base = cp.ones((n, m), dtype=cp.float64) * 0.1
    b_base = cp.ones((n, m), dtype=cp.float64) * 0.1
    c_base = cp.ones((n, m), dtype=cp.float64) * 0.1
    
    u0[0,:], u0[-1,:], u0[:,0], u0[:,-1] = uT[0,:], uT[-1,:], uT[:,0], uT[:,-1]
    
    u = solve_governing_equation(u0, a_base, b_base, c_base, dt, dx, dy, n, m, t_max)
    
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
    size = n * m
    a_base = cp.array(params[:size]).reshape(n, m)
    b_base = cp.array(params[size:2*size]).reshape(n, m)
    c_base = cp.array(params[2*size:]).reshape(n, m)
    
    u = solve_governing_equation(u0, a_base, b_base, c_base, dt, dx, dy, n, m, t_max)
    loss = cp.sum((u - uT) ** 2).get()
    return loss

# Initialization functions for u0 and uT
def init_u0(x, y):
    return 0.5 * (1 + cp.sin(np.pi * x) * cp.cos(np.pi * y))

def init_uT(x, y):
    return 0.5 * (1 + cp.cos(np.pi * x) * cp.sin(np.pi * y))

# Test Optimization with time-dependent coefficients
def test_optimization():
    n, m = 7, 7
    dx, dy = 0.2, 0.2
    dt = 0.1
    t_max = 1
    
    # Create spatial grid on GPU
    x = cp.linspace(0, 1, n)
    y = cp.linspace(0, 1, m)
    X, Y = cp.meshgrid(x, y, indexing='ij')
    
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
    print("\nOptimization Test Initialization:")
    print("u0 min/max:", u0_np.min(), u0_np.max())
    print("uT min/max:", uT_np.min(), uT_np.max())
    
    # Initial guess on CPU for scipy
    initial_params_np = np.ones(3 * n * m) * 0.05
    
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
    
    print("\nOptimization Test:")
    print("Optimization success:", result.success)
    print("Optimization message:", result.message)
    print("Final loss:", result.fun)
    print("Number of iterations:", result.nit)
    
    return result.success or result.fun < 1e-8

# Run all tests
def run_tests():
    print("Running Tests (GPU version, time-dependent coefficients)...")
    #thomas_passed = test_thomas_algorithm()
    gov_eq_passed = test_governing_equation()
    opt_passed = test_optimization()
    
    print("\nTest Summary:")
    #print("Thomas Algorithm:", "PASSED" if thomas_passed else "FAILED")
    print("Governing Equation:", "PASSED" if gov_eq_passed else "FAILED")
    print("Optimization:", "PASSED" if opt_passed else "FAILED")
    
    #return thomas_passed and gov_eq_passed and opt_passed
    return gov_eq_passed and opt_passed

if __name__ == "__main__":
    run_tests()
    cp.cuda.runtime.deviceSynchronize()
