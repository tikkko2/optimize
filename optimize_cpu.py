import numpy as np
from scipy.optimize import minimize

# Thomas Algorithm for tridiagonal system
def thomas_algorithm(a, b, c, d):
    """
    Solves a tridiagonal system Ax = d
    a: lower diagonal
    b: main diagonal
    c: upper diagonal
    d: right-hand side
    """
    n = len(b)
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)
    
    # Forward elimination
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denom
    
    d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / (b[n-1] - a[n-2] * c_prime[n-2])
    
    # Back substitution
    x = np.zeros(n)
    x[n-1] = d_prime[n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x

# Test Thomas Algorithm
def test_thomas_algorithm():
    a = np.array([1, 1, 1])  # lower diagonal
    b = np.array([4, 4, 4, 4])  # main diagonal
    c = np.array([1, 1, 1])  # upper diagonal
    d = np.array([5, 6, 6, 5])  # RHS
    
    x = thomas_algorithm(a, b, c, d)
    expected = np.linalg.solve(
        np.array([
            [4, 1, 0, 0],
            [1, 4, 1, 0],
            [0, 1, 4, 1],
            [0, 0, 1, 4]
        ]), d)
    
    print("Thomas Algorithm Test:")
    print("Computed:", x)
    print("Expected:", expected)
    print("Error:", np.linalg.norm(x - expected))
    return np.allclose(x, expected)

# Governing equation solver using Lie splitting
def solve_governing_equation(u0, a, b, c, dt, dx, dy, n, m, t_max):
    """
    Solves the governing equation using Lie splitting
    """
    u = u0.copy()
    nt = int(t_max / dt)
    
    # Precompute coefficients
    ap = (a + np.abs(a)) / 2
    am = (a - np.abs(a)) / 2
    bp = (b + np.abs(b)) / 2
    bm = (b - np.abs(b)) / 2
    
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
    n, m = 5, 5
    dx, dy = 0.1, 0.1
    dt = 0.001
    t_max = 0.1
    
    # Initialize test data
    u0 = np.zeros((n, m))
    uT = np.ones((n, m)) * 0.5
    a = np.ones((n, m)) * 0.1
    b = np.ones((n, m)) * 0.1
    c = np.ones((n, m)) * 0.1
    
    # Apply boundary conditions
    u0[0,:], u0[-1,:], u0[:,0], u0[:,-1] = uT[0,:], uT[-1,:], uT[:,0], uT[:,-1]
    
    u = solve_governing_equation(u0, a, b, c, dt, dx, dy, n, m, t_max)
    
    print("\nGoverning Equation Test:")
    print("Solution shape:", u.shape)
    print("Solution min/max:", u.min(), u.max())
    print("Boundary conditions preserved:", 
          np.allclose(u[0,:], uT[0,:]) and 
          np.allclose(u[-1,:], uT[-1,:]) and 
          np.allclose(u[:,0], uT[:,0]) and 
          np.allclose(u[:,-1], uT[:,-1]))
    
    return np.all((u >= 0) & (u <= 1))

# Loss function
def loss_function(params, u0, uT, dt, dx, dy, n, m, t_max):
    size = n * m
    a = params[:size].reshape(n, m)
    b = params[size:2*size].reshape(n, m)
    c = params[2*size:].reshape(n, m)
    
    u = solve_governing_equation(u0, a, b, c, dt, dx, dy, n, m, t_max)
    return np.sum((u - uT) ** 2)

# Test Optimization
def test_optimization():
    n, m = 20, 20
    dx, dy = 0.2, 0.2
    dt = 0.05
    t_max = 0.1
    
    # Initialize test data
    u0 = np.random.rand(n, m)
    uT = np.random.rand(n, m)
    true_a = np.ones((n, m)) * 0.1
    true_b = np.ones((n, m)) * 0.1
    true_c = np.ones((n, m)) * 0.1
    
    # Apply boundary conditions
    u0[0,:], u0[-1,:], u0[:,0], u0[:,-1] = uT[0,:], uT[-1,:], uT[:,0], uT[:,-1]
    
    # Initial guess
    initial_params = np.ones(3 * n * m) * 0.05
    
    # Add bounds and relaxed tolerances
    bounds = [(0, 1)] * (3 * n * m)  # Constrain parameters to [0, 1]
    result = minimize(
        loss_function,
        initial_params,
        args=(u0, uT, dt, dx, dy, n, m, t_max),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': True}
    )
    
    print("\nOptimization Test:")
    print("Optimization success:", result.success)
    print("Optimization message:", result.message)
    print("Final loss:", result.fun)
    print("Number of iterations:", result.nit)
    
    return result.success or result.fun < 1e-8  # Consider small loss as success

# Run all tests
def run_tests():
    print("Running Tests...")
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
