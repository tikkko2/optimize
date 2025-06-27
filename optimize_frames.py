import cupy as cp
import numpy as np
import cv2
from scipy.optimize import minimize

# Thomas Algorithm for tridiagonal system (GPU version)
def thomas_algorithm(a, b, c, d):
    n = len(b)
    c_prime = cp.zeros(n-1)
    d_prime = cp.zeros(n)
    
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denom
    
    d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / (b[n-1] - a[n-2] * c_prime[n-2])
    
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
    
    a, b, c = init_time_dependent_coeffs(a_base, b_base, c_base, t, t_max)
    
    for k in range(nt):
        a_k = a[:, :, k]
        b_k = b[:, :, k]
        c_k = c[:, :, k]
        
        ap = (a_k + cp.abs(a_k)) / 2
        am = (a_k - cp.abs(a_k)) / 2
        bp = (b_k + cp.abs(b_k)) / 2
        bm = (b_k - cp.abs(b_k)) / 2
        
        u_x = u.copy()
        for j in range(1, m-1):
            lower = -dt/dx * ap[1:-1,j]
            diag = 1 + dt/dx * (ap[1:-1,j] - am[1:-1,j])
            upper = dt/dx * am[1:-1,j]
            rhs = u[1:-1,j] + dt * c_k[1:-1,j] * (1 - u[1:-1,j])
            
            u_x[1:-1,j] = thomas_algorithm(lower, diag, upper, rhs)
        
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

# Load video frames and compute optical flow
def load_video_frames_and_flow(video_path, n, m, frame_idx=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Read first frame (u0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read first frame")
    
    # Read second frame (uT)
    ret, frame1 = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read second frame")
    
    cap.release()
    
    # Convert to grayscale and resize
    frame0_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame0_resized = cv2.resize(frame0_gray, (m, n), interpolation=cv2.INTER_AREA)
    frame1_resized = cv2.resize(frame1_gray, (m, n), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    u0_np = frame0_resized.astype(np.float64) / 255.0
    uT_np = frame1_resized.astype(np.float64) / 255.0
    u0 = cp.array(u0_np)
    uT = cp.array(uT_np)
    
    # Compute optical flow using GPU
    frame0_cuda = cv2.cuda_GpuMat()
    frame1_cuda = cv2.cuda_GpuMat()
    frame0_cuda.upload(frame0_resized)
    frame1_cuda.upload(frame1_resized)
    
    optical_flow = cv2.cuda_FarnebackOpticalFlow.create(
        numLevels=3, pyrScale=0.5, fastPyramids=False, winSize=15,
        numIters=3, polyN=5, polySigma=1.2, flags=0)
    
    flow_cuda = optical_flow.calc(frame0_cuda, frame1_cuda, None)
    flow = flow_cuda.download()
    
    # Extract vx (a_base), vy (b_base)
    vx = flow[:, :, 0]  # x-component
    vy = flow[:, :, 1]  # y-component
    
    # Scale to [0, 1]
    vx = (vx - vx.min()) / (vx.max() - vx.min() + 1e-10)
    vy = (vy - vy.min()) / (vy.max() - vy.min() + 1e-10)
    
    a_base = cp.array(vx)
    b_base = cp.array(vy)
    
    return u0, uT, a_base, b_base

# Loss function
def loss_function(params, u0, uT, dt, dx, dy, n, m, t_max):
    size = n * m
    a_base = cp.array(params[:size]).reshape(n, m)
    b_base = cp.array(params[size:2*size]).reshape(n, m)
    c_base = cp.array(params[2*size:]).reshape(n, m)
    
    u = solve_governing_equation(u0, a_base, b_base, c_base, dt, dx, dy, n, m, t_max)
    loss = cp.sum((u - uT) ** 2).get()
    return loss

# Test Optimization with video frames and optical flow
def test_optimization():
    n, m = 5, 5
    dx, dy = 0.2, 0.2
    dt = 0.001
    t_max = 0.1
    video_path = "input_video.mp4"  # Replace with your video path
    
    # Load video frames and optical flow
    u0, uT, a_base, b_base = load_video_frames_and_flow(video_path, n, m)
    c_base = cp.zeros((n, m))  # Initial c = 0
    
    # Apply boundary conditions
    u0[0,:], u0[-1,:], u0[:,0], u0[:,-1] = uT[0,:], uT[-1,:], uT[:,0], uT[:,-1]
    
    # Verify initialization
    u0_np = cp.asnumpy(u0)
    uT_np = cp.asnumpy(uT)
    a_base_np = cp.asnumpy(a_base)
    b_base_np = cp.asnumpy(b_base)
    print("\nOptimization Test Initialization:")
    print("u0 min/max:", u0_np.min(), u0_np.max())
    print("uT min/max:", uT_np.min(), uT_np.max())
    print("a_base min/max:", a_base_np.min(), a_base_np.max())
    print("b_base min/max:", b_base_np.min(), b_base_np.max())
    print("c_base min/max:", 0.0, 0.0)
    
    # Initial guess on CPU for scipy
    initial_params_np = np.concatenate([
        cp.asnumpy(a_base).flatten(),
        cp.asnumpy(b_base).flatten(),
        cp.asnumpy(c_base).flatten()
    ])
    
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
    print("Running Tests (GPU version, video frames)...")
    thomas_passed = test_thomas_algorithm()
    gov_eq_passed = test_governing_equation()
    opt_passed = test_optimization()
    
    print("\nTest Summary:")
    print("Thomas Algorithm:", "PASSED" if thomas_passed else "FAILED")
    print("Governing Equation:", "PASSED" if gov_eq_passed else "FAILED")
    print("Optimization:", "PASSED" if opt_passed else "FAILED")
    
    return thomas_passed and gov_eq_passed and opt_passed

if __name__ == "__main__":
    # Check CUDA availability
    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        raise RuntimeError("No CUDA-enabled GPU detected")
    run_tests()
    cp.cuda.runtime.deviceSynchronize()