# Import necessary libraries
import cupy as cp           # GPU acceleration library (like NumPy but for GPU)
import numpy as np          # Standard numerical computing library for CPU
import cv2                  # Computer vision library for video/image processing
from scipy.optimize import minimize  # Optimization library to find best parameters
import os                   # For file operations

# Thomas Algorithm for tridiagonal system (GPU version)
def thomas_algorithm(a, b, c, d):
    """
    Solves a system of equations where the matrix has a special pattern (tridiagonal)
    This is much faster than general matrix solving
    a, b, c are the diagonals of the matrix, d is the right-hand side
    """
    n = len(b)                    # Get the size of the system
    c_prime = cp.zeros(n-1)       # Create empty array for modified upper diagonal
    d_prime = cp.zeros(n)         # Create empty array for modified right-hand side
    
    # Forward elimination step - modify the matrix to make it easier to solve
    c_prime[0] = c[0] / b[0]      # Calculate first modified coefficient
    d_prime[0] = d[0] / b[0]      # Calculate first modified right-hand side
    
    # Continue forward elimination for middle elements
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * c_prime[i-1]    # Calculate denominator
        c_prime[i] = c[i] / denom               # Update coefficient
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denom  # Update right-hand side
    
    # Handle the last element specially
    d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / (b[n-1] - a[n-2] * c_prime[n-2])
    
    # Back substitution step - solve for the unknowns starting from the end
    x = cp.zeros(n)               # Create array to store the solution
    x[n-1] = d_prime[n-1]         # Last element is known directly
    
    # Work backwards to find all other elements
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x                      # Return the solution

# Initialize time-dependent coefficients
def init_time_dependent_coeffs(a_base, b_base, c_base, t, t_max):
    """
    Creates coefficients that change over time using sinusoidal patterns
    This makes the equations more realistic as real-world parameters often vary
    """
    nt = len(t)                   # Number of time steps
    n, m = a_base.shape          # Size of the spatial grid
    
    # Create 3D arrays to store coefficients for each point in space and time
    a = cp.zeros((n, m, nt))     # Coefficient 'a' for all locations and times
    b = cp.zeros((n, m, nt))     # Coefficient 'b' for all locations and times
    c = cp.zeros((n, m, nt))     # Coefficient 'c' for all locations and times
    
    # Calculate coefficients for each time step
    for k in range(nt):
        t_k = t[k]               # Current time
        # Make coefficients curve with time (like waves)
        a[:, :, k] = a_base * (1 + 0.1 * cp.sin(np.pi * t_k / t_max))
        b[:, :, k] = b_base * (1 + 0.1 * cp.cos(np.pi * t_k / t_max))
        c[:, :, k] = c_base * (1 + 0.1 * cp.sin(2 * np.pi * t_k / t_max))
    
    return a, b, c

# Governing equation solver using Lie splitting (GPU version)
def solve_governing_equation(u0, a_base, b_base, c_base, dt, dx, dy, n, m, t_max):
    """
    Solves a partial differential equation that describes how a quantity 'u' 
    changes over time and space. Uses Lie splitting to handle x and y directions separately.
    """
    u = u0.copy()                # Start with initial condition
    nt = int(t_max / dt)         # Calculate number of time steps needed
    t = cp.linspace(0, t_max, nt)  # Create array of time points
    
    # Get time-varying coefficients
    a, b, c = init_time_dependent_coeffs(a_base, b_base, c_base, t, t_max)
    
    # Time stepping loop - advance solution forward in time
    for k in range(nt):
        # Get coefficients for current time step
        a_k = a[:, :, k]
        b_k = b[:, :, k]
        c_k = c[:, :, k]
        
        # Split coefficients into positive and negative parts (for numerical stability)
        ap = (a_k + cp.abs(a_k)) / 2    # Positive part of 'a'
        am = (a_k - cp.abs(a_k)) / 2    # Negative part of 'a'
        bp = (b_k + cp.abs(b_k)) / 2    # Positive part of 'b'
        bm = (b_k - cp.abs(b_k)) / 2    # Negative part of 'b'
        
        # Step 1: Solve in x-direction (column by column)
        u_x = u.copy()
        for j in range(1, m-1):      # Loop through each column (skip boundaries)
            # Set up tridiagonal system for this column
            lower = -dt/dx * ap[1:-1,j]                              # Lower diagonal
            diag = 1 + dt/dx * (ap[1:-1,j] - am[1:-1,j])            # Main diagonal
            upper = dt/dx * am[1:-1,j]                               # Upper diagonal
            # rhs = u[1:-1,j] + dt * c_k[1:-1,j] * (1 - u[1:-1,j])   # Right-hand side
            rhs = u[1:-1,j] + dt * c_k[1:-1,j] * u[1:-1,j]  # Right-hand side
            
            # Solve the system for this column
            u_x[1:-1,j] = thomas_algorithm(lower, diag, upper, rhs)
        
        # Step 2: Solve in y-direction (row by row)
        u_y = u_x.copy()
        for i in range(1, n-1):      # Loop through each row (skip boundaries)
            # Set up tridiagonal system for this row
            lower = -dt/dy * bp[i,1:-1]                              # Lower diagonal
            diag = 1 + dt/dy * (bp[i,1:-1] - bm[i,1:-1])            # Main diagonal
            upper = dt/dy * bm[i,1:-1]                               # Upper diagonal
            rhs = u_x[i,1:-1] + dt * c_k[i,1:-1] * (1 - u_x[i,1:-1])  # Right-hand side
            
            # Solve the system for this row
            u_y[i,1:-1] = thomas_algorithm(lower, diag, upper, rhs)
        
        u = u_y.copy()               # Update solution for next time step
    
    return u                         # Return final solution

# Enhanced video frame loading and optical flow computation
def load_video_frames_and_extract_flow(video_path, target_width=None, target_height=None, frame_idx=0, downsample_factor=8):
    """
    Load two consecutive frames from a video and compute optical flow
    Enhanced version with better resolution handling and optical flow processing
    
    Args:
        video_path: Path to the video file
        target_width: Desired width (if None, use original width)
        target_height: Desired height (if None, use original height)
        frame_idx: Starting frame index
        downsample_factor: Factor to reduce resolution (1 = no downsampling)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)    # Open the video file
    if not cap.isOpened():                # Check if video opened successfully
        raise ValueError("Could not open video file")
    
    # Get original video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original video size: {original_width}x{original_height}, Total frames: {total_frames}")
    
    # Set target dimensions
    if target_width is None:
        target_width = original_width // 32
    if target_height is None:
        target_height = original_height // 16
    
    n, m = target_height, target_width  # n=height, m=width
    print(f"Target processing size: {m}x{n} (width x height)")
    
    # Read first frame (initial condition u0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Go to specified frame
    ret, frame0 = cap.read()              # Read the frame
    if not ret:                           # Check if frame was read successfully
        cap.release()
        raise ValueError(f"Could not read frame {frame_idx}")
    
    # Read second frame (target condition uT)
    ret, frame1 = cap.read()              # Read next frame
    if not ret:                           # Check if frame was read successfully
        cap.release()
        raise ValueError(f"Could not read frame {frame_idx + 1}")
    
    cap.release()                         # Close the video file
    
    # Convert to grayscale (remove color information)
    frame0_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    print(f"Frames converted to grayscale, shapes: {frame0_gray.shape}, {frame1_gray.shape}")
    
    # Resize frames to target size
    frame0_resized = cv2.resize(frame0_gray, (m, n), interpolation=cv2.INTER_AREA)
    frame1_resized = cv2.resize(frame1_gray, (m, n), interpolation=cv2.INTER_AREA)
    
    print(f"Frames resized to: {frame0_resized.shape}")
    
    # Ensure frames are in correct data type for optical flow
    frame0_resized = frame0_resized.astype(np.uint8)
    frame1_resized = frame1_resized.astype(np.uint8)
    
    # Normalize pixel values to range [0, 1] and convert to GPU arrays
    u0 = cp.array(frame0_resized.astype(np.float64) / 255.0)  # First frame as initial condition
    uT = cp.array(frame1_resized.astype(np.float64) / 255.0)  # Second frame as target
    
    # Compute dense optical flow using Farneback method
    try:
        flow = cv2.calcOpticalFlowFarneback(frame0_resized, frame1_resized, None, 
                                           pyr_scale=0.5, levels=3, winsize=15, 
                                           iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        
        if flow is None:
            raise ValueError("Optical flow computation returned None")
            
        print(f"Optical flow computed successfully, shape: {flow.shape}")
        
    except Exception as e:
        print(f"Error computing optical flow: {e}")
        # Fallback: create zero flow
        flow = np.zeros((n, m, 2), dtype=np.float32)
        print("Using zero optical flow as fallback")
    
    # Extract motion components
    vx = flow[:, :, 0]                    # Horizontal motion (x-component)
    vy = flow[:, :, 1]                    # Vertical motion (y-component)
    
    # Check for valid flow values
    if np.any(np.isnan(vx)) or np.any(np.isnan(vy)):
        print("Warning: NaN values found in optical flow, setting to zero")
        vx = np.nan_to_num(vx, 0)
        vy = np.nan_to_num(vy, 0)
    
    # Normalize and scale optical flow vectors
    vx_max = max(abs(vx.min()), abs(vx.max()))
    vy_max = max(abs(vy.min()), abs(vy.max()))
    
    # if vx_max > 1e-6:  # Avoid division by very small numbers
    #     vx_normalized = vx / vx_max * 0.1  # Scale to reasonable range
    # else:
    #     vx_normalized = np.zeros_like(vx)
    #     print("Warning: No significant horizontal motion detected")
        
    # if vy_max > 1e-6:  # Avoid division by very small numbers
    #     vy_normalized = vy / vy_max * 0.1  # Scale to reasonable range
    # else:
    #     vy_normalized = np.zeros_like(vy)
    #     print("Warning: No significant vertical motion detected")
    
    # Convert optical flow to coefficient arrays (a and b are optical flow vectors)
    a_base = vx # cp.array(vx_normalized)       Horizontal optical flow as 'a' coefficient
    b_base = vy # cp.array(vy_normalized)       Vertical optical flow as 'b' coefficient
    
    print(f"Optical flow range - vx: [{vx.min():.3f}, {vx.max():.3f}], vy: [{vy.min():.3f}, {vy.max():.3f}]")
    print(f"Normalized flow range - a: [{cp.asnumpy(a_base).min():.3f}, {cp.asnumpy(a_base).max():.3f}], " +
          f"b: [{cp.asnumpy(b_base).min():.3f}, {cp.asnumpy(b_base).max():.3f}]")
    
    return u0, uT, a_base, b_base, n, m

# Initialize parameters with better initial guess
def initialize_parameters(n, m, a_base, b_base, c_init_value=0.0):
    """
    Initialize parameters for optimization with reasonable initial guess
    
    Args:
        n, m: Grid dimensions (height, width)
        a_base, b_base: Optical flow vectors
        c_init_value: Initial value for reaction coefficient
    """
    # Use optical flow as initial guess for a and b
    a_init = cp.asnumpy(a_base).flatten()
    b_init = cp.asnumpy(b_base).flatten()
    
    # Initialize reaction coefficient with constant value
    c_init = np.full(n * m, c_init_value)
    
    # Combine all parameters into single vector
    initial_params = np.concatenate([a_init, b_init, c_init])
    
    print(f"Parameter vector size: {len(initial_params)} (3 × {n} × {m})")
    
    return initial_params

# Loss function with improved numerical stability
def loss_function(params, u0, uT, dt, dx, dy, n, m, t_max, regularization=1e-6):
    """
    Calculate how different our simulation result is from the target
    Enhanced with regularization for better optimization stability
    """
    size = n * m                          # Total number of grid points
    
    try:
        # Extract coefficients from parameter vector
        a_base = cp.array(params[:size]).reshape(n, m)           # First n*m parameters are 'a'
        b_base = cp.array(params[size:2*size]).reshape(n, m)     # Next n*m parameters are 'b'
        c_base = cp.array(params[2*size:]).reshape(n, m)         # Last n*m parameters are 'c'
        
        # Run simulation with these parameters
        u = solve_governing_equation(u0, a_base, b_base, c_base, dt, dx, dy, n, m, t_max)
        
        # Calculate main loss (difference between simulation and target)
        main_loss = cp.sum((u - uT) ** 2)
        
        # Add regularization to prevent overfitting and maintain smoothness
        reg_a = cp.sum(cp.diff(a_base, axis=0) ** 2) + cp.sum(cp.diff(a_base, axis=1) ** 2)
        reg_b = cp.sum(cp.diff(b_base, axis=0) ** 2) + cp.sum(cp.diff(b_base, axis=1) ** 2)  
        reg_c = cp.sum(cp.diff(c_base, axis=0) ** 2) + cp.sum(cp.diff(c_base, axis=1) ** 2)
        
        total_loss = main_loss + regularization * (reg_a + reg_b + reg_c)
        
        return total_loss.get()    # .get() converts from GPU to CPU
        
    except Exception as e:
        print(f"Error in loss function: {e}")
        return 1e10  # Return large loss if simulation fails

# Main optimization function
def optimize_pde_parameters(video_path, frame_idx=0, target_width=None, target_height=None, 
                           downsample_factor=8, dt=0.01, t_max=0.1, max_iterations=50):
    """
    Main function to optimize PDE parameters using video frames and optical flow
    
    Args:
        video_path: Path to input video
        frame_idx: Starting frame index
        target_width, target_height: Target processing resolution
        downsample_factor: Factor to reduce original resolution
        dt: Time step for PDE solver
        t_max: Total simulation time
        max_iterations: Maximum optimization iterations
    """
    print("Starting PDE parameter optimization...")
    
    # Load video frames and extract optical flow
    u0, uT, a_base, b_base, n, m = load_video_frames_and_extract_flow(
        video_path, target_width, target_height, frame_idx, downsample_factor)
    
    # Set spatial steps based on grid size
    dx, dy = 1.0 / m, 1.0 / n
    
    # Apply boundary conditions (edges of simulation match target)
    u0[0,:] = uT[0,:]    # Top boundary
    u0[-1,:] = uT[-1,:]  # Bottom boundary  
    u0[:,0] = uT[:,0]    # Left boundary
    u0[:,-1] = uT[:,-1]  # Right boundary
    
    print(f"Grid size: {n}×{m}, Time step: {dt}, Total time: {t_max}")
    print(f"Initial condition range: [{cp.asnumpy(u0).min():.3f}, {cp.asnumpy(u0).max():.3f}]")
    print(f"Target condition range: [{cp.asnumpy(uT).min():.3f}, {cp.asnumpy(uT).max():.3f}]")
    
    # Initialize parameters with optical flow as initial guess
    initial_params = initialize_parameters(n, m, a_base, b_base)

    vx_max = max(abs(a_base.min()), abs(a_base.max()))
    vy_max = max(abs(b_base.min()), abs(b_base.max()))

    # Set up optimization bounds (reasonable ranges for each parameter type)
    bounds_a = [(-vx_max - 1, vx_max + 1)] * (n * m)  # Bounds for 'a' coefficients (horizontal flow)
    bounds_b = [(-vy_max - 1, vy_max + 1)] * (n * m)  # Bounds for 'b' coefficients (vertical flow)  
    bounds_c = [(-1, 1)] * (n * m)       # Bounds for 'c' coefficients (reaction term)
    bounds = bounds_a + bounds_b + bounds_c
    
    print(f"Starting optimization with {len(initial_params)} parameters...")
    
    # Run optimization to find best parameters
    result = minimize(
        loss_function,                    # Function to minimize
        initial_params,                   # Starting point (initial guess)
        args=(u0, uT, dt, dx, dy, n, m, t_max),  # Additional arguments
        method='L-BFGS-B',                # Optimization algorithm
        bounds=bounds,                    # Parameter constraints
        options={                         # Optimization settings
            'maxiter': max_iterations,    # Maximum iterations
            'ftol': 1e-12,                # Function tolerance
            'gtol': 1e-8,                 # Gradient tolerance
            'disp': True                  # Display progress
        }
    )
    
    # Extract optimized parameters
    size = n * m
    a_opt = result.x[:size].reshape(n, m)
    b_opt = result.x[size:2*size].reshape(n, m)
    c_opt = result.x[2*size:].reshape(n, m)
    
    # Print optimization results
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Final loss: {result.fun:.2e}")
    print(f"Iterations: {result.nit}")
    print(f"Function evaluations: {result.nfev}")
    
    print(f"\nOptimized parameter ranges:")
    print(f"a (horizontal flow): [{a_opt.min():.4f}, {a_opt.max():.4f}]")
    print(f"b (vertical flow): [{b_opt.min():.4f}, {b_opt.max():.4f}]")
    print(f"c (reaction term): [{c_opt.min():.4f}, {c_opt.max():.4f}]")
    
    return result, (u0, uT, cp.array(a_opt), cp.array(b_opt), cp.array(c_opt), n, m)

# Test function with real video
def test_with_video(video_path="handvid.mp4"):
    """
    Test the complete system with a real video file
    """
    if not os.path.exists(video_path):
        print(f"Video file {video_path} not found. Please provide a valid video file.")
        print("Creating a simple test case instead...")
        return test_synthetic_case()
    
    try:
        # Run optimization with video
        print("Starting video-based optimization...")
        result, (u0, uT, a_opt, b_opt, c_opt, n, m) = optimize_pde_parameters(
            video_path, 
            frame_idx=10,          # Start from frame 10
            downsample_factor=8,   # Reduce resolution for faster processing
            max_iterations=20      # Fewer iterations for testing
        )
        
        print("Optimization completed, verifying results...")
        
        # Test the optimized solution
        dt, t_max = 0.01, 0.1
        dx, dy = 1.0 / m, 1.0 / n
        
        u_final = solve_governing_equation(u0, a_opt, b_opt, c_opt, dt, dx, dy, n, m, t_max)
        final_error = cp.sum((u_final - uT) ** 2).get()
        
        print(f"\nFinal verification:")
        print(f"Final error: {final_error:.2e}")
        print(f"Relative error: {final_error / cp.sum(uT ** 2).get():.2e}")
        
        return result.success and final_error < 1.0
        
    except FileNotFoundError as e:
        print(f"File error: {e}")
        return False
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error in video test: {e}")
        import traceback
        traceback.print_exc()
        return False

# Synthetic test case (fallback when no video available)
def test_synthetic_case():
    """
    Test with synthetic data when video is not available
    """
    print("Running synthetic test case...")
    
    # Create synthetic data
    n, m = 20, 20
    dx, dy = 1.0 / m, 1.0 / n
    dt, t_max = 0.01, 0.05
    
    # Create initial condition (Gaussian blob)
    x = cp.linspace(0, 1, m)
    y = cp.linspace(0, 1, n)
    X, Y = cp.meshgrid(x, y)
    u0 = cp.exp(-10 * ((X - 0.3)**2 + (Y - 0.3)**2))
    
    # Create target condition (shifted Gaussian)
    uT = cp.exp(-10 * ((X - 0.7)**2 + (Y - 0.7)**2))
    
    # Create synthetic optical flow (shift from initial to target)
    a_true = cp.ones((n, m)) * 0.1  # Horizontal flow
    b_true = cp.ones((n, m)) * 0.1  # Vertical flow
    c_true = cp.ones((n, m)) * 0.2  # Reaction term
    
    # Apply boundary conditions
    u0[0,:] = uT[0,:]
    u0[-1,:] = uT[-1,:]
    u0[:,0] = uT[:,0]
    u0[:,-1] = uT[:,-1]
    
    # Initialize parameters
    initial_params = initialize_parameters(n, m, a_true * 0.5, b_true * 0.5, 0.1)
    
    # Set up bounds
    bounds = [(-0.5, 0.5)] * (2 * n * m) + [(0, 1)] * (n * m)
    
    # Run optimization
    result = minimize(
        loss_function, initial_params, 
        args=(u0, uT, dt, dx, dy, n, m, t_max),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 30, 'disp': True}
    )
    
    print(f"Synthetic test - Success: {result.success}, Final loss: {result.fun:.2e}")
    return result.success and result.fun < 0.1

# Main execution
if __name__ == "__main__":
    # Check if CUDA is available
    try:
        cp.cuda.runtime.getDeviceCount()
        print("CUDA GPU detected and available")
    except:
        print("Warning: CUDA not available, performance will be slower")
    
    # Test with video or synthetic data
    video_file = "handvid.mp4"  # Change this to your video file path
    
    if os.path.exists(video_file):
        print(f"Testing with video file: {video_file}")
        success = test_with_video(video_file)
    else:
        print("No video file found, running synthetic test")
        success = test_synthetic_case()
    
    if success:
        print("\n[SUCCESS] Test completed successfully!")
    else:
        print("\n[FAILED] Test failed!")
    
    # Ensure all GPU operations are completed
    cp.cuda.runtime.deviceSynchronize()
