# Import necessary libraries
import cupy as cp           # GPU acceleration library (like NumPy but for GPU)
import numpy as np          # Standard numerical computing library for CPU
import cv2                  # Computer vision library for video/image processing
from scipy.optimize import minimize  # Optimization library to find best parameters
import os                   # For file operations
import time                 # For timing and timestamps
from tqdm import tqdm       # Progress bar library
import sys                  # For flushing output

# Global variables for progress tracking
loss_call_count = 0
best_loss = float('inf')
optimization_start_time = None

def print_with_timestamp(message):
    """Print message with current timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

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
    print_with_timestamp("Initializing time-dependent coefficients...")
    nt = len(t)                   # Number of time steps
    n, m = a_base.shape          # Size of the spatial grid
    
    # Create 3D arrays to store coefficients for each point in space and time
    a = cp.zeros((n, m, nt))     # Coefficient 'a' for all locations and times
    b = cp.zeros((n, m, nt))     # Coefficient 'b' for all locations and times
    c = cp.zeros((n, m, nt))     # Coefficient 'c' for all locations and times
    
    # Calculate coefficients for each time step with progress bar
    print("Computing time-varying coefficients...")
    for k in tqdm(range(nt), desc="Time coefficients", leave=False):
        t_k = t[k]               # Current time
        # Make coefficients curve with time (like waves)
        a[:, :, k] = a_base * (1 + 0.1 * cp.sin(np.pi * t_k / t_max))
        b[:, :, k] = b_base * (1 + 0.1 * cp.cos(np.pi * t_k / t_max))
        c[:, :, k] = c_base * (1 + 0.1 * cp.sin(2 * np.pi * t_k / t_max))
    
    print_with_timestamp(f"Time-dependent coefficients initialized for {nt} time steps")
    return a, b, c

# Governing equation solver using Lie splitting (GPU version)
def solve_governing_equation(u0, a_base, b_base, c_base, dt, dx, dy, n, m, t_max, show_progress=True):
    """
    Solves a partial differential equation that describes how a quantity 'u' 
    changes over time and space. Uses Lie splitting to handle x and y directions separately.
    """
    u = u0.copy()                # Start with initial condition
    nt = int(t_max / dt)         # Calculate number of time steps needed
    t = cp.linspace(0, t_max, nt)  # Create array of time points
    
    if show_progress:
        print_with_timestamp(f"Starting PDE solver: {nt} time steps, dt={dt:.4f}")
    
    # Get time-varying coefficients
    a, b, c = init_time_dependent_coeffs(a_base, b_base, c_base, t, t_max)
    
    # Time stepping loop - advance solution forward in time
    progress_bar = tqdm(range(nt), desc="PDE time steps", disable=not show_progress)
    
    for k in progress_bar:
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
            rhs = u[1:-1,j] + dt * c_k[1:-1,j] * u[1:-1,j]          # Right-hand side
            
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
        
        # Update progress bar with current time
        if show_progress and k % max(1, nt // 20) == 0:  # Update every 5%
            current_time = t[k].get() if hasattr(t[k], 'get') else float(t[k])
            progress_bar.set_postfix({'Time': f'{current_time:.4f}/{t_max:.4f}'})
    
    progress_bar.close()
    if show_progress:
        print_with_timestamp("PDE solution completed")
    
    return u                         # Return final solution

# Enhanced video frame loading and optical flow computation
def load_video_frames_and_extract_flow(video_path, target_width=None, target_height=None, frame_idx=0, downsample_factor=8):
    """
    Load two consecutive frames from a video and compute optical flow
    Enhanced version with better resolution handling and optical flow processing
    """
    print_with_timestamp(f"Loading video: {video_path}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)    # Open the video file
    if not cap.isOpened():                # Check if video opened successfully
        raise ValueError("Could not open video file")
    
    # Get original video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print_with_timestamp(f"Video properties: {original_width}x{original_height}, {total_frames} frames")
    
    # Set target dimensions
    if target_width is None:
        target_width = original_width // 32
    if target_height is None:
        target_height = original_height // 16
    
    n, m = target_height, target_width  # n=height, m=width
    print_with_timestamp(f"Target processing size: {m}x{n} (width x height)")
    
    # Read frames with progress indication
    print("Reading video frames...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Go to specified frame
    ret, frame0 = cap.read()              # Read the frame
    if not ret:                           # Check if frame was read successfully
        cap.release()
        raise ValueError(f"Could not read frame {frame_idx}")
    
    ret, frame1 = cap.read()              # Read next frame
    if not ret:                           # Check if frame was read successfully
        cap.release()
        raise ValueError(f"Could not read frame {frame_idx + 1}")
    
    cap.release()                         # Close the video file
    print_with_timestamp("Frames loaded successfully")
    
    # Convert to grayscale (remove color information)
    print("Converting frames to grayscale...")
    frame0_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    print_with_timestamp(f"Frames converted to grayscale: {frame0_gray.shape}")
    
    # Resize frames to target size
    print("Resizing frames...")
    frame0_resized = cv2.resize(frame0_gray, (m, n), interpolation=cv2.INTER_AREA)
    frame1_resized = cv2.resize(frame1_gray, (m, n), interpolation=cv2.INTER_AREA)
    
    print_with_timestamp(f"Frames resized to: {frame0_resized.shape}")
    
    # Ensure frames are in correct data type for optical flow
    frame0_resized = frame0_resized.astype(np.uint8)
    frame1_resized = frame1_resized.astype(np.uint8)
    
    # Normalize pixel values and convert to GPU arrays
    print("Normalizing and transferring to GPU...")
    u0 = cp.array(frame0_resized.astype(np.float64) / 255.0)  # First frame as initial condition
    uT = cp.array(frame1_resized.astype(np.float64) / 255.0)  # Second frame as target
    
    # Compute dense optical flow using Farneback method
    print("Computing optical flow...")
    try:
        flow = cv2.calcOpticalFlowFarneback(frame0_resized, frame1_resized, None, 
                                           pyr_scale=0.5, levels=3, winsize=15, 
                                           iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        
        if flow is None:
            raise ValueError("Optical flow computation returned None")
            
        print_with_timestamp(f"Optical flow computed successfully: {flow.shape}")
        
    except Exception as e:
        print_with_timestamp(f"Error computing optical flow: {e}")
        # Fallback: create zero flow
        flow = np.zeros((n, m, 2), dtype=np.float32)
        print_with_timestamp("Using zero optical flow as fallback")
    
    # Extract motion components
    vx = flow[:, :, 0]                    # Horizontal motion (x-component)
    vy = flow[:, :, 1]                    # Vertical motion (y-component)
    
    # Check for valid flow values
    if np.any(np.isnan(vx)) or np.any(np.isnan(vy)):
        print_with_timestamp("Warning: NaN values found in optical flow, setting to zero")
        vx = np.nan_to_num(vx, 0)
        vy = np.nan_to_num(vy, 0)
    
    # Convert optical flow to coefficient arrays
    a_base = vx  # Horizontal optical flow as 'a' coefficient
    b_base = vy  # Vertical optical flow as 'b' coefficient
    
    print_with_timestamp(f"Optical flow range - vx: [{vx.min():.3f}, {vx.max():.3f}], vy: [{vy.min():.3f}, {vy.max():.3f}]")
    
    return u0, uT, a_base, b_base, n, m

# Initialize parameters with better initial guess
def initialize_parameters(n, m, a_base, b_base, c_init_value=0.0):
    """
    Initialize parameters for optimization with reasonable initial guess
    """
    print_with_timestamp("Initializing optimization parameters...")
    
    # Use optical flow as initial guess for a and b
    a_init = cp.asnumpy(a_base).flatten()
    b_init = cp.asnumpy(b_base).flatten()
    
    # Initialize reaction coefficient with constant value
    c_init = np.full(n * m, c_init_value)
    
    # Combine all parameters into single vector
    initial_params = np.concatenate([a_init, b_init, c_init])
    
    print_with_timestamp(f"Parameter vector initialized: {len(initial_params)} parameters (3 × {n} × {m})")
    
    return initial_params

# Loss function with improved numerical stability and progress tracking
def loss_function(params, u0, uT, dt, dx, dy, n, m, t_max, regularization=1e-6):
    """
    Calculate how different our simulation result is from the target
    Enhanced with regularization and progress tracking
    """
    global loss_call_count, best_loss, optimization_start_time
    
    loss_call_count += 1
    size = n * m                          # Total number of grid points
    
    try:
        # Extract coefficients from parameter vector
        a_base = cp.array(params[:size]).reshape(n, m)           # First n*m parameters are 'a'
        b_base = cp.array(params[size:2*size]).reshape(n, m)     # Next n*m parameters are 'b'
        c_base = cp.array(params[2*size:]).reshape(n, m)         # Last n*m parameters are 'c'
        
        # Run simulation with these parameters (disable progress for loss function calls)
        u = solve_governing_equation(u0, a_base, b_base, c_base, dt, dx, dy, n, m, t_max, show_progress=False)
        
        # Calculate main loss (difference between simulation and target)
        main_loss = cp.sum((u - uT) ** 2)
        
        # Add regularization to prevent overfitting and maintain smoothness
        reg_a = cp.sum(cp.diff(a_base, axis=0) ** 2) + cp.sum(cp.diff(a_base, axis=1) ** 2)
        reg_b = cp.sum(cp.diff(b_base, axis=0) ** 2) + cp.sum(cp.diff(b_base, axis=1) ** 2)  
        reg_c = cp.sum(cp.diff(c_base, axis=0) ** 2) + cp.sum(cp.diff(c_base, axis=1) ** 2)
        
        total_loss = main_loss + regularization * (reg_a + reg_b + reg_c)
        loss_value = total_loss.get()    # .get() converts from GPU to CPU
        
        # Track best loss and print progress periodically
        if loss_value < best_loss:
            best_loss = loss_value
            elapsed_time = time.time() - optimization_start_time if optimization_start_time else 0
            print_with_timestamp(f"Iteration {loss_call_count}: New best loss = {loss_value:.2e} (after {elapsed_time:.1f}s)")
        elif loss_call_count % 10 == 0:  # Print every 10th evaluation
            elapsed_time = time.time() - optimization_start_time if optimization_start_time else 0
            print(f"Iteration {loss_call_count}: Loss = {loss_value:.2e} (best: {best_loss:.2e}) [{elapsed_time:.1f}s]")
            sys.stdout.flush()
        
        return loss_value
        
    except Exception as e:
        print_with_timestamp(f"Error in loss function (iteration {loss_call_count}): {e}")
        return 1e10  # Return large loss if simulation fails

# Main optimization function
def optimize_pde_parameters(video_path, frame_idx=0, target_width=None, target_height=None, 
                           downsample_factor=8, dt=0.01, t_max=0.1, max_iterations=50):
    """
    Main function to optimize PDE parameters using video frames and optical flow
    """
    global loss_call_count, best_loss, optimization_start_time
    
    print_with_timestamp("="*60)
    print_with_timestamp("STARTING PDE PARAMETER OPTIMIZATION")
    print_with_timestamp("="*60)
    
    start_time = time.time()
    
    # Reset progress tracking variables
    loss_call_count = 0
    best_loss = float('inf')
    
    # Load video frames and extract optical flow
    print_with_timestamp("Phase 1: Loading video and extracting optical flow")
    u0, uT, a_base, b_base, n, m = load_video_frames_and_extract_flow(
        video_path, target_width, target_height, frame_idx, downsample_factor)
    
    # Set spatial steps based on grid size
    dx, dy = 1.0 / m, 1.0 / n
    
    # Apply boundary conditions (edges of simulation match target)
    print_with_timestamp("Applying boundary conditions...")
    u0[0,:] = uT[0,:]    # Top boundary
    u0[-1,:] = uT[-1,:]  # Bottom boundary  
    u0[:,0] = uT[:,0]    # Left boundary
    u0[:,-1] = uT[:,-1]  # Right boundary
    
    print_with_timestamp(f"Grid configuration: {n}×{m}, Time step: {dt}, Total time: {t_max}")
    print_with_timestamp(f"Initial condition range: [{cp.asnumpy(u0).min():.3f}, {cp.asnumpy(u0).max():.3f}]")
    print_with_timestamp(f"Target condition range: [{cp.asnumpy(uT).min():.3f}, {cp.asnumpy(uT).max():.3f}]")
    
    # Initialize parameters with optical flow as initial guess
    print_with_timestamp("Phase 2: Parameter initialization")
    initial_params = initialize_parameters(n, m, a_base, b_base)

    vx_max = max(abs(a_base.min()), abs(a_base.max()))
    vy_max = max(abs(b_base.min()), abs(b_base.max()))

    # Set up optimization bounds
    print_with_timestamp("Setting up optimization bounds...")
    bounds_a = [(-vx_max - 1, vx_max + 1)] * (n * m)  # Bounds for 'a' coefficients
    bounds_b = [(-vy_max - 1, vy_max + 1)] * (n * m)  # Bounds for 'b' coefficients  
    bounds_c = [(-1, 1)] * (n * m)                     # Bounds for 'c' coefficients
    bounds = bounds_a + bounds_b + bounds_c
    
    print_with_timestamp(f"Phase 3: Starting optimization with {len(initial_params)} parameters")
    print_with_timestamp(f"Maximum iterations: {max_iterations}")
    
    # Set optimization start time for progress tracking
    optimization_start_time = time.time()
    
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
    
    optimization_time = time.time() - optimization_start_time
    total_time = time.time() - start_time
    
    # Extract optimized parameters
    size = n * m
    a_opt = result.x[:size].reshape(n, m)
    b_opt = result.x[size:2*size].reshape(n, m)
    c_opt = result.x[2*size:].reshape(n, m)
    
    # Print comprehensive results
    print_with_timestamp("="*60)
    print_with_timestamp("OPTIMIZATION RESULTS")
    print_with_timestamp("="*60)
    print_with_timestamp(f"Success: {result.success}")
    print_with_timestamp(f"Message: {result.message}")
    print_with_timestamp(f"Final loss: {result.fun:.2e}")
    print_with_timestamp(f"Best loss achieved: {best_loss:.2e}")
    print_with_timestamp(f"Total iterations: {result.nit}")
    print_with_timestamp(f"Function evaluations: {result.nfev}")
    print_with_timestamp(f"Optimization time: {optimization_time:.1f} seconds")
    print_with_timestamp(f"Total runtime: {total_time:.1f} seconds")
    
    print_with_timestamp(f"\nOptimized parameter ranges:")
    print_with_timestamp(f"a (horizontal flow): [{a_opt.min():.4f}, {a_opt.max():.4f}]")
    print_with_timestamp(f"b (vertical flow): [{b_opt.min():.4f}, {b_opt.max():.4f}]")
    print_with_timestamp(f"c (reaction term): [{c_opt.min():.4f}, {c_opt.max():.4f}]")
    
    return result, (u0, uT, cp.array(a_opt), cp.array(b_opt), cp.array(c_opt), n, m)

# Test function with real video
def test_with_video(video_path="handvid.mp4"):
    """
    Test the complete system with a real video file
    """
    print_with_timestamp(f"Starting video-based test with: {video_path}")
    
    if not os.path.exists(video_path):
        print_with_timestamp(f"Video file {video_path} not found. Switching to synthetic test...")
        return test_synthetic_case()
    
    try:
        # Run optimization with video
        result, (u0, uT, a_opt, b_opt, c_opt, n, m) = optimize_pde_parameters(
            video_path, 
            frame_idx=10,          # Start from frame 10
            downsample_factor=8,   # Reduce resolution for faster processing
            max_iterations=20      # Fewer iterations for testing
        )
        
        print_with_timestamp("Verifying optimization results...")
        
        # Test the optimized solution
        dt, t_max = 0.01, 0.1
        dx, dy = 1.0 / m, 1.0 / n
        
        u_final = solve_governing_equation(u0, a_opt, b_opt, c_opt, dt, dx, dy, n, m, t_max)
        final_error = cp.sum((u_final - uT) ** 2).get()
        relative_error = final_error / cp.sum(uT ** 2).get()
        
        print_with_timestamp("Final verification results:")
        print_with_timestamp(f"Absolute error: {final_error:.2e}")
        print_with_timestamp(f"Relative error: {relative_error:.2e}")
        
        success = result.success and final_error < 1.0
        print_with_timestamp(f"Test result: {'SUCCESS' if success else 'FAILED'}")
        
        return success
        
    except FileNotFoundError as e:
        print_with_timestamp(f"File error: {e}")
        return False
    except cv2.error as e:
        print_with_timestamp(f"OpenCV error: {e}")
        return False
    except Exception as e:
        print_with_timestamp(f"Unexpected error in video test: {e}")
        import traceback
        traceback.print_exc()
        return False

# Synthetic test case (fallback when no video available)
def test_synthetic_case():
    """
    Test with synthetic data when video is not available
    """
    print_with_timestamp("Running synthetic test case...")
    
    # Create synthetic data
    n, m = 20, 20
    dx, dy = 1.0 / m, 1.0 / n
    dt, t_max = 0.01, 0.05
    
    print_with_timestamp(f"Creating synthetic test case: {n}×{m} grid")
    
    # Create initial condition (Gaussian blob)
    x = cp.linspace(0, 1, m)
    y = cp.linspace(0, 1, n)
    X, Y = cp.meshgrid(x, y)
    u0 = cp.exp(-10 * ((X - 0.3)**2 + (Y - 0.3)**2))
    
    # Create target condition (shifted Gaussian)
    uT = cp.exp(-10 * ((X - 0.7)**2 + (Y - 0.7)**2))
    
    # Create synthetic optical flow
    a_true = cp.ones((n, m)) * 0.1  # Horizontal flow
    b_true = cp.ones((n, m)) * 0.1  # Vertical flow
    c_true = cp.ones((n, m)) * 0.2  # Reaction term
    
    print_with_timestamp("Synthetic data created")
    
    # Apply boundary conditions
    u0[0,:] = uT[0,:]
    u0[-1,:] = uT[-1,:]
    u0[:,0] = uT[:,0]
    u0[:,-1] = uT[:,-1]
    
    # Initialize parameters
    initial_params = initialize_parameters(n, m, a_true * 0.5, b_true * 0.5, 0.1)
    
    # Set up bounds
    bounds = [(-0.5, 0.5)] * (2 * n * m) + [(0, 1)] * (n * m)
    
    print_with_timestamp("Starting synthetic optimization...")
    optimization_start_time = time.time()
    
    # Run optimization
    result = minimize(
        loss_function, initial_params, 
        args=(u0, uT, dt, dx, dy, n, m, t_max),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 30, 'disp': True}
    )
    
    optimization_time = time.time() - optimization_start_time
    
    print_with_timestamp(f"Synthetic test completed in {optimization_time:.1f} seconds")
    print_with_timestamp(f"Success: {result.success}, Final loss: {result.fun:.2e}")
    
    success = result.success and result.fun < 0.1
    return success

# Main execution
if __name__ == "__main__":
    print_with_timestamp("="*60)
    print_with_timestamp("PDE PARAMETER OPTIMIZATION SYSTEM")
    print_with_timestamp("="*60)
    
    # Check if CUDA is available
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        print_with_timestamp(f"CUDA GPU detected: {device_count} device(s) available")
    except:
        print_with_timestamp("Warning: CUDA not available, performance will be slower")
    
    # Test with video or synthetic data
    video_file = "handvid.mp4"  # Change this to your video file path
    
    if os.path.exists(video_file):
        print_with_timestamp(f"Video file found: {video_file}")
        success = test_with_video(video_file)
    else:
        print_with_timestamp(f"Video file not found: {video_file}")
        print_with_timestamp("Running synthetic test instead")
        success = test_synthetic_case()
    
    print_with_timestamp("="*60)
    if success:
        print_with_timestamp("[SUCCESS] All tests completed successfully!")
    else:
        print_with_timestamp("[FAILED] Test failed!")
    print_with_timestamp("="*60)
    
    # Ensure all GPU operations are completed
    print_with_timestamp("Synchronizing GPU operations...")
    cp.cuda.runtime.deviceSynchronize()
    print_with_timestamp("Program completed")
