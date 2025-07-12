import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

class OpticalFlowSolver:
    def __init__(self, T=1.0, dt_factor=0.5, max_iter=5):
        self.T = T
        self.dt_factor = dt_factor
        self.max_iter = max_iter
        self.iteration_count = 0
        self.force_stop = False
        
    def replicate_padding(self, image, p=8):
        """Add replicate padding around the image"""
        if len(image.shape) == 2:
            return np.pad(image, ((p, p), (p, p)), mode='edge')
        else:
            return np.pad(image, ((p, p), (p, p), (0, 0)), mode='edge')
    
    def remove_padding(self, image, p=8):
        """Remove padding from image"""
        if len(image.shape) == 2:
            return image[p:-p, p:-p]
        else:
            return image[p:-p, p:-p, :]
    
    def compute_gradients(self, I):
        """Compute spatial gradients using central differences"""
        Ix = np.zeros_like(I)
        Iy = np.zeros_like(I)
        
        # Central differences for interior points
        Ix[1:-1, 1:-1] = (I[1:-1, 2:] - I[1:-1, :-2]) / 2.0
        Iy[1:-1, 1:-1] = (I[2:, 1:-1] - I[:-2, 1:-1]) / 2.0
        
        # Handle boundaries
        Ix[:, 0] = I[:, 1] - I[:, 0]
        Ix[:, -1] = I[:, -1] - I[:, -2]
        Iy[0, :] = I[1, :] - I[0, :]
        Iy[-1, :] = I[-1, :] - I[-2, :]
        
        return Ix, Iy
    
    def compute_cip_derivatives(self, I):
        """Compute derivatives needed for CIP method"""
        h, w = I.shape
        
        # Compute first derivatives
        Ix = np.zeros_like(I)
        Iy = np.zeros_like(I)
        
        # Central differences for interior points
        Ix[1:-1, 1:-1] = (I[1:-1, 2:] - I[1:-1, :-2]) / 2.0
        Iy[1:-1, 1:-1] = (I[2:, 1:-1] - I[:-2, 1:-1]) / 2.0
        
        # Boundary conditions for first derivatives
        Ix[:, 0] = I[:, 1] - I[:, 0]
        Ix[:, -1] = I[:, -1] - I[:, -2]
        Iy[0, :] = I[1, :] - I[0, :]
        Iy[-1, :] = I[-1, :] - I[-2, :]
        
        # Compute cross derivative Ixy
        Ixy = np.zeros_like(I)
        
        # Central differences for cross derivative
        for i in range(1, h-1):
            for j in range(1, w-1):
                Ixy[i, j] = (I[i+1, j+1] - I[i+1, j-1] - I[i-1, j+1] + I[i-1, j-1]) / 4.0
        
        # Boundary conditions for cross derivative
        Ixy[0, :] = Ixy[1, :]
        Ixy[-1, :] = Ixy[-2, :]
        Ixy[:, 0] = Ixy[:, 1]
        Ixy[:, -1] = Ixy[:, -2]
        
        return Ix, Iy, Ixy
    
    def cip_interpolation(self, I, Ix, Iy, Ixy, i, j, xi, eta):
        """CIP interpolation using constrained interpolation profile"""
        # Hermite interpolation with derivatives
        # xi and eta are normalized coordinates in [0,1]
        
        # Get values at grid points
        I00 = I[i, j]
        I10 = I[i+1, j] if i+1 < I.shape[0] else I[i, j]
        I01 = I[i, j+1] if j+1 < I.shape[1] else I[i, j]
        I11 = I[i+1, j+1] if i+1 < I.shape[0] and j+1 < I.shape[1] else I[i, j]
        
        # Get derivatives at grid points
        Ix00 = Ix[i, j]
        Ix10 = Ix[i+1, j] if i+1 < I.shape[0] else Ix[i, j]
        Ix01 = Ix[i, j+1] if j+1 < I.shape[1] else Ix[i, j]
        Ix11 = Ix[i+1, j+1] if i+1 < I.shape[0] and j+1 < I.shape[1] else Ix[i, j]
        
        Iy00 = Iy[i, j]
        Iy10 = Iy[i+1, j] if i+1 < I.shape[0] else Iy[i, j]
        Iy01 = Iy[i, j+1] if j+1 < I.shape[1] else Iy[i, j]
        Iy11 = Iy[i+1, j+1] if i+1 < I.shape[0] and j+1 < I.shape[1] else Iy[i, j]
        
        Ixy00 = Ixy[i, j]
        Ixy10 = Ixy[i+1, j] if i+1 < I.shape[0] else Ixy[i, j]
        Ixy01 = Ixy[i, j+1] if j+1 < I.shape[1] else Ixy[i, j]
        Ixy11 = Ixy[i+1, j+1] if i+1 < I.shape[0] and j+1 < I.shape[1] else Ixy[i, j]
        
        # Bicubic Hermite interpolation
        # Basis functions
        xi2 = xi * xi
        xi3 = xi2 * xi
        eta2 = eta * eta
        eta3 = eta2 * eta
        
        # Hermite basis functions
        h00 = 2*xi3 - 3*xi2 + 1
        h10 = xi3 - 2*xi2 + xi
        h01 = -2*xi3 + 3*xi2
        h11 = xi3 - xi2
        
        g00 = 2*eta3 - 3*eta2 + 1
        g10 = eta3 - 2*eta2 + eta
        g01 = -2*eta3 + 3*eta2
        g11 = eta3 - eta2
        
        # Interpolate value
        result = (h00*g00*I00 + h10*g00*Ix00 + h00*g10*Iy00 + h10*g10*Ixy00 +
                 h01*g00*I10 + h11*g00*Ix10 + h01*g10*Iy10 + h11*g10*Ixy10 +
                 h00*g01*I01 + h10*g01*Ix01 + h00*g11*Iy01 + h10*g11*Ixy01 +
                 h01*g01*I11 + h11*g01*Ix11 + h01*g11*Iy11 + h11*g11*Ixy11)
        
        return result
    
    def solve_convection_pde_cip(self, I_initial, u, v, T_half):
        """Solve convection PDE using high-order CIP method"""
        h, w = I_initial.shape
        
        # Ensure arrays are properly typed
        I_initial = np.asarray(I_initial, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        
        # Compute stable time step using CFL condition
        max_u = np.max(np.abs(u))
        max_v = np.max(np.abs(v))
        max_vel = max(max_u, max_v)
        
        if max_vel > 1e-10:
            dt = self.dt_factor / max_vel
        else:
            dt = 0.01
            
        dt = min(dt, T_half / 100)  # Ensure sufficient time steps
        
        # Initialize arrays for CIP method
        I = I_initial.copy()
        t = 0
        
        print(f"      CIP method: dt={dt:.6f}, max_vel={max_vel:.6f}")
        
        while t < T_half:
            if t + dt > T_half:
                dt = T_half - t
                
            # Compute derivatives for CIP interpolation
            Ix, Iy, Ixy = self.compute_cip_derivatives(I)
            
            # Store current state
            I_old = I.copy()
            Ix_old = Ix.copy()
            Iy_old = Iy.copy()
            Ixy_old = Ixy.copy()
            
            # CIP advection step
            I_new = np.zeros_like(I)
            Ix_new = np.zeros_like(Ix)
            Iy_new = np.zeros_like(Iy)
            Ixy_new = np.zeros_like(Ixy)
            
            for i in range(h):
                for j in range(w):
                    # Departure point (backward tracing)
                    x_dep = j - u[i, j] * dt
                    y_dep = i - v[i, j] * dt
                    
                    # Clamp to domain
                    x_dep = np.clip(x_dep, 0, w-1.001)
                    y_dep = np.clip(y_dep, 0, h-1.001)
                    
                    # Grid indices
                    i_dep = int(y_dep)
                    j_dep = int(x_dep)
                    
                    # Normalized coordinates
                    xi = x_dep - j_dep
                    eta = y_dep - i_dep
                    
                    # Ensure indices are within bounds
                    i_dep = min(i_dep, h-2)
                    j_dep = min(j_dep, w-2)
                    
                    # CIP interpolation for value
                    I_new[i, j] = self.cip_interpolation(I_old, Ix_old, Iy_old, Ixy_old, 
                                                        i_dep, j_dep, xi, eta)
                    
                    # CIP interpolation for derivatives (simplified)
                    # For higher accuracy, derivatives should also be properly interpolated
                    # Here we use a simpler approach
                    if i_dep + 1 < h and j_dep + 1 < w:
                        # Bilinear interpolation for derivatives as approximation
                        w00 = (1-xi) * (1-eta)
                        w10 = xi * (1-eta)
                        w01 = (1-xi) * eta
                        w11 = xi * eta
                        
                        Ix_new[i, j] = (w00 * Ix_old[i_dep, j_dep] + 
                                       w10 * Ix_old[i_dep, j_dep+1] +
                                       w01 * Ix_old[i_dep+1, j_dep] + 
                                       w11 * Ix_old[i_dep+1, j_dep+1])
                        
                        Iy_new[i, j] = (w00 * Iy_old[i_dep, j_dep] + 
                                       w10 * Iy_old[i_dep, j_dep+1] +
                                       w01 * Iy_old[i_dep+1, j_dep] + 
                                       w11 * Iy_old[i_dep+1, j_dep+1])
                        
                        Ixy_new[i, j] = (w00 * Ixy_old[i_dep, j_dep] + 
                                        w10 * Ixy_old[i_dep, j_dep+1] +
                                        w01 * Ixy_old[i_dep+1, j_dep] + 
                                        w11 * Ixy_old[i_dep+1, j_dep+1])
                    else:
                        Ix_new[i, j] = Ix_old[i_dep, j_dep]
                        Iy_new[i, j] = Iy_old[i_dep, j_dep]
                        Ixy_new[i, j] = Ixy_old[i_dep, j_dep]
            
            # Update solution
            I = I_new
            t += dt
            
        return I
    
    def initial_optical_flow(self, I_initial, I_final):
        """Compute initial optical flow estimate using gradient-based method"""
        print("    Computing initial optical flow...")
        
        # Compute spatial and temporal gradients
        Ix, Iy = self.compute_gradients(I_initial)
        It = I_final - I_initial
        
        # Lucas-Kanade optical flow equations
        denominator = Ix**2 + Iy**2 + 1e-10
        
        # Simple optical flow estimation
        u = -It * Ix / denominator
        v = -It * Iy / denominator
        
        # Smooth the flow fields to reduce noise
        u = gaussian_filter(u, sigma=1.0)
        v = gaussian_filter(v, sigma=1.0)
        
        # Clip extreme values
        u = np.clip(u, -5, 5)
        v = np.clip(v, -5, 5)
        
        print(f"    Initial flow range: u=[{np.min(u):.3f}, {np.max(u):.3f}], v=[{np.min(v):.3f}, {np.max(v):.3f}]")
        
        return u, v
    
    def solve_source_pde(self, I_initial, c, d, T_half):
        """Solve source PDE: I_t = c + d*I, T/2 < t < T"""
        h, w = I_initial.shape
        
        # Ensure arrays are properly typed
        I_initial = np.asarray(I_initial, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        
        # Compute stable time step
        max_d = np.max(np.abs(d))
        if max_d > 1e-10:
            dt = min(0.01, 0.5 / max_d)
        else:
            dt = 0.01
            
        dt = min(dt, T_half / 100)  # Ensure sufficient time steps
        
        I = I_initial.copy()
        t = 0
        
        while t < T_half:
            if t + dt > T_half:
                dt = T_half - t
                
            # Source equation: I_t = c + d*I
            I_new = I + dt * (c + d * I)
            
            I = I_new
            t += dt
            
        return I
    
    def convection_objective(self, params, I_initial, I_final, channel_name):
        """Objective function for convection part optimization"""
        try:
            if self.force_stop:
                return 1e10
                
            h, w = I_initial.shape
            
            # Extract flow parameters
            u = params[:h*w].reshape(h, w)
            v = params[h*w:].reshape(h, w)
            
            # Solve convection PDE for T/2 using CIP method
            I_convection = self.solve_convection_pde_cip(I_initial, u, v, self.T/2)
            
            # Compute MSE with final image
            mse = np.mean((I_convection - I_final)**2)
            reg_term = 0.001 * (np.mean(u**2) + np.mean(v**2))
            total_error = mse + reg_term
            
            self.iteration_count += 1
            if self.iteration_count % 5 == 0:
                print(f"    Convection Iteration {self.iteration_count}: Channel {channel_name}")
                print(f"      Current MSE: {mse:.6f}")
                print(f"      Total error: {total_error:.6f}")
            
            if self.iteration_count >= self.max_iter:
                print(f"    Maximum iterations ({self.max_iter}) reached for {channel_name} convection")
                self.force_stop = True
                return total_error
                
            return total_error
            
        except Exception as e:
            print(f"    Error in convection objective: {e}")
            self.force_stop = True
            return 1e10
    
    def source_objective(self, params, I_initial, I_final, channel_name):
        """Objective function for source part optimization"""
        try:
            if self.force_stop:
                return 1e10
                
            h, w = I_initial.shape
            
            # Extract source parameters
            c = params[:h*w].reshape(h, w)
            d = params[h*w:].reshape(h, w)
            
            # Solve source PDE for T/2
            I_source = self.solve_source_pde(I_initial, c, d, self.T/2)
            
            # Compute MSE with final image
            mse = np.mean((I_source - I_final)**2)
            reg_term = 0.001 * (np.mean(c**2) + np.mean(d**2))
            total_error = mse + reg_term
            
            self.iteration_count += 1
            if self.iteration_count % 5 == 0:
                print(f"    Source Iteration {self.iteration_count}: Channel {channel_name}")
                print(f"      Current MSE: {mse:.6f}")
                print(f"      Total error: {total_error:.6f}")
            
            if self.iteration_count >= self.max_iter:
                print(f"    Maximum iterations ({self.max_iter}) reached for {channel_name} source")
                self.force_stop = True
                return total_error
                
            return total_error
            
        except Exception as e:
            print(f"    Error in source objective: {e}")
            self.force_stop = True
            return 1e10
    
    def solve_single_channel(self, I_initial, I_final, channel_name):
        print(f"\n  Solving for {channel_name} channel...")
        
        h, w = I_initial.shape
        
        # Step 1: Solve convection part using CIP method
        print(f"    Step 1: Solving convection part for {channel_name} using CIP method")
        
        # Initial guess for flow
        u0, v0 = self.initial_optical_flow(I_initial, I_final)
        
        # Flatten parameters for convection
        convection_params0 = np.concatenate([u0.flatten(), v0.flatten()])
        
        # Set bounds for flow parameters
        flow_bounds = [(-10, 10)] * (2 * h * w)
        
        # Reset iteration counter and force stop flag
        self.iteration_count = 0
        self.force_stop = False
        
        # Optimize convection part
        result_convection = minimize(
            self.convection_objective,
            convection_params0,
            args=(I_initial, I_final, channel_name),
            method='L-BFGS-B',
            bounds=flow_bounds,
            options={
                'maxiter': min(self.max_iter, 10),
                'maxfun': min(self.max_iter * 10, 10),
                'disp': False,
                'gtol': 1e-3,
                'ftol': 1e-3
            }
        )
        
        # Extract optimized flow parameters
        if len(result_convection.x) == len(convection_params0):
            u_opt = result_convection.x[:h*w].reshape(h, w)
            v_opt = result_convection.x[h*w:].reshape(h, w)
        else:
            u_opt, v_opt = u0, v0
        
        # Solve convection PDE with optimized flow to get I_convection(T/2)
        I_convection_half = self.solve_convection_pde_cip(I_initial, u_opt, v_opt, self.T/2)
        
        print(f"    Convection optimization completed for {channel_name} (CIP method)")
        
        # Step 2: Solve source part
        print(f"    Step 2: Solving source part for {channel_name}")
        
        # Initial guess for source parameters
        c0 = np.ones((h, w))*0.0001
        d0 = np.ones((h, w))*0.001
        
        # Flatten parameters for source
        source_params0 = np.concatenate([c0.flatten(), d0.flatten()])
        
        # Set bounds for source parameters
        source_bounds = [(-2, 2)] * (2 * h * w)
        
        # Reset iteration counter and force stop flag
        self.iteration_count = 0
        self.force_stop = False
        
        # Optimize source part
        result_source = minimize(
            self.source_objective,
            source_params0,
            args=(I_convection_half, I_final, channel_name),
            method='L-BFGS-B',
            bounds=source_bounds,
            options={
                'maxiter': min(self.max_iter, 5),
                'maxfun': min(self.max_iter * 10, 3),
                'disp': False,
                'gtol': 1e-10,
                'ftol': 1e-10
            }
        )
        
        # Extract optimized source parameters
        if len(result_source.x) == len(source_params0):
            c_opt = result_source.x[:h*w].reshape(h, w)
            d_opt = result_source.x[h*w:].reshape(h, w)
        else:
            c_opt, d_opt = c0, d0
        
        # Solve source PDE with optimized parameters to get final result
        I_computed = self.solve_source_pde(I_convection_half, c_opt, d_opt, self.T/2)
        
        print(f"    Source optimization completed for {channel_name}")
        
        return I_computed, u_opt, v_opt, c_opt, d_opt, I_convection_half
    
    def solve(self, I_initial, I_final, p=8):
        """Main solving function following split algorithm with CIP method"""
        print("Starting optical flow solver with split convection-source approach...")
        print("Using high-order CIP method for convection equation solving")
        
        # Ensure images are float and in range [0,1]
        I_initial = I_initial.astype(np.float64)
        I_final = I_final.astype(np.float64)
        
        if np.max(I_initial) > 1:
            I_initial = I_initial / 255.0
        if np.max(I_final) > 1:
            I_final = I_final / 255.0
        
        print(f"Image size: {I_initial.shape}")
        print(f"Padding size: {p}")
        print(f"Time split: Convection (0 to {self.T/2}) [CIP method], Source ({self.T/2} to {self.T})")
        
        # Step 1: Perform replicate padding
        print("\nStep 1: Applying replicate padding...")
        I_initial_padded = self.replicate_padding(I_initial, p)
        I_final_padded = self.replicate_padding(I_final, p)
        print(f"Padded image size: {I_initial_padded.shape}")
        
        # Step 2: Decouple image in r,g,b channels
        print("\nStep 2: Decoupling RGB channels...")
        
        if len(I_initial.shape) == 3 and I_initial.shape[2] == 3:
            # Color image
            print("Processing color image with RGB channels")
            
            # Extract channels
            r_initial = I_initial_padded[:, :, 0]
            g_initial = I_initial_padded[:, :, 1]
            b_initial = I_initial_padded[:, :, 2]
            
            r_final = I_final_padded[:, :, 0]
            g_final = I_final_padded[:, :, 1]
            b_final = I_final_padded[:, :, 2]
            
            # Step 3: Solve split minimization problem for each channel
            print("\nStep 3: Solving split minimization problem for each channel...")
            
            # Solve for R channel
            r_computed, u_r, v_r, c_r, d_r, r_conv_half = self.solve_single_channel(r_initial, r_final, "Red")
            
            # Solve for G channel
            g_computed, u_g, v_g, c_g, d_g, g_conv_half = self.solve_single_channel(g_initial, g_final, "Green")
            
            # Solve for B channel
            b_computed, u_b, v_b, c_b, d_b, b_conv_half = self.solve_single_channel(b_initial, b_final, "Blue")
            
            # Step 6: Extract original image size and compose I_computed
            print("\nStep 6: Composing final image from computed channels...")
            
            # Remove padding from computed channels
            r_computed_unpadded = self.remove_padding(r_computed, p)
            g_computed_unpadded = self.remove_padding(g_computed, p)
            b_computed_unpadded = self.remove_padding(b_computed, p)
            
            # Compose final image
            I_computed = np.stack([r_computed_unpadded, g_computed_unpadded, b_computed_unpadded], axis=2)
            
            # Store channel results for plotting
            self.channel_results = {
                'r': {'initial': I_initial[:,:,0], 'final': I_final[:,:,0], 'computed': r_computed_unpadded,
                      'convection_half': self.remove_padding(r_conv_half, p)},
                'g': {'initial': I_initial[:,:,1], 'final': I_final[:,:,1], 'computed': g_computed_unpadded,
                      'convection_half': self.remove_padding(g_conv_half, p)},
                'b': {'initial': I_initial[:,:,2], 'final': I_final[:,:,2], 'computed': b_computed_unpadded,
                      'convection_half': self.remove_padding(b_conv_half, p)}
            }
            
            # Store parameters
            self.parameters = {
                'r': {'u': u_r, 'v': v_r, 'c': c_r, 'd': d_r},
                'g': {'u': u_g, 'v': v_g, 'c': c_g, 'd': d_g},
                'b': {'u': u_b, 'v': v_b, 'c': c_b, 'd': d_b}
            }
            
        else:
            # Grayscale image
            print("Processing grayscale image")
            
            # Step 3: Solve split minimization problem for single channel
            print("\nStep 3: Solving split minimization problem for grayscale channel...")
            I_computed_padded, u_opt, v_opt, c_opt, d_opt, I_conv_half = self.solve_single_channel(
                I_initial_padded, I_final_padded, "Gray"
            )
            
            # Step 6: Extract original image size
            I_computed = self.remove_padding(I_computed_padded, p)
            
            # Store results for plotting
            self.channel_results = {
                'gray': {'initial': I_initial, 'final': I_final, 'computed': I_computed,
                        'convection_half': self.remove_padding(I_conv_half, p)}
            }
            
            self.parameters = {
                'gray': {'u': u_opt, 'v': v_opt, 'c': c_opt, 'd': d_opt}
            }
        
        print("\nSplit optimization with CIP method complete!")
        return I_computed
    
    def plot_results(self, I_initial, I_final, I_computed):
            """Plot results showing convection and source steps"""
            print("\nStep 7: Plotting results...")
            
            if len(I_initial.shape) == 3:
                # Color image plotting with convection intermediate step
                fig, axes = plt.subplots(4, 5, figsize=(20, 16))
                
                # Main images (first row)
                axes[0, 0].imshow(I_initial)
                axes[0, 0].set_title('I_initial')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(I_final)
                axes[0, 1].set_title('I_final')
                axes[0, 1].axis('off')
                
                # Show convection result at T/2
                I_conv_half = np.stack([
                    self.channel_results['r']['convection_half'],
                    self.channel_results['g']['convection_half'],
                    self.channel_results['b']['convection_half']
                ], axis=2)
                axes[0, 2].imshow(I_conv_half)
                axes[0, 2].set_title('I_convection(T/2)')
                axes[0, 2].axis('off')
                
                axes[0, 3].imshow(I_computed)
                axes[0, 3].set_title('I_computed')
                axes[0, 3].axis('off')
                
                error = np.abs(I_final - I_computed)
                axes[0, 4].imshow(error)
                axes[0, 4].set_title('|I_final - I_computed|')
                axes[0, 4].axis('off')
                
                # Channel-wise plots
                channels = ['r', 'g', 'b']
                channel_names = ['Red', 'Green', 'Blue']
                
                for i, (ch, name) in enumerate(zip(channels, channel_names)):
                    row = i + 1
                    ch_data = self.channel_results[ch]
                    
                    # Initial
                    axes[row, 0].imshow(ch_data['initial'], cmap='gray', vmin=0, vmax=1)
                    axes[row, 0].set_title(f'{name} Initial')
                    axes[row, 0].axis('off')
                    
                    # Final
                    axes[row, 1].imshow(ch_data['final'], cmap='gray', vmin=0, vmax=1)
                    axes[row, 1].set_title(f'{name} Final')
                    axes[row, 1].axis('off')
                    
                    # Convection at T/2
                    axes[row, 2].imshow(ch_data['convection_half'], cmap='gray', vmin=0, vmax=1)
                    axes[row, 2].set_title(f'{name} Convection(T/2)')
                    axes[row, 2].axis('off')
                    
                    # Computed
                    axes[row, 3].imshow(ch_data['computed'], cmap='gray', vmin=0, vmax=1)
                    axes[row, 3].set_title(f'{name} Computed')
                    axes[row, 3].axis('off')
                    
                    # Error
                    ch_error = np.abs(ch_data['final'] - ch_data['computed'])
                    im = axes[row, 4].imshow(ch_error, cmap='hot')
                    axes[row, 4].set_title(f'{name} Error')
                    axes[row, 4].axis('off')
                    plt.colorbar(im, ax=axes[row, 4])
            
            else:
                # Grayscale image plotting
                fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                
                ch_data = self.channel_results['gray']
                
                axes[0].imshow(ch_data['initial'], cmap='gray', vmin=0, vmax=1)
                axes[0].set_title('I_initial')
                axes[0].axis('off')
                
                axes[1].imshow(ch_data['final'], cmap='gray', vmin=0, vmax=1)
                axes[1].set_title('I_final')
                axes[1].axis('off')
                
                axes[2].imshow(ch_data['convection_half'], cmap='gray', vmin=0, vmax=1)
                axes[2].set_title('I_convection(T/2)')
                axes[2].axis('off')
                
                axes[3].imshow(ch_data['computed'], cmap='gray', vmin=0, vmax=1)
                axes[3].set_title('I_computed')
                axes[3].axis('off')
                
                error = np.abs(ch_data['final'] - ch_data['computed'])
                im = axes[4].imshow(error, cmap='hot')
                axes[4].set_title('|I_final - I_computed|')
                axes[4].axis('off')
                plt.colorbar(im, ax=axes[4])
            
            plt.tight_layout()
            plt.show()
            
            # Print final statistics
            print("\nFinal Results (Split Method):")
            if len(I_initial.shape) == 3:
                for ch in ['r', 'g', 'b']:
                    ch_data = self.channel_results[ch]
                    params = self.parameters[ch]
                    ch_error = np.abs(ch_data['final'] - ch_data['computed'])
                    mse = np.mean((ch_data['final'] - ch_data['computed'])**2)
                    
                    print(f"\n{ch.upper()} Channel:")
                    print(f"  Convection: I_t + u*I_x + v*I_y = 0, 0 < t < {self.T/2}")
                    print(f"  Source: I_t = {np.mean(params['c']):.4f} + {np.mean(params['d']):.4f}*I, {self.T/2} < t < {self.T}")
                    print(f"  MSE: {mse:.6f}")
                    print(f"  Max Error: {np.max(ch_error):.6f}")
                    print(f"  Mean Error: {np.mean(ch_error):.6f}")
            else:
                ch_data = self.channel_results['gray']
                params = self.parameters['gray']
                error = np.abs(ch_data['final'] - ch_data['computed'])
                mse = np.mean((ch_data['final'] - ch_data['computed'])**2)
                
                print(f"\nGrayscale Channel:")
                print(f"  Convection: I_t + u*I_x + v*I_y = 0, 0 < t < {self.T/2}")
                print(f"  Source: I_t = {np.mean(params['c']):.4f} + {np.mean(params['d']):.4f}*I, {self.T/2} < t < {self.T}")
                print(f"  MSE: {mse:.6f}")
                print(f"  Max Error: {np.max(error):.6f}")
                print(f"  Mean Error: {np.mean(error):.6f}")
    
    def create_test_images():
        """Create test images for demonstration"""
        # Create a simple moving pattern with color
        h, w = 64, 64
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Create RGB initial image
        I_initial = np.zeros((h, w, 3))
        
        # Red channel - circle
        I_initial[:, :, 0] = np.exp(-((x-20)**2 + (y-20)**2)/100)
        
        # Green channel - different pattern
        I_initial[:, :, 1] = np.exp(-((x-25)**2 + (y-25)**2)/80)
        
        # Blue channel - another pattern
        I_initial[:, :, 2] = np.exp(-((x-30)**2 + (y-15)**2)/120)
        
        # Create RGB final image - moved and brightness changed
        I_final = np.zeros((h, w, 3))
        
        # Red channel - moved circle with brightness change
        I_final[:, :, 0] = 0.8 * np.exp(-((x-30)**2 + (y-25)**2)/100) + 0.1
        
        # Green channel - moved with different brightness change
        I_final[:, :, 1] = 0.9 * np.exp(-((x-35)**2 + (y-30)**2)/80) + 0.05
        
        # Blue channel - moved with brightness change
        I_final[:, :, 2] = 0.7 * np.exp(-((x-40)**2 + (y-20)**2)/120) + 0.15
        
        return I_initial, I_final
    
    def main():
        """Main function with split algorithm implementation"""
        try:
            print("=== Optical Flow with Split Convection-Source Method ===")
            print("Following split algorithm instructions...\n")
            
            # Create test images
            I_initial, I_final = create_test_images()
            
            # Create solver with split approach
            solver = OpticalFlowSolver(T=1.0, max_iter=10)
            
            # Solve following split algorithm steps
            I_computed = solver.solve(I_initial, I_final, p=8)
            
            # Plot results
            # Plot results as specified
            solver.plot_results(I_initial, I_final, I_computed)
            
            print("\n=== Algorithm completed successfully! ===")
            
        except Exception as e:
            print(f"Error in main function: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
