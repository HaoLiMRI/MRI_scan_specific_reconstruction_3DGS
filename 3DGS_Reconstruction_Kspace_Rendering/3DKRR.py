import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
# import torch.nn.functional as F
# import torch.distributed as dist


# --- Step 1: Define 3D Gaussian Model --- #
def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion (x, y, z, w) to rotation matrix.
    q: Tensor of shape (batch_size, 4)
    """
    batch_size, height, width, depth = q.shape[:4]
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    R = torch.stack([
        1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)
    ], dim=-1).view(batch_size, height, width, depth, 3, 3)
    
    
    return R

def gaussian_distribution(x, mu, sigma, rotation): #voxel_center, grid+anchor, sigma, rotation
    """
    Compute the Gaussian distribution values for a given point.
    """
    x = x.unsqueeze(1).unsqueeze(2).unsqueeze(3)   # [5, 1, 1, 1, 3] 
    x_rotated = torch.matmul(rotation, (x.expand(mu.shape) - mu).unsqueeze(-1)).squeeze(-1)
    exponent = -0.5 * torch.sum((x_rotated / sigma) ** 2, dim=-1)
    return torch.exp(exponent) / (torch.prod(sigma, dim=-1) * (2 * math.pi) ** 1.5)

# --- Step 2: Define MRI Encoding Model --- #
def compute_phase_encoding(kx, ky, kz, scale_x, scale_y, scale_z):
    """Phase shifts in encoding steps"""
    phase_x = torch.linspace(-(scale_x / 2 - kx), scale_x / 2 - kx - 1, steps=scale_x).view(scale_x, 1, 1).expand(scale_x, scale_y, scale_z) * torch.pi
    phase_y = torch.linspace(-(scale_y / 2 - ky), scale_y / 2 - ky - 1, steps=scale_y).view(1, scale_y, 1).expand(scale_x, scale_y, scale_z) * torch.pi
    if scale_z>1:
        phase_z = torch.linspace(-(scale_z / 2 - kz), scale_z / 2 - kz - 1, steps=scale_z).view(1, 1, scale_z).expand(scale_x, scale_y, scale_z) * torch.pi
    else:
        phase_z = 0
    return phase_x + phase_y + phase_z

# --- Step 3: Compute K-space --- #
def compute_k_space(params):
    """Compute K-space using 3D Gaussian Distributions"""
    b, h, w, d = params.shape[:4]
    signal_intensity = params[..., 0] # Signal intensity
    initial_phase = params[..., 1] # Signal phase
    """
    quaternion = params[..., 2:6]  # Rotation quaternion
    sigma = params[..., 6:9]  # Gaussian variances
    anchor = params[..., 9:12]  # Anchor positions

    rotation = quaternion_to_rotation_matrix(quaternion).view(b, h, w, d, 3, 3)
    
    # Prepare voxel grid
    grid = torch.stack(torch.meshgrid(
        torch.linspace(0, h - 1, h),
        torch.linspace(0, w - 1, w),
        torch.linspace(0, d - 1, d),
        indexing='ij'
    ), dim=-1).to(params.device)  # (h, w, d, 3)
    grid = grid.unsqueeze(0).expand(b, h, w, d, 3) + 0.5  # Broadcast to batch
    """
    
    # Initialize k-space
    k_space = torch.zeros((b, h, w, d), dtype=torch.complex64).to(params.device)
    signal = torch.zeros((b, h, w, d), dtype=torch.complex64).to(params.device)

    """
    for i in range(h):
        for j in range(w):
            for k in range(d):
                voxel_center = grid[:, i, j, k]  # Center of voxel
                gaussian_values = gaussian_distribution(voxel_center, grid+anchor, sigma, rotation).to(params.device)
                signal[:, i, j, k] = torch.sum(signal_intensity * torch.exp(1j * initial_phase) * gaussian_values, dim = (1, 2, 3))
    """
                

    for i in range(h):
        for j in range(w):
            for k in range(d):
#                voxel_center = grid[:, i, j, k]  # Center of voxel
#                gaussian_values = gaussian_distribution(voxel_center, grid+anchor, sigma, rotation)
                phase_encoding = compute_phase_encoding(i, j, k, h, w, d).to(params.device)
#                signal = gaussian_values * signal_intensity * torch.exp(1j * (initial_phase + phase_encoding))
#                signal = signal * torch.exp(1j * phase_encoding)
                signal = signal_intensity * torch.exp(1j * (initial_phase + phase_encoding))
                k_space[:, i, j, k] = torch.sum(signal, dim=(1, 2, 3))
    
    return k_space, signal

# --- Step 4: Extract k-space Mask --- #
def extract_kspace_mask(kspace_gt):
    """Extract mask from 3D k-space ground truth"""
    kspace_2d = torch.sum(kspace_gt.abs(), dim=0)  # Sum in N_x direction
    mask = (kspace_2d > 0).to(dtype=torch.uint8)  # Generate 2D mask in y-z plane
    mask = mask.unsqueeze(0).expand_as(kspace_gt)  # Expand mask to (N_x, N_y, N_z)
    return mask

# --- Step 5: Initialize Params --- #
def initialize_params(kspace_gt):
    batch_size, h, w, d = kspace_gt.shape
    """Initiate 3D Gaussian matrix (batch_size, h, w, d, 12)"""
    """
    params = torch.randn((batch_size, h, w, d, 12), dtype=torch.float32)#, requires_grad=True)
    with torch.no_grad():
        params[..., 0] = torch.abs(params[..., 0].data)  # Positive intensity values
        params[..., 1] = (params[..., 1].data % (2 * np.pi)) - np.pi  # Phase in the range of [-π, π]
        params[..., 2:6] = params[..., 2:6].data / torch.norm(params[..., 2:6].data, dim=-1, keepdim=True)  # Normalized quaternion values
        params[..., 6:9] = torch.abs(params[..., 6:9].data) + 1e-3  # Positive variance values
        params[..., 9:12] = torch.clamp(params[..., 9:12].data, -0.5, 0.5)  # Distance to anchor in the range of [-0.5, 0.5]
    """    
    params = torch.randn((batch_size, h, w, d, 2), dtype=torch.float32)
    with torch.no_grad():
        params[..., 0] = torch.abs(params[..., 0].data)  # Positive intensity values
        params[..., 1] = (params[..., 1].data % (2 * np.pi)) - np.pi  # Phase in the range of [-π, π]
        
    return params

def tv_loss_3d(img):
    # (batch_size, height, width, depth)
    batch_size, h, w, d = img.size()

    h_variance = torch.pow(img[:, 1:, :, :] - img[:, :-1, :, :], 2).sum(dim=(1, 2, 3))
    w_variance = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum(dim=(1, 2, 3))
    if d>1:
        d_variance = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum(dim=(1, 2, 3))
    else:
        d_variance = 0

    return (h_variance + w_variance + d_variance) / (h * w * d)

# --- Step 6: Training Function --- #
def train_model(kspace_gt, num_epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kspace_gt = kspace_gt.to(device)
    params = initialize_params(kspace_gt).to(device).requires_grad_(True)
    mask = extract_kspace_mask(kspace_gt).to(device)
    optimizer = optim.Adam([params], lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs, eta_min = 1e-4, last_epoch = -1)
    loss_fn = nn.L1Loss()
    

    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        kspace_pred, signal = compute_k_space(params)
        loss = loss_fn(kspace_pred * mask, kspace_gt) + tv_loss_3d(signal)
        loss.backward()
        optimizer.step()
        
        """
        with torch.no_grad():
            # Reapply parameter constraints after update
            params[..., 0] = torch.abs(params[..., 0])  # Ensure intensity is positive
            params[..., 1] = (params[..., 1] % (2 * np.pi)) - np.pi  # Keep phase in [-π, π]
            params[..., 2:6] = params[..., 2:6] / torch.norm(params[..., 2:6], dim=-1, keepdim=True)  # Normalize quaternion
            params[..., 6:9] = torch.abs(params[..., 6:9]) + 1e-3  # Ensure variance is positive
            params[..., 9:12] = torch.clamp(params[..., 9:12], -0.5, 0.5)  # Keep anchor distance within [-0.5, 0.5]
            # Synchronize signal intensity across batch
#            avg_intensity = params[..., 0].mean()
#            params[..., 0] = avg_intensity
        """
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
            
        scheduler.step()


# --- Step 7: Testing Code --- #
if __name__ == "__main__":
        
    
    batch_size, h, w, d = 1, 200, 200, 1  # Small test dimensions
    kspace_gt = torch.randn((batch_size, h, w, d), dtype=torch.complex64).cuda()  # Random complex k-space
    kspace_gt[:,:,0:-1:2,:] = 0
#    kspace_gt[:,:,7,:] = 0
    kspace_gt[:,:,:,0:-1:2] = 0
#    kspace_gt[:,:,:,7] = 0
    
#    print("Initial k-space ground truth:")
#    print(kspace_gt)
    
    train_model(kspace_gt, num_epochs=1000, lr=0.1)  # Run a short training session


"""
def compute_k_space(params):
    #Compute K-space using 3D Gaussian Distributions with Parallel Computation
    b, h, w, d = params.shape[:4]
    signal_intensity = params[..., 0]  # Signal intensity
    initial_phase = params[..., 1]  # Signal phase
    quaternion = params[..., 2:6]  # Rotation quaternion
    sigma = params[..., 6:9]  # Gaussian variances
    anchor = params[..., 9:12]  # Anchor positions

    # Compute rotation matrices
    rotation = quaternion_to_rotation_matrix(quaternion).view(b, h, w, d, 3, 3)

    # Prepare voxel grid (vectorized)
    grid = torch.stack(torch.meshgrid(
        torch.linspace(0, h - 1, h),
        torch.linspace(0, w - 1, w),
        torch.linspace(0, d - 1, d),
        indexing='ij'
    ), dim=-1).to(params.device)  # (h, w, d, 3)
    
    grid = grid.unsqueeze(0).expand(b, h, w, d, 3) + 0.5  # Broadcast to batch

    # Compute Gaussian values in parallel using einsum
    x_rotated = torch.einsum('bhwdk,bhwdkl->bhwdl', (grid - (grid + anchor)), rotation)
    gaussian_exponent = -0.5 * torch.sum((x_rotated / sigma) ** 2, dim=-1)
    gaussian_values = torch.exp(gaussian_exponent) / (torch.prod(sigma, dim=-1) * (2 * torch.pi) ** 1.5)

    # Compute signal efficiently
    signal = torch.einsum('bhwd,bhwd->bhwd', signal_intensity * torch.exp(1j * initial_phase), gaussian_values)

    # Compute phase encoding in parallel
    k_idx = torch.arange(h, device=params.device).view(h, 1, 1).expand(h, w, d)
    phase_x = ((-h / 2 + k_idx) * torch.pi).unsqueeze(0).expand(b, h, w, d)
    
    k_idx = torch.arange(w, device=params.device).view(1, w, 1).expand(h, w, d)
    phase_y = ((-w / 2 + k_idx) * torch.pi).unsqueeze(0).expand(b, h, w, d)

    k_idx = torch.arange(d, device=params.device).view(1, 1, d).expand(h, w, d)
    phase_z = ((-d / 2 + k_idx) * torch.pi).unsqueeze(0).expand(b, h, w, d)

    phase_encoding = phase_x + phase_y + phase_z

    # Apply phase encoding and sum across batch efficiently
    k_space = torch.einsum('bhwd,bhwd->bhwd', signal, torch.exp(1j * phase_encoding)).sum(dim=(1, 2, 3))

    return k_space
"""