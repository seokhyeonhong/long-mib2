import os
import torch

def L2P(pred_global_p, GT_global_p, context_frames):
    # pred_global_p: (B, T, J, 3)
    # GT_global_p: (B, T, J, 3)
    # return: (B, T, J)
    
    pred_global_p = pred_global_p[:, context_frames:]
    GT_global_p   = GT_global_p[:, context_frames:]
    # L2P
    norm = torch.norm(pred_global_p - GT_global_p, dim=-1)
    return torch.sum(norm, dim=-1)

def L2Q(pred_global_Q, GT_global_Q, context_frames):
    # pred_global_Q: (B, T, J, 4)
    # GT_global_Q: (B, T, J, 4)

    pred_global_Q = pred_global_Q[:, context_frames:]
    GT_global_Q   = GT_global_Q[:, context_frames:]

    B, T, J, _ = pred_global_Q.shape
    w_positive = (pred_global_Q[..., 0:1] > 0).float()
    pred_global_Q = pred_global_Q * w_positive + (1 - w_positive) * (-pred_global_Q)
    
    w_positive = (GT_global_Q[..., 0:1] > 0).float()
    GT_global_Q = GT_global_Q * w_positive + (1 - w_positive) * (-GT_global_Q)
    
    # L2Q
    norm = torch.norm(pred_global_Q - GT_global_Q, dim=-1)
    return torch.sum(norm, dim=-1)

def NPSS(pred, GT):
    # GT: (B, T, D)
    # pred: (B, T, D)

    # Fourier coefficients along the time dimension
    GT_fourier_coeffs = torch.real(torch.fft.fft(GT, dim=1))
    pred_fourier_coeffs = torch.real(torch.fft.fft(pred, dim=1))

    # square of the Fourier coefficients
    GT_power = torch.square(GT_fourier_coeffs)
    pred_power = torch.square(pred_fourier_coeffs)

    # sum of powers over time
    GT_power_sum = torch.sum(GT_power, dim=1)
    pred_power_sum = torch.sum(pred_power, dim=1)

    # normalize powers with total
    GT_power_norm = GT_power / GT_power_sum.unsqueeze(1)
    pred_power_norm = pred_power / pred_power_sum.unsqueeze(1)

    # cumulative sum over time
    GT_power_cumsum = torch.cumsum(GT_power_norm, dim=1)
    pred_power_cumsum = torch.cumsum(pred_power_norm, dim=1)

    # earth mover distance
    emd = torch.norm((pred_power_cumsum - GT_power_cumsum), p=1, dim=1)

    # weighted EMD
    power_weighted_emd = torch.sum(emd * GT_power_sum) / torch.sum(GT_power_sum)

    return power_weighted_emd

def L2T(pred_traj, GT_traj, context_frames):
    # pred_traj: (B, T, 4)
    # GT_traj: (B, T, 4)

    pred_traj = pred_traj[:, context_frames:]
    GT_traj   = GT_traj[:, context_frames:]

    xz = torch.norm(pred_traj[..., 0:2] - GT_traj[..., 0:2], dim=-1) # (B, T)
    fwd = torch.norm(pred_traj[..., 2:4] - GT_traj[..., 2:4], dim=-1) # (B, T)
    return xz + fwd