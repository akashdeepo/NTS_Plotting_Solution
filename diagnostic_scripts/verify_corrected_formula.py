"""
Verify the corrected G_FI formula matches Dr. Rachev's specification
"""

import numpy as np
import sys
sys.path.insert(0, r'lib\temStaPy_v0.5')
from temStaPy.distNTS import qnts, pnts

def cgmy_to_nts(C, G, M, Y, mu=0, sigma=1):
    alpha = Y
    theta = C * (G**Y + M**Y) / Y
    beta_raw = (M**Y - G**Y) / (M**Y + G**Y)
    gamma_param = sigma
    beta = beta_raw * sigma
    return [alpha, theta, beta, gamma_param, mu]

def compute_pwf(u_grid, prior_params, post_params):
    q_prior = qnts(u_grid, prior_params)
    w = pnts(q_prior, post_params)
    return w

def safe_clip(arr, epsilon=1e-6):
    return np.clip(arr, epsilon, 1-epsilon)

def logit_shift_GFI_CORRECTED(w, p, epsilon=1e-6):
    """CORRECTED: Division by sqrt[p(1-p)]"""
    w_clip = safe_clip(w, epsilon)
    p_clip = safe_clip(p, epsilon)
    logit_w = np.log(w_clip / (1 - w_clip))
    logit_p = np.log(p_clip / (1 - p_clip))
    G = logit_w - logit_p
    G_FI = G / np.sqrt(p_clip * (1 - p_clip))  # CORRECTED: added sqrt
    return G_FI

def logit_shift_GFI_OLD(w, p, epsilon=1e-6):
    """OLD (WRONG): Division by p(1-p)"""
    w_clip = safe_clip(w, epsilon)
    p_clip = safe_clip(p, epsilon)
    logit_w = np.log(w_clip / (1 - w_clip))
    logit_p = np.log(p_clip / (1 - p_clip))
    G = logit_w - logit_p
    G_FI = G / (p_clip * (1 - p_clip))  # OLD: no sqrt
    return G_FI

# Case 1 parameters
mu, C, Y = 0.0, 0.6, 1.2
G1, M1 = 5.0, 5.0
sigma_bench = 1.0
sigma_fear = 1.4
sigma_greed = 0.7

nts_bench = cgmy_to_nts(C, G1, M1, Y, mu, sigma_bench)
nts_fear = cgmy_to_nts(C, G1, M1, Y, mu, sigma_fear)
nts_greed = cgmy_to_nts(C, G1, M1, Y, mu, sigma_greed)

N = 100
p_grid = np.linspace(1.0/(N+1), N/(N+1), N)

w_bench = p_grid
w_fear = compute_pwf(p_grid, nts_bench, nts_fear)
w_greed = compute_pwf(p_grid, nts_bench, nts_greed)

# Compute both versions
G_FI_corrected = logit_shift_GFI_CORRECTED(w_greed, p_grid)
G_FI_old = logit_shift_GFI_OLD(w_greed, p_grid)

print("="*80)
print("VERIFICATION: G_FI Formula Correction")
print("="*80)

idx_25 = np.argmin(np.abs(p_grid - 0.25))

print(f"\nAt p=0.25, w(p)={w_greed[idx_25]:.6f}:")
print(f"  OLD formula (÷ p(1-p)):      G_FI = {G_FI_old[idx_25]:+.6f}")
print(f"  CORRECTED (÷ sqrt[p(1-p)]): G_FI = {G_FI_corrected[idx_25]:+.6f}")
print(f"  Ratio (corrected/old):       {G_FI_corrected[idx_25]/G_FI_old[idx_25]:.6f}")

print(f"\nExpected ratio: sqrt(1 / [p(1-p)]) = sqrt(1 / [0.25*0.75])")
print(f"              = sqrt(1 / 0.1875) = sqrt(5.333) = {np.sqrt(1/(0.25*0.75)):.6f}")

print("\n" + "-"*80)
print("Range comparison:")
print("-"*80)
print(f"OLD formula:      min={np.min(G_FI_old):+.2f}, max={np.max(G_FI_old):+.2f}")
print(f"CORRECTED formula: min={np.min(G_FI_corrected):+.2f}, max={np.max(G_FI_corrected):+.2f}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("The corrected formula now divides by sqrt[p(1-p)] as per Dr. Rachev's")
print("Equation (2) in the blue text. This makes the index scale like the")
print("square root of Fisher information instead of Fisher information itself.")
print("="*80)
