"""
Diagnostic analysis of information-theoretic indices
Check if they're actually producing meaningful values
"""

import numpy as np
import sys
sys.path.insert(0, r'lib\temStaPy_v0.5')
from temStaPy.distNTS import qnts, pnts

def cgmy_to_nts(C, G, M, Y, mu=0, sigma=1):
    """Convert CGMY parameters to NTS parameters"""
    alpha = Y
    theta = C * (G**Y + M**Y) / Y
    beta_raw = (M**Y - G**Y) / (M**Y + G**Y)
    gamma_param = sigma
    beta = beta_raw * sigma
    return [alpha, theta, beta, gamma_param, mu]

def compute_pwf(u_grid, prior_params, post_params):
    """Compute PWF: w(u) = F_post(Q_prior(u))"""
    q_prior = qnts(u_grid, prior_params)
    w = pnts(q_prior, post_params)
    return w

def safe_clip(arr, epsilon=1e-6):
    """Clip array to [epsilon, 1-epsilon]"""
    return np.clip(arr, epsilon, 1-epsilon)

def logit_shift_GFI(w, p, epsilon=1e-6):
    """
    Logit-shift greed-fear index with Fisher-information normalization
    G_FI(p) = [logit(w(p)) - logit(p)] / [p(1-p)]
    """
    w_clip = safe_clip(w, epsilon)
    p_clip = safe_clip(p, epsilon)

    logit_w = np.log(w_clip / (1 - w_clip))
    logit_p = np.log(p_clip / (1 - p_clip))

    G = logit_w - logit_p
    G_FI = G / (p_clip * (1 - p_clip))

    return G_FI

# Case 1 parameters (from the script)
mu, C, Y = 0.0, 0.6, 1.2
G1, M1 = 5.0, 5.0
sigma_bench = 1.0
sigma_fear = 1.4
sigma_greed = 0.7

nts_bench = cgmy_to_nts(C, G1, M1, Y, mu, sigma_bench)
nts_fear = cgmy_to_nts(C, G1, M1, Y, mu, sigma_fear)
nts_greed = cgmy_to_nts(C, G1, M1, Y, mu, sigma_greed)

print("="*80)
print("DIAGNOSTIC: Information-Theoretic Indices")
print("="*80)

# Compute PWFs on smaller grid for diagnostics
N = 100
p_grid = np.linspace(1.0/(N+1), N/(N+1), N)

print(f"\nComputing PWFs on grid with {N} points...")
w_bench = p_grid  # Benchmark is identity
w_fear = compute_pwf(p_grid, nts_bench, nts_fear)
w_greed = compute_pwf(p_grid, nts_bench, nts_greed)

# Check PWF values at key points
print("\nPWF values at key points:")
idx_25 = np.argmin(np.abs(p_grid - 0.25))
idx_50 = np.argmin(np.abs(p_grid - 0.50))
idx_75 = np.argmin(np.abs(p_grid - 0.75))

print(f"At p=0.25:")
print(f"  Benchmark: w={w_bench[idx_25]:.6f}")
print(f"  Fearful:   w={w_fear[idx_25]:.6f}, diff={w_fear[idx_25]-p_grid[idx_25]:+.6f}")
print(f"  Greedy:    w={w_greed[idx_25]:.6f}, diff={w_greed[idx_25]-p_grid[idx_25]:+.6f}")

print(f"\nAt p=0.50:")
print(f"  Benchmark: w={w_bench[idx_50]:.6f}")
print(f"  Fearful:   w={w_fear[idx_50]:.6f}, diff={w_fear[idx_50]-p_grid[idx_50]:+.6f}")
print(f"  Greedy:    w={w_greed[idx_50]:.6f}, diff={w_greed[idx_50]-p_grid[idx_50]:+.6f}")

print(f"\nAt p=0.75:")
print(f"  Benchmark: w={w_bench[idx_75]:.6f}")
print(f"  Fearful:   w={w_fear[idx_75]:.6f}, diff={w_fear[idx_75]-p_grid[idx_75]:+.6f}")
print(f"  Greedy:    w={w_greed[idx_75]:.6f}, diff={w_greed[idx_75]-p_grid[idx_75]:+.6f}")

# Compute G_FI at these points
G_FI_bench = logit_shift_GFI(w_bench, p_grid)
G_FI_fear = logit_shift_GFI(w_fear, p_grid)
G_FI_greed = logit_shift_GFI(w_greed, p_grid)

print("\n" + "-"*80)
print("G_FI index values at key points:")
print("-"*80)

print(f"At p=0.25:")
print(f"  Benchmark: G_FI={G_FI_bench[idx_25]:+.6f}")
print(f"  Fearful:   G_FI={G_FI_fear[idx_25]:+.6f}")
print(f"  Greedy:    G_FI={G_FI_greed[idx_25]:+.6f}")

print(f"\nAt p=0.50:")
print(f"  Benchmark: G_FI={G_FI_bench[idx_50]:+.6f}")
print(f"  Fearful:   G_FI={G_FI_fear[idx_50]:+.6f}")
print(f"  Greedy:    G_FI={G_FI_greed[idx_50]:+.6f}")

print(f"\nAt p=0.75:")
print(f"  Benchmark: G_FI={G_FI_bench[idx_75]:+.6f}")
print(f"  Fearful:   G_FI={G_FI_fear[idx_75]:+.6f}")
print(f"  Greedy:    G_FI={G_FI_greed[idx_75]:+.6f}")

# Check ranges
print("\n" + "-"*80)
print("G_FI index statistics:")
print("-"*80)

print(f"Benchmark: min={np.min(G_FI_bench):+.6f}, max={np.max(G_FI_bench):+.6f}, mean={np.mean(G_FI_bench):+.6f}")
print(f"Fearful:   min={np.min(G_FI_fear):+.6f}, max={np.max(G_FI_fear):+.6f}, mean={np.mean(G_FI_fear):+.6f}")
print(f"Greedy:    min={np.min(G_FI_greed):+.6f}, max={np.max(G_FI_greed):+.6f}, mean={np.mean(G_FI_greed):+.6f}")

# Integral scores
G_FI_score_bench = np.trapz(G_FI_bench, p_grid)
G_FI_score_fear = np.trapz(G_FI_fear, p_grid)
G_FI_score_greed = np.trapz(G_FI_greed, p_grid)

print(f"\nAggregate scores (integral):")
print(f"  Benchmark: {G_FI_score_bench:+.8f}")
print(f"  Fearful:   {G_FI_score_fear:+.8f}")
print(f"  Greedy:    {G_FI_score_greed:+.8f}")

print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)

max_diff_fear = np.max(np.abs(G_FI_fear - G_FI_bench))
max_diff_greed = np.max(np.abs(G_FI_greed - G_FI_bench))

print(f"Maximum deviation from benchmark:")
print(f"  Fearful: {max_diff_fear:.6f}")
print(f"  Greedy:  {max_diff_greed:.6f}")

if max_diff_fear < 0.01 and max_diff_greed < 0.01:
    print("\nWARNING: Very small deviations! The indices may not show much separation.")
    print("This suggests Case 1 (sigma variation) may not produce strong effects")
    print("in these information-theoretic metrics.")
else:
    print("\nGood: Indices show measurable separation from benchmark.")

print("="*80)
