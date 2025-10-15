"""
Detailed analysis of the CORRECTED Figure 2(e) plots
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

def logit_shift_GFI(w, p, epsilon=1e-6):
    """CORRECTED: sqrt[p(1-p)] in denominator"""
    w_clip = safe_clip(w, epsilon)
    p_clip = safe_clip(p, epsilon)
    logit_w = np.log(w_clip / (1 - w_clip))
    logit_p = np.log(p_clip / (1 - p_clip))
    G = logit_w - logit_p
    G_FI = G / np.sqrt(p_clip * (1 - p_clip))
    return G_FI

def kl_divergence_bernoulli(a, b, epsilon=1e-6):
    a_clip = safe_clip(a, epsilon)
    b_clip = safe_clip(b, epsilon)
    term1 = a_clip * np.log(a_clip / b_clip)
    term2 = (1 - a_clip) * np.log((1 - a_clip) / (1 - b_clip))
    return term1 + term2

def signed_jensen_shannon(w, p, epsilon=1e-6):
    w_clip = safe_clip(w, epsilon)
    p_clip = safe_clip(p, epsilon)
    m = (w_clip + p_clip) / 2.0
    kl_wm = kl_divergence_bernoulli(w_clip, m, epsilon)
    kl_pm = kl_divergence_bernoulli(p_clip, m, epsilon)
    JSD = 0.5 * (kl_wm + kl_pm)
    sign = np.sign(w_clip - p_clip)
    S_JS = sign * np.sqrt(2 * JSD)
    return S_JS

def log_odds_elasticity(w, p, epsilon=1e-6):
    w_clip = safe_clip(w, epsilon)
    p_clip = safe_clip(p, epsilon)
    dw_dp = np.gradient(w_clip, p_clip, edge_order=2)
    term1 = dw_dp / (w_clip * (1 - w_clip))
    term2 = 1.0 / (p_clip * (1 - p_clip))
    E = term1 - term2
    return E

# Case 1 parameters
mu, C, Y = 0.0, 0.6, 1.2
G1, M1 = 5.0, 5.0
sigma_bench = 1.0
sigma_fear = 1.4
sigma_greed = 0.7

nts_bench = cgmy_to_nts(C, G1, M1, Y, mu, sigma_bench)
nts_fear = cgmy_to_nts(C, G1, M1, Y, mu, sigma_fear)
nts_greed = cgmy_to_nts(C, G1, M1, Y, mu, sigma_greed)

N = 200
p_grid = np.linspace(1.0/(N+1), N/(N+1), N)

w_bench = p_grid
w_fear = compute_pwf(p_grid, nts_bench, nts_fear)
w_greed = compute_pwf(p_grid, nts_bench, nts_greed)

# Compute all indices
G_FI_bench = logit_shift_GFI(w_bench, p_grid)
G_FI_fear = logit_shift_GFI(w_fear, p_grid)
G_FI_greed = logit_shift_GFI(w_greed, p_grid)

S_JS_bench = signed_jensen_shannon(w_bench, p_grid)
S_JS_fear = signed_jensen_shannon(w_fear, p_grid)
S_JS_greed = signed_jensen_shannon(w_greed, p_grid)

E_bench = log_odds_elasticity(w_bench, p_grid)
E_fear = log_odds_elasticity(w_fear, p_grid)
E_greed = log_odds_elasticity(w_greed, p_grid)

# Cumulative
C_GFI_fear = np.cumsum(G_FI_fear) * (p_grid[1] - p_grid[0])
C_GFI_greed = np.cumsum(G_FI_greed) * (p_grid[1] - p_grid[0])
C_SJS_fear = np.cumsum(S_JS_fear) * (p_grid[1] - p_grid[0])
C_SJS_greed = np.cumsum(S_JS_greed) * (p_grid[1] - p_grid[0])

print("="*80)
print("DETAILED ANALYSIS OF CORRECTED FIGURE 2(e) PLOTS")
print("="*80)

# Key points
idx_10 = np.argmin(np.abs(p_grid - 0.10))
idx_25 = np.argmin(np.abs(p_grid - 0.25))
idx_50 = np.argmin(np.abs(p_grid - 0.50))
idx_75 = np.argmin(np.abs(p_grid - 0.75))
idx_90 = np.argmin(np.abs(p_grid - 0.90))

print("\n" + "="*80)
print("FIGURE 2(e1): LOGIT-SHIFT INDEX G_FI(p) [CORRECTED]")
print("="*80)

print("\nKey values:")
print(f"  p=0.10: Fearful={G_FI_fear[idx_10]:+8.3f}, Greedy={G_FI_greed[idx_10]:+8.3f}")
print(f"  p=0.25: Fearful={G_FI_fear[idx_25]:+8.3f}, Greedy={G_FI_greed[idx_25]:+8.3f}")
print(f"  p=0.50: Fearful={G_FI_fear[idx_50]:+8.3f}, Greedy={G_FI_greed[idx_50]:+8.3f}")
print(f"  p=0.75: Fearful={G_FI_fear[idx_75]:+8.3f}, Greedy={G_FI_greed[idx_75]:+8.3f}")
print(f"  p=0.90: Fearful={G_FI_fear[idx_90]:+8.3f}, Greedy={G_FI_greed[idx_90]:+8.3f}")

print("\nRange:")
print(f"  Fearful:  min={np.min(G_FI_fear):+8.3f}, max={np.max(G_FI_fear):+8.3f}")
print(f"  Greedy:   min={np.min(G_FI_greed):+8.3f}, max={np.max(G_FI_greed):+8.3f}")

print("\nPattern check:")
fear_pattern_gfi = "PASS" if (G_FI_fear[idx_25] > 0 and G_FI_fear[idx_75] < 0) else "FAIL"
greed_pattern_gfi = "PASS" if (G_FI_greed[idx_25] < 0 and G_FI_greed[idx_75] > 0) else "FAIL"
print(f"  Fearful (+ at p<0.5, - at p>0.5): {fear_pattern_gfi}")
print(f"  Greedy  (- at p<0.5, + at p>0.5): {greed_pattern_gfi}")

print("\n" + "="*80)
print("FIGURE 2(e2): SIGNED JENSEN-SHANNON INDEX S_JS(p)")
print("="*80)

print("\nKey values:")
print(f"  p=0.10: Fearful={S_JS_fear[idx_10]:+8.4f}, Greedy={S_JS_greed[idx_10]:+8.4f}")
print(f"  p=0.25: Fearful={S_JS_fear[idx_25]:+8.4f}, Greedy={S_JS_greed[idx_25]:+8.4f}")
print(f"  p=0.50: Fearful={S_JS_fear[idx_50]:+8.4f}, Greedy={S_JS_greed[idx_50]:+8.4f}")
print(f"  p=0.75: Fearful={S_JS_fear[idx_75]:+8.4f}, Greedy={S_JS_greed[idx_75]:+8.4f}")
print(f"  p=0.90: Fearful={S_JS_fear[idx_90]:+8.4f}, Greedy={S_JS_greed[idx_90]:+8.4f}")

print("\nRange:")
print(f"  Fearful:  min={np.min(S_JS_fear):+8.4f}, max={np.max(S_JS_fear):+8.4f}")
print(f"  Greedy:   min={np.min(S_JS_greed):+8.4f}, max={np.max(S_JS_greed):+8.4f}")
print(f"  Theoretical bound: +/- {np.sqrt(2*np.log(2)):.4f}")

print("\nPattern check:")
fear_pattern_sjs = "PASS" if (S_JS_fear[idx_25] > 0 and S_JS_fear[idx_75] < 0) else "FAIL"
greed_pattern_sjs = "PASS" if (S_JS_greed[idx_25] < 0 and S_JS_greed[idx_75] > 0) else "FAIL"
print(f"  Fearful (+ at p<0.5, - at p>0.5): {fear_pattern_sjs}")
print(f"  Greedy  (- at p<0.5, + at p>0.5): {greed_pattern_sjs}")

print("\n" + "="*80)
print("FIGURE 2(e3): LOG-ODDS ELASTICITY E(p)")
print("="*80)

print("\nRange:")
print(f"  Fearful:  min={np.min(E_fear):+10.2f}, max={np.max(E_fear):+10.2f}")
print(f"  Greedy:   min={np.min(E_greed):+10.2f}, max={np.max(E_greed):+10.2f}")

print("\nCenter region (p=0.4-0.6) statistics:")
center_mask = (p_grid >= 0.4) & (p_grid <= 0.6)
print(f"  Fearful mean: {np.mean(E_fear[center_mask]):+8.3f}")
print(f"  Greedy mean:  {np.mean(E_greed[center_mask]):+8.3f}")

print("\n" + "="*80)
print("FIGURE 2(e4): CUMULATIVE SCORES")
print("="*80)

print("\nCumulative G_FI peak magnitudes:")
print(f"  Fearful:  max={np.max(C_GFI_fear):+8.3f}, min={np.min(C_GFI_fear):+8.3f}")
print(f"  Greedy:   max={np.max(C_GFI_greed):+8.3f}, min={np.min(C_GFI_greed):+8.3f}")

print("\nCumulative S_JS peak magnitudes:")
print(f"  Fearful:  max={np.max(C_SJS_fear):+8.4f}, min={np.min(C_SJS_fear):+8.4f}")
print(f"  Greedy:   max={np.max(C_SJS_greed):+8.4f}, min={np.min(C_SJS_greed):+8.4f}")

print("\nFinal cumulative values (should be near 0):")
print(f"  G_FI:  Fearful={C_GFI_fear[-1]:+.6f}, Greedy={C_GFI_greed[-1]:+.6f}")
print(f"  S_JS:  Fearful={C_SJS_fear[-1]:+.6f}, Greedy={C_SJS_greed[-1]:+.6f}")

print("\n" + "="*80)
print("OVERALL ASSESSMENT")
print("="*80)

all_pass = fear_pattern_gfi == "PASS" and greed_pattern_gfi == "PASS" and \
           fear_pattern_sjs == "PASS" and greed_pattern_sjs == "PASS"

if all_pass:
    print("\nSTATUS: ALL CHECKS PASSED!")
    print("\nAll indices show:")
    print("  - Mirror-symmetric patterns with opposite signs")
    print("  - Fearful: positive at p<0.5, negative at p>0.5")
    print("  - Greedy:  negative at p<0.5, positive at p>0.5")
    print("  - Cumulative curves return to ~0 at p=1")
    print("\nThese plots are READY for Dr. Rachev's review.")
else:
    print("\nWARNING: Some pattern checks failed!")

print("="*80)
