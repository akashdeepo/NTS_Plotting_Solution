"""
Check the Signed Jensen-Shannon index behavior
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

S_JS_bench = signed_jensen_shannon(w_bench, p_grid)
S_JS_fear = signed_jensen_shannon(w_fear, p_grid)
S_JS_greed = signed_jensen_shannon(w_greed, p_grid)

print("="*80)
print("SIGNED JENSEN-SHANNON INDEX ANALYSIS")
print("="*80)

idx_25 = np.argmin(np.abs(p_grid - 0.25))
idx_50 = np.argmin(np.abs(p_grid - 0.50))
idx_75 = np.argmin(np.abs(p_grid - 0.75))

print("\nS_JS values at key points:")
print(f"At p=0.25:")
print(f"  Benchmark: S_JS={S_JS_bench[idx_25]:+.6f}")
print(f"  Fearful:   S_JS={S_JS_fear[idx_25]:+.6f}")
print(f"  Greedy:    S_JS={S_JS_greed[idx_25]:+.6f}")

print(f"\nAt p=0.50:")
print(f"  Benchmark: S_JS={S_JS_bench[idx_50]:+.6f}")
print(f"  Fearful:   S_JS={S_JS_fear[idx_50]:+.6f}")
print(f"  Greedy:    S_JS={S_JS_greed[idx_50]:+.6f}")

print(f"\nAt p=0.75:")
print(f"  Benchmark: S_JS={S_JS_bench[idx_75]:+.6f}")
print(f"  Fearful:   S_JS={S_JS_fear[idx_75]:+.6f}")
print(f"  Greedy:    S_JS={S_JS_greed[idx_75]:+.6f}")

print("\n" + "-"*80)
print("Statistics:")
print("-"*80)
print(f"Benchmark: min={np.min(S_JS_bench):+.6f}, max={np.max(S_JS_bench):+.6f}")
print(f"Fearful:   min={np.min(S_JS_fear):+.6f}, max={np.max(S_JS_fear):+.6f}")
print(f"Greedy:    min={np.min(S_JS_greed):+.6f}, max={np.max(S_JS_greed):+.6f}")

# Check if bounded by ln(2) ≈ 0.693
ln2 = np.log(2)
print(f"\nTheoretical bound: +/- sqrt(2*ln(2)) = +/- {np.sqrt(2*ln2):.6f}")
print(f"Fearful max magnitude: {max(abs(np.min(S_JS_fear)), abs(np.max(S_JS_fear))):.6f}")
print(f"Greedy max magnitude: {max(abs(np.min(S_JS_greed)), abs(np.max(S_JS_greed))):.6f}")

print("\n" + "-"*80)
print("EXPECTED BEHAVIOR CHECK:")
print("-"*80)

# For fearful: w > p at low p, w < p at high p -> S_JS positive at low p, negative at high p
# For greedy: w < p at low p, w > p at high p -> S_JS negative at low p, positive at high p

fearful_correct = S_JS_fear[idx_25] > 0 and S_JS_fear[idx_75] < 0
greedy_correct = S_JS_greed[idx_25] < 0 and S_JS_greed[idx_75] > 0

print(f"Fearful pattern (+ at p<0.5, - at p>0.5): {fearful_correct}")
print(f"Greedy pattern  (- at p<0.5, + at p>0.5): {greedy_correct}")

if fearful_correct and greedy_correct:
    print("\nCHECK PASSED: S_JS shows expected mirror-symmetric patterns!")
else:
    print("\nWARNING: S_JS patterns don't match expectations")

print("="*80)
