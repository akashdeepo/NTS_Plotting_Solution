"""
Generate Figures with DR. RACHEV'S TRUE SPECIFICATIONS (Found Oct 13 Evening)
CORRECTED Case 2 parameters: sigma = 0.92 for BOTH greedy and fearful!
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add temStaPy to path
sys.path.insert(0, r'c:\Users\Akash\OneDrive\Desktop\PWFs for NTS\lib\temStaPy_v0.5')
from temStaPy.distNTS import qnts, pnts, dnts

# Create output directory
OUTPUT_DIR = "manuscript_figures_FINAL"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("="*80)
print("GENERATING FIGURES WITH DR. RACHEV'S TRUE SPECIFICATIONS")
print("CORRECTED: Case 2 uses sigma=0.92 for BOTH greedy and fearful!")
print("="*80)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cgmy_to_nts(C, G, M, Y, mu=0, sigma=1):
    """Convert CGMY parameters to NTS parameters for temStaPy"""
    alpha = Y
    theta = C * (G**Y + M**Y) / Y
    beta_raw = (M**Y - G**Y) / (M**Y + G**Y)
    gamma_param = sigma
    beta = beta_raw * sigma
    return [alpha, theta, beta, gamma_param, mu]

def compute_pwf(u_grid, prior_params, post_params):
    """Compute Probability Weighting Function: w(u) = F_post(Q_prior(u))"""
    q_prior = qnts(u_grid, prior_params)
    w = pnts(q_prior, post_params)
    return w

def compute_variance_from_nts(nts_params):
    """Compute variance from NTS distribution by sampling"""
    x = np.linspace(-10, 10, 10001)
    pdf = dnts(x, nts_params)
    dx = x[1] - x[0]
    pdf_norm = pdf / (pdf.sum() * dx)
    mean = np.sum(x * pdf_norm * dx)
    var = np.sum(x**2 * pdf_norm * dx) - mean**2
    return var, mean

# ============================================================================
# COMMON PARAMETERS (Fixed across all cases per manuscript)
# ============================================================================
mu = 0.0
C = 0.6
Y = 1.2

# ============================================================================
# CASE 1: SCALE/VOLATILITY CHANNEL (unchanged)
# ============================================================================
print("\n" + "-"*80)
print("CASE 1: Scale/Volatility Channel")
print("-"*80)

G1 = 5.0
M1 = 5.0
sigma_bench = 1.0
sigma_fear = 1.4
sigma_greed = 0.7

nts_bench = cgmy_to_nts(C, G1, M1, Y, mu, sigma_bench)
nts_fear = cgmy_to_nts(C, G1, M1, Y, mu, sigma_fear)
nts_greed = cgmy_to_nts(C, G1, M1, Y, mu, sigma_greed)

# Figure 2(d): PWF for Case 1
print("Generating Figure 2(d): PWF for Case 1...")

u_grid = np.linspace(0.001, 0.999, 999)
w_fear = compute_pwf(u_grid, nts_bench, nts_fear)
w_greed = compute_pwf(u_grid, nts_bench, nts_greed)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='45 degree line', alpha=0.7)
ax.plot(u_grid, w_fear, 'b-', linewidth=2, label='Fearful (sigma = 1.4)')
ax.plot(u_grid, w_greed, 'r-', linewidth=2, label='Greedy (sigma = 0.7)')

ax.set_xlabel('u', fontsize=12)
ax.set_ylabel('w(u)', fontsize=12)
ax.set_title('Figure 2(d): PWF for Case 1 - Scale/Volatility Channel', fontsize=13)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2d_Case1_PWF')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

# Figure 2(e): Symmetric Greed-Fear Index
print("Generating Figure 2(e): Symmetric greed-fear index...")

mask = (u_grid > 0.01) & (u_grid < 0.99)
p = u_grid[mask]
gamma_fear = w_fear[mask] / (p * (1 - p))
gamma_greed = w_greed[mask] / (p * (1 - p))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(p, gamma_fear, 'b-', linewidth=2, label='Fearful (sigma = 1.4)')
ax.plot(p, gamma_greed, 'r-', linewidth=2, label='Greedy (sigma = 0.7)')

ax.set_xlabel('p', fontsize=12)
ax.set_ylabel('Gamma(p)', fontsize=12)
ax.set_title('Figure 2(e): Gamma(p) = w(p)/[p(1-p)] for Case 1', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2e_Case1_GreedFearIndex')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

# Figure 2(f): PDFs for Case 1
print("Generating Figure 2(f): PDFs for Case 1...")

x_grid = np.linspace(-6, 6, 1201)
pdf_bench = dnts(x_grid, nts_bench)
pdf_fear = dnts(x_grid, nts_fear)
pdf_greed = dnts(x_grid, nts_greed)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_grid, pdf_bench, 'k-', linewidth=2, label='Benchmark (sigma = 1.0)', alpha=0.8)
ax.plot(x_grid, pdf_fear, 'b-', linewidth=2, label='Fearful (sigma = 1.4)', alpha=0.8)
ax.plot(x_grid, pdf_greed, 'r-', linewidth=2, label='Greedy (sigma = 0.7)', alpha=0.8)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f_X(x)', fontsize=12)
ax.set_title('Figure 2(f): PDFs for Case 1 (benchmark, greedy, fearful)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-6, 6)
ax.set_ylim(0, None)
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2f_Case1_PDFs')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

print("  Case 1 Complete: All 3 figures generated")

# ============================================================================
# CASE 2: SKEW/ASYMMETRY CHANNEL - TRUE CORRECTED PARAMETERS!
# ============================================================================
print("\n" + "-"*80)
print("CASE 2: Skew/Asymmetry Channel - TRUE CORRECTED PARAMETERS")
print("-"*80)
print("KEY CHANGE: sigma = 0.92 for BOTH greedy AND fearful (not 0.98 and 1.02!)")

# TRUE CORRECTED Case 2 Parameters from Dr. Rachev
G0, M0, sigma0 = 5.0, 5.0, 1.0        # benchmark (symmetric)
Gg, Mg, sigmag = 9.0, 3.0, 0.92       # greedy (right-skew) - CORRECTED!
Gf, Mf, sigmaf = 3.0, 9.0, 0.92       # fearful (left-skew) - CORRECTED!

print(f"\nBenchmark: G={G0}, M={M0}, sigma={sigma0}")
print(f"Greedy (right-skew): G={Gg}, M={Mg}, sigma={sigmag} [CORRECTED]")
print(f"Fearful (left-skew): G={Gf}, M={Mf}, sigma={sigmaf} [CORRECTED]")

# Convert to NTS parameters
nts_bench_c2 = cgmy_to_nts(C, G0, M0, Y, mu, sigma0)
nts_greed_c2 = cgmy_to_nts(C, Gg, Mg, Y, mu, sigmag)
nts_fear_c2 = cgmy_to_nts(C, Gf, Mf, Y, mu, sigmaf)

# Verify variance matching
print("\nComputing variance for verification...")
var0, mean0 = compute_variance_from_nts(nts_bench_c2)
varg, meang = compute_variance_from_nts(nts_greed_c2)
varf, meanf = compute_variance_from_nts(nts_fear_c2)

print(f"Variance verification:")
print(f"  Benchmark: mean={mean0:.4f}, var={var0:.4f}")
print(f"  Greedy:    mean={meang:.4f}, var={varg:.4f} (diff: {100*(varg-var0)/var0:+.2f}%)")
print(f"  Fearful:   mean={meanf:.4f}, var={varf:.4f} (diff: {100*(varf-var0)/var0:+.2f}%)")

# Quick asymmetry check
x_test = np.array([-1.5, 1.5])
pdf_g_test = dnts(x_test, nts_greed_c2)
pdf_f_test = dnts(x_test, nts_fear_c2)
pdf_b_test = dnts(x_test, nts_bench_c2)
print(f"\nAsymmetry check (PDF ratio at x=+1.5 / x=-1.5):")
print(f"  Benchmark: {pdf_b_test[1]/pdf_b_test[0]:.4f}")
print(f"  Greedy:    {pdf_g_test[1]/pdf_g_test[0]:.4f} (should be > 1)")
print(f"  Fearful:   {pdf_f_test[1]/pdf_f_test[0]:.4f} (should be < 1)")

# Figure 2(g): PDFs for Case 2
print("\nGenerating Figure 2(g): PDFs for Case 2...")

x_grid_c2 = np.linspace(-6, 6, 1201)
pdf_bench_c2 = dnts(x_grid_c2, nts_bench_c2)
pdf_greed_c2 = dnts(x_grid_c2, nts_greed_c2)
pdf_fear_c2 = dnts(x_grid_c2, nts_fear_c2)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_grid_c2, pdf_bench_c2, 'k-', linewidth=2, label='Benchmark (G=M=5)', alpha=0.8)
ax.plot(x_grid_c2, pdf_greed_c2, 'r-', linewidth=2, label='Greedy (G=9, M=3, right-skew)', alpha=0.8)
ax.plot(x_grid_c2, pdf_fear_c2, 'b-', linewidth=2, label='Fearful (G=3, M=9, left-skew)', alpha=0.8)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f_X(x)', fontsize=12)
ax.set_title('Figure 2(g): PDFs for Case 2 (benchmark, greedy, fearful)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-6, 6)
ax.set_ylim(0, None)
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2g_Case2_PDFs')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

# Figure 2(h): CDFs for Case 2
print("Generating Figure 2(h): CDFs for Case 2...")

cdf_bench_c2 = pnts(x_grid_c2, nts_bench_c2)
cdf_greed_c2 = pnts(x_grid_c2, nts_greed_c2)
cdf_fear_c2 = pnts(x_grid_c2, nts_fear_c2)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_grid_c2, cdf_bench_c2, 'k-', linewidth=2, label='Benchmark', alpha=0.8)
ax.plot(x_grid_c2, cdf_greed_c2, 'r-', linewidth=2, label='Greedy (right-skew)', alpha=0.8)
ax.plot(x_grid_c2, cdf_fear_c2, 'b-', linewidth=2, label='Fearful (left-skew)', alpha=0.8)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('F_X(x)', fontsize=12)
ax.set_title('Figure 2(h): CDFs for Case 2', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-6, 6)
ax.set_ylim(0, 1)
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2h_Case2_CDFs')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

# Figure 2(i): PWFs for Case 2
print("Generating Figure 2(i): PWFs for Case 2...")

w_greed_c2 = compute_pwf(u_grid, nts_bench_c2, nts_greed_c2)
w_fear_c2 = compute_pwf(u_grid, nts_bench_c2, nts_fear_c2)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='45 degree line', alpha=0.7)
ax.plot(u_grid, w_greed_c2, 'r-', linewidth=2, label='Greedy (right-skew)')
ax.plot(u_grid, w_fear_c2, 'b-', linewidth=2, label='Fearful (left-skew)')

ax.set_xlabel('u', fontsize=12)
ax.set_ylabel('w(u)', fontsize=12)
ax.set_title('Figure 2(i): PWFs w(u) for the skew/asymmetry channel', fontsize=13)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2i_Case2_PWFs')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

print("  Case 2 Complete: All 3 figures generated with CORRECTED parameters")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ALL FIGURES GENERATED SUCCESSFULLY WITH CORRECTED PARAMETERS!")
print("="*80)

print("\nCase 1 (Scale/Volatility Channel):")
print("  - Figure 2(d): PWF")
print("  - Figure 2(e): Symmetric greed-fear index Gamma(p)")
print("  - Figure 2(f): PDFs")

print("\nCase 2 (Skew/Asymmetry Channel - CORRECTED):")
print("  - Figure 2(g): PDFs")
print("  - Figure 2(h): CDFs")
print("  - Figure 2(i): PWFs")

print(f"\nAll figures saved in: {os.path.abspath(OUTPUT_DIR)}")
print("\nCORRECTED Parameters (Oct 13, 2025 - Evening Discovery):")
print("  Common: mu=0, C=0.6, Y=1.2")
print("  Case 1: G=M=5, sigma varied (1.0, 1.4, 0.7)")
print("  Case 2: G/M varied (9/3, 3/9), sigma=0.92 for BOTH")
print("  Method: temStaPy library (robust FFT-based)")
print("\nReady to send to Dr. Rachev!")
print("="*80)
