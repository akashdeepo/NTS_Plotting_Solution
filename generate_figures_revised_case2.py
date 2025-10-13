"""
Generate Figures per Dr. Rachev's REVISED Manuscript (October 13, 2025)
Updated Case 2 implementation with damped cosine transform method
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings
from math import gamma
warnings.filterwarnings('ignore')

# Add temStaPy to path
sys.path.insert(0, r'c:\Users\Akash\OneDrive\Desktop\PWFs for NTS\lib\temStaPy_v0.5')
from temStaPy.distNTS import qnts, pnts, dnts

# Create output directory
OUTPUT_DIR = "manuscript_figures_revised"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("="*80)
print("GENERATING FIGURES PER DR. RACHEV'S REVISED MANUSCRIPT (Oct 13, 2025)")
print("="*80)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cgmy_to_nts(C, G, M, Y, mu=0, sigma=1):
    """Convert CGMY parameters to NTS parameters for temStaPy"""
    alpha = Y
    theta = C * (G**Y + M**Y) / Y
    beta_raw = (M**Y - G**Y) / (M**Y + G**Y)
    gamma = sigma
    beta = beta_raw * sigma
    return [alpha, theta, beta, gamma, mu]

def psi_TS(s, C, G, M, Y):
    """Characteristic exponent of the tempered-stable subordinator"""
    return C * gamma(-Y) * ((M - s)**Y - M**Y + (G + s)**Y - G**Y)

def phi_nts(u, mu, sigma, C, G, M, Y):
    """NTS characteristic function"""
    s = 1j*mu*u - 0.5*(sigma**2)*u**2
    return np.exp(psi_TS(s, C, G, M, Y))

def nts_pdf_damped_cosine(x, mu, sigma, C, G, M, Y, U=80.0, N=60000, damp=0.0):
    """
    Damped cosine inversion for pdf f_X(x)
    Per Dr. Rachev's specification for Case 2
    """
    u = np.linspace(1e-12, U, N)
    phi = phi_nts(u, mu, sigma, C, G, M, Y) * np.exp(-damp*u)
    integrand = np.real(phi * np.cos(np.outer(x, u)))
    du = (U - 1e-12) / (N - 1)
    fx = (1/np.pi) * integrand.sum(axis=1) * du
    return fx

def compute_variance(x, fx):
    """Compute variance from pdf"""
    dx = x[1] - x[0]
    fx_norm = fx / (fx.sum() * dx)
    mean = np.sum(x * fx_norm * dx)
    var = np.sum(x**2 * fx_norm * dx) - mean**2
    return var, mean

# ============================================================================
# COMMON PARAMETERS (Fixed across all cases per manuscript)
# ============================================================================
mu = 0.0
C = 0.6
Y = 1.2

# ============================================================================
# CASE 1: SCALE/VOLATILITY CHANNEL (Keep existing implementation)
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

# ----------------------------------------
# Figure 2(d): PWF for Case 1
# ----------------------------------------
print("Generating Figure 2(d): PWF for Case 1...")

u_grid = np.linspace(0.001, 0.999, 999)
q_prior = qnts(u_grid, nts_bench)
w_fear = pnts(q_prior, nts_fear)
w_greed = pnts(q_prior, nts_greed)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='45° line', alpha=0.7)
ax.plot(u_grid, w_fear, 'b-', linewidth=2, label='Fearful (σ = 1.4)')
ax.plot(u_grid, w_greed, 'r-', linewidth=2, label='Greedy (σ = 0.7)')

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

# ----------------------------------------
# Figure 2(e): Symmetric Greed-Fear Index
# ----------------------------------------
print("Generating Figure 2(e): Symmetric greed-fear index...")

mask = (u_grid > 0.01) & (u_grid < 0.99)
p = u_grid[mask]
gamma_fear = w_fear[mask] / (p * (1 - p))
gamma_greed = w_greed[mask] / (p * (1 - p))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(p, gamma_fear, 'b-', linewidth=2, label='Fearful (σ = 1.4)')
ax.plot(p, gamma_greed, 'r-', linewidth=2, label='Greedy (σ = 0.7)')

ax.set_xlabel('p', fontsize=12)
ax.set_ylabel('Γ(p)', fontsize=12)
ax.set_title('Figure 2(e): Γ(p) = w(p)/[p(1-p)] for Case 1', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2e_Case1_GreedFearIndex')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

# ----------------------------------------
# Figure 2(f): PDFs for Case 1
# ----------------------------------------
print("Generating Figure 2(f): PDFs for Case 1...")

x_grid = np.linspace(-6, 6, 1201)
pdf_bench = dnts(x_grid, nts_bench)
pdf_fear = dnts(x_grid, nts_fear)
pdf_greed = dnts(x_grid, nts_greed)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_grid, pdf_bench, 'k-', linewidth=2, label='Benchmark (σ = 1.0)', alpha=0.8)
ax.plot(x_grid, pdf_fear, 'b-', linewidth=2, label='Fearful (σ = 1.4)', alpha=0.8)
ax.plot(x_grid, pdf_greed, 'r-', linewidth=2, label='Greedy (σ = 0.7)', alpha=0.8)

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
# CASE 2: SKEW/ASYMMETRY CHANNEL (REVISED PER OCT 13 SPECIFICATIONS)
# ============================================================================
print("\n" + "-"*80)
print("CASE 2: Skew/Asymmetry Channel (REVISED)")
print("-"*80)
print("Using damped cosine transform method per Dr. Rachev's specifications")

# Revised Case 2 Parameters (from Oct 13 email)
G0, M0, sigma0 = 5.0, 5.0, 1.0        # benchmark (symmetric)
Gg, Mg, sigmag = 9.0, 3.0, 0.98       # greedy (right-skew)
Gf, Mf, sigmaf = 3.0, 9.0, 1.02       # fearful (left-skew)

print(f"\nBenchmark: G={G0}, M={M0}, sigma={sigma0}")
print(f"Greedy (right-skew): G={Gg}, M={Mg}, sigma={sigmag}")
print(f"Fearful (left-skew): G={Gf}, M={Mf}, sigma={sigmaf}")

# Shared x-grid
x = np.linspace(-6.0, 6.0, 4001)

# ----------------------------------------
# Compute PDFs using damped cosine method
# ----------------------------------------
print("\nComputing PDFs using damped cosine transform...")
f0 = nts_pdf_damped_cosine(x, mu, sigma0, C, G0, M0, Y, U=80.0, N=60000)
fg = nts_pdf_damped_cosine(x, mu, sigmag, C, Gg, Mg, Y, U=80.0, N=60000)
ff = nts_pdf_damped_cosine(x, mu, sigmaf, C, Gf, Mf, Y, U=80.0, N=60000)

# Numerical normalization
dx = x[1] - x[0]
f0 /= f0.sum() * dx
fg /= fg.sum() * dx
ff /= ff.sum() * dx

# Verify variance matching
var0, mean0 = compute_variance(x, f0)
varg, meang = compute_variance(x, fg)
varf, meanf = compute_variance(x, ff)

print(f"\nVariance verification:")
print(f"  Benchmark: mean={mean0:.4f}, var={var0:.4f}")
print(f"  Greedy:    mean={meang:.4f}, var={varg:.4f} (diff: {100*(varg-var0)/var0:+.2f}%)")
print(f"  Fearful:   mean={meanf:.4f}, var={varf:.4f} (diff: {100*(varf-var0)/var0:+.2f}%)")

# ----------------------------------------
# Figure 2(g): PDFs for Case 2
# ----------------------------------------
print("\nGenerating Figure 2(g): PDFs for Case 2...")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, f0, 'k-', linewidth=2, label='Benchmark (G=M=5)', alpha=0.8)
ax.plot(x, fg, 'r-', linewidth=2, label='Greedy (G=9, M=3, right-skew)', alpha=0.8)
ax.plot(x, ff, 'b-', linewidth=2, label='Fearful (G=3, M=9, left-skew)', alpha=0.8)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel(r'$f_X(x)$', fontsize=12)
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

# ----------------------------------------
# Compute CDFs by numerical integration
# ----------------------------------------
print("Computing CDFs by numerical integration...")
F0 = np.clip(np.cumsum(f0) * dx, 0, 1)
Fg = np.clip(np.cumsum(fg) * dx, 0, 1)
Ff = np.clip(np.cumsum(ff) * dx, 0, 1)

# ----------------------------------------
# Figure 2(h): CDFs for Case 2
# ----------------------------------------
print("Generating Figure 2(h): CDFs for Case 2...")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, F0, 'k-', linewidth=2, label='Benchmark', alpha=0.8)
ax.plot(x, Fg, 'r-', linewidth=2, label='Greedy (right-skew)', alpha=0.8)
ax.plot(x, Ff, 'b-', linewidth=2, label='Fearful (left-skew)', alpha=0.8)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel(r'$F_X(x)$', fontsize=12)
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

# ----------------------------------------
# Compute PWFs via quantile mapping
# ----------------------------------------
print("Computing PWFs via quantile mapping...")

u_plot = np.linspace(1e-3, 1-1e-3, 2001)  # exclude endpoints
Q0 = np.interp(u_plot, F0, x)              # prior quantile
w_greedy = np.interp(Q0, x, Fg)
w_fearful = np.interp(Q0, x, Ff)

# ----------------------------------------
# Figure 2(i): PWFs for Case 2
# ----------------------------------------
print("Generating Figure 2(i): PWFs for Case 2...")

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='45° line', alpha=0.7)
ax.plot(u_plot, w_greedy, 'r-', linewidth=2, label='Greedy (right-skew)')
ax.plot(u_plot, w_fearful, 'b-', linewidth=2, label='Fearful (left-skew)')

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

print("  Case 2 Complete: All 3 figures generated with REVISED parameters")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("="*80)

print("\nCase 1 (Scale/Volatility Channel):")
print("  - Figure 2(d): PWF")
print("  - Figure 2(e): Symmetric greed-fear index Gamma(p)")
print("  - Figure 2(f): PDFs")

print("\nCase 2 (Skew/Asymmetry Channel - REVISED):")
print("  - Figure 2(g): PDFs (damped cosine method)")
print("  - Figure 2(h): CDFs")
print("  - Figure 2(i): PWFs")

print(f"\nAll figures saved in: {os.path.abspath(OUTPUT_DIR)}")
print("\nParameters match Dr. Rachev's REVISED specifications (Oct 13, 2025):")
print("  Common: mu=0, C=0.6, Y=1.2")
print("  Case 1: G=M=5, sigma varied (1.0, 1.4, 0.7)")
print("  Case 2: G/M varied (9/3, 3/9), sigma adjusted (0.98, 1.02)")
print("  Method: Damped cosine transform with U=80, N=60000")
print("\nReady to send to Dr. Rachev!")
print("="*80)
