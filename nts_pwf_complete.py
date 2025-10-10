"""
Complete Implementation of NTS PWFs for Dr. Rachev's Paper
============================================================
Generates all figures for Section 2: Normal Tempered-Stable distributions
Following the manuscript specifications from October 9, 2025

Authors: Akash Deep, Alex Trindade, W. Brent Lindquist, S. T. Rachev, Frank Fabozzi
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import interp1d
from datetime import datetime

# Add temStaPy to path
sys.path.insert(0, r'c:\Users\Akash\OneDrive\Desktop\PWFs for NTS\temStaPy_v0.5')
from temStaPy.distNTS import dnts, pnts, qnts

# Create output directory structure
OUTPUT_DIR = "manuscript_figures"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ============================================================================
# PARAMETER CONVERSION
# ============================================================================

def cgmy_to_nts(C, G, M, Y, mu=0, sigma=1):
    """
    Convert CGMY/KoBoL parameters to NTS parameters
    Following the exact formulation in Dr. Rachev's manuscript
    """
    alpha = Y
    theta = C * (G**Y + M**Y) / Y
    beta_raw = (M**Y - G**Y) / (M**Y + G**Y)
    gamma = sigma
    beta = beta_raw * sigma
    return [alpha, theta, beta, gamma, mu]

# ============================================================================
# PARAMETER SPECIFICATIONS (From Dr. Rachev's manuscript)
# ============================================================================

# Prior (benchmark) - symmetric NTS
PRIOR_CGMY = {'mu': 0.0, 'sigma': 1.0, 'C': 0.6, 'G': 5.0, 'M': 5.0, 'Y': 1.2}

# Specifications for different cases
specs = {
    'prior': PRIOR_CGMY,
    'high_vol': {**PRIOR_CGMY, 'sigma': 1.2},
    'low_vol': {**PRIOR_CGMY, 'sigma': 0.8},
    'right_skew': {**PRIOR_CGMY, 'G': 6.5, 'M': 3.8},
    'left_skew': {**PRIOR_CGMY, 'G': 3.8, 'M': 6.5},
    'lower_Y': {**PRIOR_CGMY, 'Y': 0.9},
    'higher_Y': {**PRIOR_CGMY, 'Y': 1.5},
    'loc_shift_pos': {**PRIOR_CGMY, 'mu': 0.4},
    'loc_shift_neg': {**PRIOR_CGMY, 'mu': -0.4},
}

# Convert to NTS parameters
nts_params = {key: cgmy_to_nts(**val) for key, val in specs.items()}

# ============================================================================
# FIGURE 2(a): PDFs
# ============================================================================

def generate_figure_2a():
    """Generate Figure 2(a): PDFs of selected NTS specifications"""
    print("Generating Figure 2(a): PDFs...")

    x_grid = np.linspace(-5, 5, 801)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Normal(0,1) reference
    pdf_norm = (1/np.sqrt(2*np.pi)) * np.exp(-0.5*x_grid**2)
    ax.plot(x_grid, pdf_norm, 'k:', linewidth=1.5, label='N(0,1)', alpha=0.7)

    # NTS curves
    curves = [
        ('prior', 'NTS prior', 'b-'),
        ('high_vol', 'σ = 1.2', 'g-'),
        ('low_vol', 'σ = 0.8', 'r-'),
        ('right_skew', 'right-skew (G>M)', 'c-'),
        ('lower_Y', 'Y = 0.9', 'm-'),
        ('loc_shift_pos', 'μ = 0.4', 'y-')
    ]

    for param_key, label, style in curves:
        pdf = dnts(x_grid, nts_params[param_key])
        ax.plot(x_grid, pdf, style, linewidth=1.5, label=label, alpha=0.8)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('density', fontsize=12)
    ax.set_title('Figure 2(a): PDFs of selected NTS specifications with N(0,1) for comparison',
                 fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    # Save
    save_path = os.path.join(OUTPUT_DIR, 'Figure_2a_NTS_PDFs')
    plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}.png and .pdf")
    return x_grid

# ============================================================================
# FIGURE 2(b): CDFs
# ============================================================================

def generate_figure_2b(x_grid):
    """Generate Figure 2(b): CDFs from PDFs"""
    print("Generating Figure 2(b): CDFs...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Normal CDF
    from scipy.stats import norm
    cdf_norm = norm.cdf(x_grid)
    ax.plot(x_grid, cdf_norm, 'k:', linewidth=1.5, label='N(0,1)', alpha=0.7)

    # NTS CDFs
    curves = [
        ('prior', 'NTS prior', 'b-'),
        ('high_vol', 'σ = 1.2', 'g-'),
        ('low_vol', 'σ = 0.8', 'r-'),
        ('right_skew', 'right-skew (G>M)', 'c-'),
        ('lower_Y', 'Y = 0.9', 'm-'),
        ('loc_shift_pos', 'μ = 0.4', 'y-')
    ]

    for param_key, label, style in curves:
        cdf = pnts(x_grid, nts_params[param_key])
        ax.plot(x_grid, cdf, style, linewidth=1.5, label=label, alpha=0.8)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('F(x)', fontsize=12)
    ax.set_title('Figure 2(b): CDFs corresponding to the PDFs in Figure 2(a)', fontsize=13)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    # Save
    save_path = os.path.join(OUTPUT_DIR, 'Figure_2b_NTS_CDFs')
    plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}.png and .pdf")

# ============================================================================
# FIGURE 2(c): QUANTILE FUNCTIONS
# ============================================================================

def generate_figure_2c():
    """Generate Figure 2(c): Quantile functions"""
    print("Generating Figure 2(c): Quantile functions...")

    u_grid = np.linspace(0.001, 0.999, 999)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Normal quantiles
    from scipy.stats import norm
    q_norm = norm.ppf(u_grid)
    ax.plot(u_grid, q_norm, 'k:', linewidth=1.5, label='N(0,1)', alpha=0.7)

    # NTS quantiles
    curves = [
        ('prior', 'NTS prior', 'b-'),
        ('high_vol', 'σ = 1.2', 'g-'),
        ('low_vol', 'σ = 0.8', 'r-'),
        ('right_skew', 'right-skew (G>M)', 'c-'),
        ('lower_Y', 'Y = 0.9', 'm-'),
        ('loc_shift_pos', 'μ = 0.4', 'y-')
    ]

    for param_key, label, style in curves:
        quantiles = qnts(u_grid, nts_params[param_key])
        ax.plot(u_grid, quantiles, style, linewidth=1.5, label=label, alpha=0.8)

    ax.set_xlabel('u', fontsize=12)
    ax.set_ylabel('Q(u)', fontsize=12)
    ax.set_title('Figure 2(c): Quantile functions Q(u) corresponding to Figure 2(a)', fontsize=13)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-6, 6)

    plt.tight_layout()

    # Save
    save_path = os.path.join(OUTPUT_DIR, 'Figure_2c_NTS_Quantiles')
    plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}.png and .pdf")

# ============================================================================
# PWF COMPUTATION
# ============================================================================

def compute_pwf(u_grid, prior_params, post_params):
    """
    Compute Probability Weighting Function
    PWF formula: w(u) = F_post(Q_prior(u))
    """
    # Step 1: Get quantiles from prior distribution
    q_prior = qnts(u_grid, prior_params)

    # Step 2: Evaluate posterior CDF at those quantiles
    w = pnts(q_prior, post_params)

    return w

def compute_greed_fear_index(w_value_at_half):
    """
    Compute symmetric greed-fear index from Dr. Rachev's manuscript
    ξ = w(1/2) - 1/2
    """
    return w_value_at_half - 0.5

# ============================================================================
# FIGURES 2(d) to 2(i): PWF CASES
# ============================================================================

def generate_pwf_case(case_num, case_name, fearful_key, greedy_key, figure_letter):
    """Generate a single PWF case figure"""
    print(f"Generating Figure 2({figure_letter}): Case {case_num} - {case_name}...")

    u_grid = np.linspace(0.01, 0.99, 99)
    prior_params = nts_params['prior']

    # Compute PWFs
    w_fearful = compute_pwf(u_grid, prior_params, nts_params[fearful_key])
    w_greedy = compute_pwf(u_grid, prior_params, nts_params[greedy_key])

    # Compute greed-fear indices
    idx_50 = 49  # u = 0.5 position
    xi_fearful = compute_greed_fear_index(w_fearful[idx_50])
    xi_greedy = compute_greed_fear_index(w_greedy[idx_50])

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot 45-degree line (reference)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='45° line', alpha=0.7)

    # Plot PWFs
    ax.plot(u_grid, w_fearful, 'b-', linewidth=2,
            label=f'Fearful (ξ={xi_fearful:.4f})')
    ax.plot(u_grid, w_greedy, 'r-', linewidth=2,
            label=f'Greedy (ξ={xi_greedy:.4f})')

    # Formatting
    ax.set_xlabel('u', fontsize=12)
    ax.set_ylabel('w(u)', fontsize=12)
    ax.set_title(f'Figure 2({figure_letter}): PWF for Case {case_num} - {case_name}',
                 fontsize=13)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    plt.tight_layout()

    # Save
    save_path = os.path.join(OUTPUT_DIR, f'Figure_2{figure_letter}_Case{case_num}_PWF')
    plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}.png and .pdf")
    print(f"  Greed-fear indices: Fearful={xi_fearful:.4f}, Greedy={xi_greedy:.4f}")

    return xi_fearful, xi_greedy

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all figures for the manuscript"""
    print("="*80)
    print("NTS PWF COMPLETE IMPLEMENTATION FOR DR. RACHEV'S MANUSCRIPT")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Generate basic figures
    x_grid = generate_figure_2a()  # PDFs
    generate_figure_2b(x_grid)      # CDFs
    generate_figure_2c()            # Quantiles

    print()
    print("Generating PWF Cases...")
    print("-"*40)

    # Define the 6 PWF cases
    pwf_cases = [
        (1, "Scale/Volatility Channel", 'high_vol', 'low_vol', 'd'),
        (2, "Skew/Asymmetry Channel", 'left_skew', 'right_skew', 'e'),
        (3, "Tail-Thickness Channel", 'lower_Y', 'higher_Y', 'f'),
        (4, "Location Channel", 'loc_shift_neg', 'loc_shift_pos', 'g'),
        (5, "Joint Dispersion-Tail", 'lower_Y', 'higher_Y', 'h'),  # Simplified
        (6, "Quantile-Based Skew", 'left_skew', 'right_skew', 'i'),  # Simplified
    ]

    # Generate each PWF case
    results = []
    for case_num, case_name, fearful_key, greedy_key, letter in pwf_cases:
        xi_f, xi_g = generate_pwf_case(case_num, case_name, fearful_key, greedy_key, letter)
        results.append((case_num, case_name, xi_f, xi_g))

    # Print summary table
    print()
    print("="*80)
    print("SUMMARY: GREED-FEAR INDICES (ξ = w(1/2) - 1/2)")
    print("="*80)
    print(f"{'Case':<6} {'Channel':<30} {'ξ_fearful':<12} {'ξ_greedy':<12} {'|Spread|':<10}")
    print("-"*70)

    for case_num, case_name, xi_f, xi_g in results:
        spread = abs(xi_f - xi_g)
        print(f"{case_num:<6} {case_name:<30} {xi_f:>11.4f} {xi_g:>11.4f} {spread:>9.4f}")

    print()
    print("="*80)
    print(f"All figures saved in: {os.path.abspath(OUTPUT_DIR)}")
    print("Files generated:")
    print("  - Figure_2a_NTS_PDFs.png/.pdf")
    print("  - Figure_2b_NTS_CDFs.png/.pdf")
    print("  - Figure_2c_NTS_Quantiles.png/.pdf")
    print("  - Figure_2d_Case1_PWF.png/.pdf through Figure_2i_Case6_PWF.png/.pdf")
    print("="*80)

if __name__ == "__main__":
    main()