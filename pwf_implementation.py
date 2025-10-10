"""
Probability Weighting Functions (PWFs) for Normal Tempered-Stable Distributions

This script implements the PWF analysis following Section 4.3 structure,
adapted for NTS distributions using the temStaPy library.

Author: Akash Deep
Co-authors: W. Brent Lindquist, Svetlozar Rachev
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.integrate import trapezoid

# Add temStaPy to path
sys.path.insert(0, r'c:\Users\Akash\OneDrive\Desktop\PWFs for NTS\temStaPy_v0.5')

from temStaPy.distNTS import dnts, pnts, qnts

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cgmy_to_nts(C, G, M, Y, mu=0, sigma=1):
    """
    Convert CGMY parameters to NTS parameters

    Parameters:
    -----------
    C : float
        Activity/scale parameter
    G : float
        Left tempering rate
    M : float
        Right tempering rate
    Y : float
        Tail index (0 < Y < 2)
    mu : float
        Location parameter
    sigma : float
        Volatility parameter

    Returns:
    --------
    list : [alpha, theta, beta, gamma, mu]
        NTS parameters
    """
    alpha = Y
    theta = C * (G**Y + M**Y) / Y
    beta_raw = (M**Y - G**Y) / (M**Y + G**Y)
    gamma = sigma
    beta = beta_raw * sigma
    return [alpha, theta, beta, gamma, mu]


def compute_pwf(u_grid, prior_params, post_params):
    """
    Compute Probability Weighting Function

    PWF formula: w(u) = F_post(Q_prior(u))

    Parameters:
    -----------
    u_grid : array
        Probability levels in (0, 1)
    prior_params : list
        NTS parameters for prior distribution
    post_params : list
        NTS parameters for posterior distribution

    Returns:
    --------
    w : array
        PWF values w(u) for each u in u_grid
    """
    # Step 1: Get quantiles from prior distribution
    q_prior = qnts(u_grid, prior_params)

    # Step 2: Evaluate posterior CDF at those quantiles
    w = pnts(q_prior, post_params)

    return w


def plot_pwf_case(u_grid, w_fearful, w_greedy, case_name, case_number,
                  save_path=None, show_plot=True):
    """
    Create PWF plot for a single case

    Parameters:
    -----------
    u_grid : array
        Probability levels
    w_fearful : array
        PWF for fearful specification
    w_greedy : array
        PWF for greedy specification
    case_name : str
        Description of the case
    case_number : int
        Case number (1-8)
    save_path : str, optional
        Path to save figure
    show_plot : bool
        Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot 45-degree line (reference)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='45° line', alpha=0.7)

    # Plot PWFs
    ax.plot(u_grid, w_fearful, 'b-', linewidth=2, label='Fearful')
    ax.plot(u_grid, w_greedy, 'r-', linewidth=2, label='Greedy')

    # Formatting
    ax.set_xlabel('u (Prior Probability)', fontsize=12)
    ax.set_ylabel('w(u) (Posterior Probability)', fontsize=12)
    ax.set_title(f'Figure 4.4({chr(ord("e")+case_number-1)}): PWF for Case {case_number}\n{case_name}',
                 fontsize=13)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path + '.pdf', bbox_inches='tight')
        print(f"Saved: {save_path}.png and .pdf")

    if show_plot:
        plt.show()
    else:
        plt.close()


# ============================================================================
# PARAMETER SPECIFICATIONS
# ============================================================================

# Prior (benchmark) - symmetric NTS
PRIOR_CGMY = {'mu': 0.0, 'sigma': 1.0, 'C': 0.6, 'G': 5.0, 'M': 5.0, 'Y': 1.2}
PRIOR_NTS = cgmy_to_nts(**PRIOR_CGMY)

print("="*80)
print("NTS PROBABILITY WEIGHTING FUNCTIONS")
print("="*80)
print(f"\nPrior (benchmark) parameters:")
print(f"  CGMY: C={PRIOR_CGMY['C']}, G={PRIOR_CGMY['G']}, M={PRIOR_CGMY['M']}, Y={PRIOR_CGMY['Y']}")
print(f"  NTS:  alpha={PRIOR_NTS[0]:.3f}, theta={PRIOR_NTS[1]:.3f}, beta={PRIOR_NTS[2]:.3f}, "
      f"gamma={PRIOR_NTS[3]:.3f}, mu={PRIOR_NTS[4]:.3f}")

# Probability grid for PWF evaluation
u_grid = np.linspace(0.01, 0.99, 99)

# ============================================================================
# CASE 1: SCALE/VOLATILITY CHANNEL (sigma variations)
# ============================================================================

print("\n" + "="*80)
print("CASE 1: Scale/Volatility Channel")
print("="*80)

# Case 1 specifications
case1_fearful_cgmy = {**PRIOR_CGMY, 'sigma': 1.2}  # Higher volatility
case1_greedy_cgmy = {**PRIOR_CGMY, 'sigma': 0.8}   # Lower volatility

case1_fearful_nts = cgmy_to_nts(**case1_fearful_cgmy)
case1_greedy_nts = cgmy_to_nts(**case1_greedy_cgmy)

print(f"Fearful (high vol): sigma={case1_fearful_cgmy['sigma']}")
print(f"Greedy (low vol):   sigma={case1_greedy_cgmy['sigma']}")

# Compute PWFs
w1_fearful = compute_pwf(u_grid, PRIOR_NTS, case1_fearful_nts)
w1_greedy = compute_pwf(u_grid, PRIOR_NTS, case1_greedy_nts)

print(f"PWF computed successfully")
print(f"  Fearful: w(0.5) = {w1_fearful[49]:.4f}")
print(f"  Greedy:  w(0.5) = {w1_greedy[49]:.4f}")

# Plot
plot_pwf_case(u_grid, w1_fearful, w1_greedy,
              "Scale/Volatility Channel (σ variations)",
              case_number=1,
              save_path="Figure_4_4_e_Case1_PWF_volatility")

# ============================================================================
# CASE 2: SKEW/ASYMMETRY CHANNEL (G/M variations)
# ============================================================================

print("\n" + "="*80)
print("CASE 2: Skew/Asymmetry Channel")
print("="*80)

# Case 2 specifications
case2_greedy_cgmy = {**PRIOR_CGMY, 'G': 6.5, 'M': 3.8}   # Right skew (greedy)
case2_fearful_cgmy = {**PRIOR_CGMY, 'G': 3.8, 'M': 6.5}  # Left skew (fearful)

case2_greedy_nts = cgmy_to_nts(**case2_greedy_cgmy)
case2_fearful_nts = cgmy_to_nts(**case2_fearful_cgmy)

print(f"Greedy (right skew):  G={case2_greedy_cgmy['G']}, M={case2_greedy_cgmy['M']}")
print(f"Fearful (left skew):  G={case2_fearful_cgmy['G']}, M={case2_fearful_cgmy['M']}")

# Compute PWFs
w2_fearful = compute_pwf(u_grid, PRIOR_NTS, case2_fearful_nts)
w2_greedy = compute_pwf(u_grid, PRIOR_NTS, case2_greedy_nts)

print(f"PWF computed successfully")
print(f"  Fearful: w(0.5) = {w2_fearful[49]:.4f}")
print(f"  Greedy:  w(0.5) = {w2_greedy[49]:.4f}")

# Plot
plot_pwf_case(u_grid, w2_fearful, w2_greedy,
              "Skew/Asymmetry Channel (G/M variations)",
              case_number=2,
              save_path="Figure_4_4_f_Case2_PWF_skew")

# ============================================================================
# CASE 3: TAIL-THICKNESS CHANNEL (Y variations)
# ============================================================================

print("\n" + "="*80)
print("CASE 3: Tail-Thickness Channel")
print("="*80)

# Case 3 specifications
case3_fearful_cgmy = {**PRIOR_CGMY, 'Y': 0.9}   # Heavier tails
case3_greedy_cgmy = {**PRIOR_CGMY, 'Y': 1.5}    # Lighter tails

case3_fearful_nts = cgmy_to_nts(**case3_fearful_cgmy)
case3_greedy_nts = cgmy_to_nts(**case3_greedy_cgmy)

print(f"Fearful (heavy tails): Y={case3_fearful_cgmy['Y']}")
print(f"Greedy (light tails):  Y={case3_greedy_cgmy['Y']}")

# Compute PWFs
w3_fearful = compute_pwf(u_grid, PRIOR_NTS, case3_fearful_nts)
w3_greedy = compute_pwf(u_grid, PRIOR_NTS, case3_greedy_nts)

print(f"PWF computed successfully")
print(f"  Fearful: w(0.5) = {w3_fearful[49]:.4f}")
print(f"  Greedy:  w(0.5) = {w3_greedy[49]:.4f}")

# Plot
plot_pwf_case(u_grid, w3_fearful, w3_greedy,
              "Tail-Thickness Channel (Y variations)",
              case_number=3,
              save_path="Figure_4_4_g_Case3_PWF_tails")

# ============================================================================
# CASE 4: LOCATION CHANNEL (mu variations)
# ============================================================================

print("\n" + "="*80)
print("CASE 4: Location Channel")
print("="*80)

# Case 4 specifications
case4_greedy_cgmy = {**PRIOR_CGMY, 'mu': 0.4}    # Shift right (optimistic)
case4_fearful_cgmy = {**PRIOR_CGMY, 'mu': -0.4}  # Shift left (pessimistic)

case4_greedy_nts = cgmy_to_nts(**case4_greedy_cgmy)
case4_fearful_nts = cgmy_to_nts(**case4_fearful_cgmy)

print(f"Greedy (shift right):  mu={case4_greedy_cgmy['mu']}")
print(f"Fearful (shift left):  mu={case4_fearful_cgmy['mu']}")

# Compute PWFs
w4_fearful = compute_pwf(u_grid, PRIOR_NTS, case4_fearful_nts)
w4_greedy = compute_pwf(u_grid, PRIOR_NTS, case4_greedy_nts)

print(f"PWF computed successfully")
print(f"  Fearful: w(0.5) = {w4_fearful[49]:.4f}")
print(f"  Greedy:  w(0.5) = {w4_greedy[49]:.4f}")

# Plot
plot_pwf_case(u_grid, w4_fearful, w4_greedy,
              "Location Channel (μ variations)",
              case_number=4,
              save_path="Figure_4_4_h_Case4_PWF_location")

# ============================================================================
# CASE 5: JOINT DISPERSION-TAIL CHANNEL
# ============================================================================

print("\n" + "="*80)
print("CASE 5: Joint Dispersion-Tail Channel")
print("="*80)

# Case 5 specifications (without fixed variance)
# Fearful: Lower Y (heavier tails) + higher C (more activity)
# Greedy: Higher Y (lighter tails) + lower C (less activity)

case5_fearful_cgmy = {**PRIOR_CGMY, 'Y': 0.8, 'C': 0.8}
case5_greedy_cgmy = {**PRIOR_CGMY, 'Y': 1.5, 'C': 0.4}

case5_fearful_nts = cgmy_to_nts(**case5_fearful_cgmy)
case5_greedy_nts = cgmy_to_nts(**case5_greedy_cgmy)

print(f"Fearful (fat tails + high dispersion): Y={case5_fearful_cgmy['Y']}, C={case5_fearful_cgmy['C']}")
print(f"Greedy (thin tails + low dispersion):  Y={case5_greedy_cgmy['Y']}, C={case5_greedy_cgmy['C']}")

# Compute PWFs
w5_fearful = compute_pwf(u_grid, PRIOR_NTS, case5_fearful_nts)
w5_greedy = compute_pwf(u_grid, PRIOR_NTS, case5_greedy_nts)

print(f"PWF computed successfully")
print(f"  Fearful: w(0.5) = {w5_fearful[49]:.4f}")
print(f"  Greedy:  w(0.5) = {w5_greedy[49]:.4f}")

# Plot
plot_pwf_case(u_grid, w5_fearful, w5_greedy,
              "Joint Dispersion-Tail Channel",
              case_number=5,
              save_path="Figure_4_4_i_Case5_PWF_joint")

# ============================================================================
# CASE 6: QUANTILE-BASED SKEW WITH CONSTANT VARIANCE
# ============================================================================

print("\n" + "="*80)
print("CASE 6: Quantile-Based Skew (Constant Variance)")
print("="*80)
print("NOTE: This case requires variance calculation - using simplified version")

# Simplified Case 6: Right vs left skew with matched G*M product
GM_product = PRIOR_CGMY['G'] * PRIOR_CGMY['M']  # 5*5 = 25

case6_greedy_cgmy = {**PRIOR_CGMY, 'G': 6.5, 'M': 25/6.5}   # Right skew, preserve G*M
case6_fearful_cgmy = {**PRIOR_CGMY, 'G': 25/6.5, 'M': 6.5}  # Left skew, preserve G*M

case6_greedy_nts = cgmy_to_nts(**case6_greedy_cgmy)
case6_fearful_nts = cgmy_to_nts(**case6_fearful_cgmy)

print(f"Greedy (right skew):  G={case6_greedy_cgmy['G']:.2f}, M={case6_greedy_cgmy['M']:.2f}")
print(f"Fearful (left skew):  G={case6_fearful_cgmy['G']:.2f}, M={case6_fearful_cgmy['M']:.2f}")
print(f"G*M product preserved: {GM_product}")

# Compute PWFs
w6_fearful = compute_pwf(u_grid, PRIOR_NTS, case6_fearful_nts)
w6_greedy = compute_pwf(u_grid, PRIOR_NTS, case6_greedy_nts)

print(f"PWF computed successfully")
print(f"  Fearful: w(0.5) = {w6_fearful[49]:.4f}")
print(f"  Greedy:  w(0.5) = {w6_greedy[49]:.4f}")

# Plot
plot_pwf_case(u_grid, w6_fearful, w6_greedy,
              "Quantile-Based Skew (Simplified)",
              case_number=6,
              save_path="Figure_4_4_j_Case6_PWF_quantile_skew")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: ALL PWF CASES COMPLETED")
print("="*80)

summary_data = [
    ("Case 1", "Scale/Volatility", w1_fearful[49], w1_greedy[49]),
    ("Case 2", "Skew/Asymmetry", w2_fearful[49], w2_greedy[49]),
    ("Case 3", "Tail Thickness", w3_fearful[49], w3_greedy[49]),
    ("Case 4", "Location", w4_fearful[49], w4_greedy[49]),
    ("Case 5", "Joint Dispersion-Tail", w5_fearful[49], w5_greedy[49]),
    ("Case 6", "Quantile Skew", w6_fearful[49], w6_greedy[49]),
]

print(f"\n{'Case':<10} {'Channel':<25} {'w_fear(0.5)':<15} {'w_greed(0.5)':<15}")
print("-"*65)
for case, channel, w_f, w_g in summary_data:
    print(f"{case:<10} {channel:<25} {w_f:<15.4f} {w_g:<15.4f}")

print("\n" + "="*80)
print("All PWF plots saved in current directory")
print("Files: Figure_4_4_e through Figure_4_4_j (.png and .pdf)")
print("="*80)
