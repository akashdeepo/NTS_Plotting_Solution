"""
Test script to generate NTS PDF plots using temStaPy library
Based on the parameters specified in the email for Figure 4.4(a)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Add temStaPy to path
sys.path.insert(0, r'c:\Users\Akash\OneDrive\Desktop\plotting issue\temStaPy_v0.5')

from temStaPy.distNTS import dnts, change_stdntsparam2ntsparam

# Parameter sets from the email
# Prior: mu=0, sigma=1, C=0.6, G=5, M=5, Y=1.2
# Scale channel: sigma=1.2 and sigma=0.8
# Skew channel: G=6.5, M=3.8 with GM fixed
# Tail channel: Y=0.9
# Location channel: mu=0.4

# Note: NTS uses different parameterization (alpha, theta, beta, gamma, mu)
# We need to map CGMY parameters (C, G, M, Y) to NTS parameters (alpha, theta, beta, gamma)

# For NTS: alpha corresponds to Y, and we need to compute theta, beta, gamma from C, G, M

def cgmy_to_nts(C, G, M, Y, mu=0, sigma=1):
    """
    Convert CGMY parameters to NTS parameters
    NTS parameterization: [alpha, theta, beta, gamma, mu]

    Based on the relationship:
    - alpha = Y (tail index)
    - theta relates to the scale
    - beta relates to asymmetry (G vs M)
    - gamma relates to volatility (sigma)
    """
    alpha = Y

    # These conversions are approximate and may need refinement
    # Based on the NTS literature and the temStaPy code structure
    theta = C * (G**Y + M**Y) / Y

    # Asymmetry parameter
    beta_raw = (M**Y - G**Y) / (M**Y + G**Y)

    # Scale parameters
    gamma = sigma
    beta = beta_raw * sigma

    return [alpha, theta, beta, gamma, mu]

# Define parameter sets
print("Setting up parameter configurations...")

# Prior configuration
prior_cgmy = {'mu': 0.0, 'sigma': 1.0, 'C': 0.6, 'G': 5.0, 'M': 5.0, 'Y': 1.2}
prior_nts = cgmy_to_nts(prior_cgmy['C'], prior_cgmy['G'], prior_cgmy['M'],
                        prior_cgmy['Y'], prior_cgmy['mu'], prior_cgmy['sigma'])

# High volatility
high_vol_cgmy = {**prior_cgmy, 'sigma': 1.2}
high_vol_nts = cgmy_to_nts(high_vol_cgmy['C'], high_vol_cgmy['G'], high_vol_cgmy['M'],
                           high_vol_cgmy['Y'], high_vol_cgmy['mu'], high_vol_cgmy['sigma'])

# Low volatility
low_vol_cgmy = {**prior_cgmy, 'sigma': 0.8}
low_vol_nts = cgmy_to_nts(low_vol_cgmy['C'], low_vol_cgmy['G'], low_vol_cgmy['M'],
                          low_vol_cgmy['Y'], low_vol_cgmy['mu'], low_vol_cgmy['sigma'])

# Right skew
right_skew_cgmy = {**prior_cgmy, 'G': 6.5, 'M': 3.8}
right_skew_nts = cgmy_to_nts(right_skew_cgmy['C'], right_skew_cgmy['G'], right_skew_cgmy['M'],
                             right_skew_cgmy['Y'], right_skew_cgmy['mu'], right_skew_cgmy['sigma'])

# Thick tails
thick_tails_cgmy = {**prior_cgmy, 'Y': 0.9}
thick_tails_nts = cgmy_to_nts(thick_tails_cgmy['C'], thick_tails_cgmy['G'], thick_tails_cgmy['M'],
                              thick_tails_cgmy['Y'], thick_tails_cgmy['mu'], thick_tails_cgmy['sigma'])

# Location shift
loc_shift_cgmy = {**prior_cgmy, 'mu': 0.4}
loc_shift_nts = cgmy_to_nts(loc_shift_cgmy['C'], loc_shift_cgmy['G'], loc_shift_cgmy['M'],
                            loc_shift_cgmy['Y'], loc_shift_cgmy['mu'], loc_shift_cgmy['sigma'])

# Test grid
xs = np.linspace(-5, 5, 801)

# Compute PDFs
print("Computing PDFs using temStaPy...")
try:
    pdf_prior = dnts(xs, prior_nts)
    print(f"Prior PDF computed: min={pdf_prior.min():.6f}, max={pdf_prior.max():.6f}")
except Exception as e:
    print(f"Error computing prior PDF: {e}")
    pdf_prior = np.zeros_like(xs)

try:
    pdf_high = dnts(xs, high_vol_nts)
    print(f"High vol PDF computed: min={pdf_high.min():.6f}, max={pdf_high.max():.6f}")
except Exception as e:
    print(f"Error computing high vol PDF: {e}")
    pdf_high = np.zeros_like(xs)

try:
    pdf_low = dnts(xs, low_vol_nts)
    print(f"Low vol PDF computed: min={pdf_low.min():.6f}, max={pdf_low.max():.6f}")
except Exception as e:
    print(f"Error computing low vol PDF: {e}")
    pdf_low = np.zeros_like(xs)

try:
    pdf_rskew = dnts(xs, right_skew_nts)
    print(f"Right skew PDF computed: min={pdf_rskew.min():.6f}, max={pdf_rskew.max():.6f}")
except Exception as e:
    print(f"Error computing right skew PDF: {e}")
    pdf_rskew = np.zeros_like(xs)

try:
    pdf_thick = dnts(xs, thick_tails_nts)
    print(f"Thick tails PDF computed: min={pdf_thick.min():.6f}, max={pdf_thick.max():.6f}")
except Exception as e:
    print(f"Error computing thick tails PDF: {e}")
    pdf_thick = np.zeros_like(xs)

try:
    pdf_loc = dnts(xs, loc_shift_nts)
    print(f"Location shift PDF computed: min={pdf_loc.min():.6f}, max={pdf_loc.max():.6f}")
except Exception as e:
    print(f"Error computing location shift PDF: {e}")
    pdf_loc = np.zeros_like(xs)

# Normal reference
pdf_norm = 1/np.sqrt(2*np.pi)*np.exp(-0.5*xs**2)
print(f"Normal PDF computed: min={pdf_norm.min():.6f}, max={pdf_norm.max():.6f}")

# Plot
print("Creating plot...")
plt.figure(figsize=(10, 6))
plt.plot(xs, pdf_norm, label='N(0,1)', linewidth=2, linestyle='--')
plt.plot(xs, pdf_prior, label='NTS prior', linewidth=1.5)
plt.plot(xs, pdf_high, label='σ=1.2 (high vol)', linewidth=1.5)
plt.plot(xs, pdf_low, label='σ=0.8 (low vol)', linewidth=1.5)
plt.plot(xs, pdf_rskew, label='Right skew (G>M)', linewidth=1.5)
plt.plot(xs, pdf_thick, label='Y=0.9 (thick tails)', linewidth=1.5)
plt.plot(xs, pdf_loc, label='μ=0.4 (location)', linewidth=1.5)

plt.xlabel('x', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Figure 4.4(a): PDFs of selected NTS specifications\n(with N(0,1) reference)', fontsize=13)
plt.legend(fontsize=9, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save
plt.savefig('Figure_4_4_a_NTS_PDFs_temStaPy.png', dpi=300, bbox_inches='tight')
plt.savefig('Figure_4_4_a_NTS_PDFs_temStaPy.pdf', bbox_inches='tight')
print("\nPlots saved:")
print("- Figure_4_4_a_NTS_PDFs_temStaPy.png")
print("- Figure_4_4_a_NTS_PDFs_temStaPy.pdf")

plt.show()

print("\nParameter mappings used:")
print(f"Prior NTS params: {prior_nts}")
print(f"High vol NTS params: {high_vol_nts}")
print(f"Low vol NTS params: {low_vol_nts}")
print(f"Right skew NTS params: {right_skew_nts}")
print(f"Thick tails NTS params: {thick_tails_nts}")
print(f"Location shift NTS params: {loc_shift_nts}")
