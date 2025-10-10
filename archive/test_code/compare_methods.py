"""
Compare different methods for computing NTS densities:
1. temStaPy library (FFT-based Gil-Pelaez)
2. GPT-5's COS method with mpmath
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time

# Add temStaPy to path
sys.path.insert(0, r'c:\Users\Akash\OneDrive\Desktop\plotting issue\temStaPy_v0.5')

from temStaPy.distNTS import dnts

print("="*80)
print("NTS Density Computation - Method Comparison")
print("="*80)

# Test parameters - using standard NTS parameterization from documentation
# Example from temStaR docs: alpha=1.2, theta=1, beta=-0.2, gamma=0.3, mu=0.1
test_params = [1.2, 1.0, -0.2, 0.3, 0.1]

xs = np.linspace(-3, 3, 401)

print("\n1. Testing temStaPy (FFT-Gil-Pelaez method)...")
start = time.time()
try:
    pdf_temstapy = dnts(xs, test_params)
    elapsed = time.time() - start
    print(f"   [OK] Success in {elapsed:.3f}s")
    print(f"   - Min: {pdf_temstapy.min():.6f}")
    print(f"   - Max: {pdf_temstapy.max():.6f}")
    print(f"   - Integral (trapz): {np.trapz(pdf_temstapy, xs):.6f} (should be ~1.0)")
    temstapy_works = True
except Exception as e:
    print(f"   [FAIL] Error: {e}")
    pdf_temstapy = np.zeros_like(xs)
    temstapy_works = False

print("\n2. Testing mpmath COS method (GPT-5 approach)...")
try:
    import mpmath as mp
    mp.mp.dps = 30  # Lower precision for speed test

    def psi_TS_cgmy(s, C, G, M, Y):
        """CGMY/KoBoL Laplace exponent"""
        return C * mp.gamma(-Y) * ((M - s)**Y - M**Y + (G + s)**Y - G**Y)

    def phi_nts_cgmy(u, mu, sigma, C, G, M, Y):
        """NTS characteristic function via CGMY"""
        s = 1j*mu*u - 0.5*(sigma**2)*u**2
        return mp.exp(psi_TS_cgmy(s, C, G, M, Y))

    def cos_pdf(x, mu, sigma, C, G, M, Y, L=12.0, K=4096, alpha=0.4):
        """Damped COS method"""
        k = np.arange(K)
        uk = k*np.pi/L
        cf = np.array([complex(phi_nts_cgmy(mp.mpf(ui)-1j*alpha, mu, sigma, C, G, M, Y))
                      for ui in uk])
        Vk = np.cos(uk*x) * np.exp(alpha*x)
        Vk[0] *= 0.5
        return (1/L) * np.real(np.dot(cf, Vk))

    # Test with CGMY parameters
    start = time.time()
    # This is SLOW - just test one point
    test_x = 0.0
    test_pdf_val = cos_pdf(test_x, 0.0, 1.0, 0.6, 5.0, 5.0, 1.2)
    elapsed = time.time() - start
    print(f"   [OK] COS method available")
    print(f"   - Time for single point: {elapsed:.3f}s")
    print(f"   - PDF at x=0: {test_pdf_val:.6f}")
    print(f"   [WARNING] Full computation would take ~{elapsed*len(xs):.1f}s for {len(xs)} points")
    mpmath_available = True

except ImportError:
    print("   [FAIL] mpmath not installed - COS method unavailable")
    print("   Install with: pip install mpmath")
    mpmath_available = False
except Exception as e:
    print(f"   [FAIL] Error: {e}")
    mpmath_available = False

# Plot comparison
if temstapy_works:
    print("\n3. Generating comparison plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: PDF
    axes[0].plot(xs, pdf_temstapy, 'b-', linewidth=2, label='temStaPy (FFT)')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title(f'NTS PDF\nParams: α={test_params[0]}, θ={test_params[1]}, ' +
                      f'β={test_params[2]}, γ={test_params[3]}, μ={test_params[4]}',
                      fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Right plot: Log-scale (to see tails)
    axes[1].semilogy(xs, pdf_temstapy + 1e-10, 'b-', linewidth=2, label='temStaPy (FFT)')
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('Density (log scale)', fontsize=12)
    axes[1].set_title('NTS PDF - Log Scale\n(showing tail behavior)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('nts_method_comparison.png', dpi=300, bbox_inches='tight')
    print("   [OK] Plot saved: nts_method_comparison.png")
    plt.show()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"temStaPy (FFT):  {'[OK] Working' if temstapy_works else '[FAIL] Failed'}")
print(f"mpmath COS:      {'[OK] Available (but slow)' if mpmath_available else '[FAIL] Not available'}")
print("\nRECOMMENDATION:")
print("-> Use temStaPy for production work")
print("-> Fast, stable, and well-tested")
print("-> mpmath COS is theoretically elegant but computationally expensive")
print("="*80)
