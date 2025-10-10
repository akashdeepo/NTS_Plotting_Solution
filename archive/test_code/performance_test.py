"""
Performance Testing for NTS PWF Computations
"""

import numpy as np
import time
import sys

# Add temStaPy to path
sys.path.insert(0, r'c:\Users\Akash\OneDrive\Desktop\PWFs for NTS\temStaPy_v0.5')
from temStaPy.distNTS import dnts, pnts, qnts

def time_nts_computation():
    """Time various NTS computations"""

    # Test parameters
    nts_params = [1.2, 6.899, 0.0, 1.0, 0.0]  # alpha, theta, beta, gamma, mu
    x_grid = np.linspace(-5, 5, 1000)
    u_grid = np.linspace(0.01, 0.99, 99)

    print("="*60)
    print("NTS COMPUTATION PERFORMANCE TESTING")
    print("="*60)

    # Test 1: Density computation
    start = time.time()
    densities = dnts(x_grid, nts_params)
    density_time = time.time() - start
    print(f"\n1. Density computation (1000 points):")
    print(f"   Time: {density_time:.4f} seconds")
    print(f"   Per point: {density_time/1000*1000:.2f} ms")

    # Test 2: CDF computation
    start = time.time()
    cdfs = pnts(x_grid, nts_params)
    cdf_time = time.time() - start
    print(f"\n2. CDF computation (1000 points):")
    print(f"   Time: {cdf_time:.4f} seconds")
    print(f"   Per point: {cdf_time/1000*1000:.2f} ms")

    # Test 3: Quantile computation
    start = time.time()
    quantiles = qnts(u_grid, nts_params)
    quantile_time = time.time() - start
    print(f"\n3. Quantile computation (99 points):")
    print(f"   Time: {quantile_time:.4f} seconds")
    print(f"   Per point: {quantile_time/99*1000:.2f} ms")

    # Test 4: Full PWF computation
    print(f"\n4. Full PWF computation:")
    prior_params = nts_params
    post_params = [1.2, 6.899, 0.2, 1.0, 0.0]  # Slightly different beta

    start = time.time()
    # Step 1: Get quantiles from prior
    q_prior = qnts(u_grid, prior_params)
    q_time = time.time() - start

    # Step 2: Evaluate posterior CDF
    start2 = time.time()
    w = pnts(q_prior, post_params)
    cdf_eval_time = time.time() - start2

    total_pwf_time = q_time + cdf_eval_time
    print(f"   Quantile step: {q_time:.4f} seconds")
    print(f"   CDF eval step: {cdf_eval_time:.4f} seconds")
    print(f"   Total PWF time: {total_pwf_time:.4f} seconds")

    # Compare with claimed times
    print(f"\n" + "="*60)
    print("COMPARISON WITH PAPER CLAIMS:")
    print("="*60)
    print(f"Paper claim: ~0.05 seconds per distribution")
    print(f"Our result: {density_time:.4f} seconds for density")
    print(f"            {cdf_time:.4f} seconds for CDF")
    print(f"            {total_pwf_time:.4f} seconds for PWF")

    # Check if we're in the right ballpark
    if density_time < 0.1 and cdf_time < 0.1:
        print("\n✓ Performance is excellent (< 0.1s)")
    elif density_time < 0.5 and cdf_time < 0.5:
        print("\n✓ Performance is good (< 0.5s)")
    else:
        print("\n⚠ Performance might need optimization")

    return {
        'density_time': density_time,
        'cdf_time': cdf_time,
        'quantile_time': quantile_time,
        'pwf_time': total_pwf_time
    }

if __name__ == "__main__":
    results = time_nts_computation()

    # Summary
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Average computation time per distribution:")
    avg_time = np.mean([results['density_time'], results['cdf_time']])
    print(f"  {avg_time:.4f} seconds")
    print(f"\nThis is {400/avg_time:.0f}x faster than the 400-second")
    print(f"COS method mentioned in the paper introduction!")