# Probability Weighting Functions for Normal Tempered-Stable Distributions

**Research Paper Documentation**

**Authors**: Akash Deep, W. Brent Lindquist, Svetlozar T. Rachev
**Date**: October 2025
**Institution**: Texas Tech University

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Background](#theoretical-background)
3. [PWF Definition and Computation](#pwf-definition-and-computation)
4. [The Six Cases](#the-six-cases)
5. [Implementation Details](#implementation-details)
6. [Results Interpretation](#results-interpretation)
7. [Technical Notes](#technical-notes)
8. [References](#references)

---

## Overview

This document describes the implementation of Probability Weighting Functions (PWFs) for the Normal Tempered-Stable (NTS) distribution family. The work extends Section 4.3 (Tempered-Stable distributions) to the NTS case, following the structure laid out in "Bridging Behavioral and Rational Finance: New Classes of Probability Weighting Functions."

### Purpose

PWFs map probabilities to probabilities, revealing how different distributional specifications (fearful vs. greedy) transform decision weights. They provide a canonical, scale-invariant way to compare behavioral dispositions across parametric families.

### Key Innovation

While Section 4.3 analyzed pure tempered-stable (TS) distributions, this work addresses Normal Tempered-Stable (NTS) distributions, which subordinate Brownian motion to a TS process. NTS distributions preserve semi-heavy tails while maintaining conditional Gaussianity, making them particularly relevant for:
- Asset return modeling
- Option pricing
- Risk management
- Behavioral finance applications

---

## Theoretical Background

### Normal Tempered-Stable Distribution

The NTS distribution arises from the variance-mean mixture:

```
X = μT + σ√T N
```

Where:
- `T ~ TS(C, G, M, Y)` is a tempered-stable subordinator
- `N ~ N(0,1)` is a standard normal
- `T ⊥ N` (independence)
- `μ ∈ ℝ` is the location parameter
- `σ > 0` is the volatility parameter

### CGMY/KoBoL Parameterization

The subordinator T follows the CGMY/KoBoL specification with:

- **C > 0**: Activity/scale parameter (jump intensity)
- **G > 0**: Left tempering rate
- **M > 0**: Right tempering rate
- **Y ∈ (0,2)**: Tail index (Blumenthal-Getoor activity index)

### Parameter Roles

| Parameter | Economic Interpretation | Effect |
|-----------|------------------------|--------|
| **μ** | Location/drift | Horizontal shift, no shape change |
| **σ** | Volatility/scale | Symmetric dispersion around center |
| **C** | Jump activity | Overall tail weight and central thickness |
| **G, M** | Tempering rates | Asymmetry: G>M gives right skew, G<M gives left skew |
| **Y** | Tail index | Tail heaviness: lower Y = heavier tails |

### Key Properties

1. **Moments**: All positive moments exist for Y ∈ (0,2) with G, M > 0
2. **Tails**: Semi-heavy (between Gaussian and α-stable)
3. **Characteristic Function**:
   ```
   φ_X(u) = exp{Ψ_T(iμu - ½σ²u²)}
   ```
4. **Numerical Stability**: FFT-based methods work reliably

---

## PWF Definition and Computation

### Definition

For a prior distribution F_prior and posterior distribution F_post, the Probability Weighting Function is:

```
w(u) = F_post(Q_prior(u))    for u ∈ (0,1)
```

Where:
- `u` = probability level
- `Q_prior(u)` = quantile function (inverse CDF) of prior
- `F_post(x)` = CDF of posterior
- `w(u)` = transformed probability

### Interpretation

**What does w(u) mean?**

For a given prior probability level u:
1. Find the x-value at the u-th quantile of the prior: `x* = Q_prior(u)`
2. Ask: "What percentile is x* in the posterior distribution?"
3. That percentile is `w(u) = F_post(x*)`

**Example**:
- Prior has median (50th percentile) at x = 0
- Posterior might have F_post(0) = 0.45
- This means w(0.5) = 0.45
- Interpretation: What was the median in the prior is now the 45th percentile in the posterior

### Geometric Signatures

The shape of w(u) reveals the nature of the distributional change:

| PWF Shape | Interpretation |
|-----------|---------------|
| Above diagonal (u < 0.5) | Posterior puts less mass in left tail |
| Below diagonal (u < 0.5) | Posterior puts more mass in left tail |
| Above diagonal (u > 0.5) | Posterior puts more mass in right tail |
| Below diagonal (u > 0.5) | Posterior puts less mass in right tail |
| Concave | Greedy: compresses tails, expands center |
| Convex | Fearful: expands tails, compresses center |
| S-shape | Combined tail and scale effects |
| Affine tilt | Pure location shift |

---

## The Six Cases

### Case 1: Scale/Volatility Channel (σ variations)

**Purpose**: Isolate dispersion effects without changing asymmetry or tail index

**Specifications**:
- **Prior**: σ = 1.0, C = 0.6, G = M = 5, Y = 1.2, μ = 0
- **Fearful**: σ = 1.2 (higher volatility, wider distribution)
- **Greedy**: σ = 0.8 (lower volatility, tighter distribution)

**Economic Interpretation**:
- **Fearful**: Accepts greater uncertainty; assigns more weight to extreme outcomes (both gains and losses)
- **Greedy**: Seeks concentrated outcomes; underweights extremes, overweights median

**Expected PWF Shape**:
- Fearful: Below diagonal in middle, approaches diagonal at extremes (symmetric S-shape)
- Greedy: Above diagonal in middle, approaches diagonal at extremes (inverted S)

**Behavioral Finance Mapping**:
- High σ = Fear/uncertainty regime (flight to quality, VIX spike)
- Low σ = Greed/calm regime (search for yield, complacency)

---

### Case 2: Skew/Asymmetry Channel (G/M variations)

**Purpose**: Vary left/right tail balance while preserving dispersion and tail index

**Specifications**:
- **Prior**: G = M = 5 (symmetric), C = 0.6, Y = 1.2, σ = 1.0, μ = 0
- **Greedy**: G = 6.5, M = 3.8 (right skew, longer right tail)
- **Fearful**: G = 3.8, M = 6.5 (left skew, longer left tail)

**Economic Interpretation**:
- **Greedy**: Optimistic; stretches upside potential, attenuates downside
- **Fearful**: Pessimistic; amplifies downside risk, compresses upside

**Expected PWF Shape**:
- Greedy: Strongly concave, entirely above diagonal
- Fearful: Strongly convex, entirely below diagonal

**Behavioral Finance Mapping**:
- Right skew (G>M) = Bull market sentiment, carry trade attractiveness
- Left skew (G<M) = Bear market sentiment, crash concerns

**Key Feature**: No crossing of diagonal; pure asymmetry effect

---

### Case 3: Tail-Thickness Channel (Y variations)

**Purpose**: Change tail heaviness while maintaining symmetry and location

**Specifications**:
- **Prior**: Y = 1.2, C = 0.6, G = M = 5, σ = 1.0, μ = 0
- **Fearful**: Y = 0.9 (heavier tails, more tail activity)
- **Greedy**: Y = 1.5 (lighter tails, more Gaussian-like)

**Economic Interpretation**:
- **Fearful**: Elevated rare-event salience; both extreme gains and losses are more likely
- **Greedy**: Confidence in mean-reversion; extremes are discounted

**Expected PWF Shape**:
- Fearful: Below diagonal for u > 0.5, above for u < 0.5 (symmetric about 0.5)
- Greedy: Above diagonal for u > 0.5, below for u < 0.5

**Behavioral Finance Mapping**:
- Low Y = Tail-risk regime (post-crisis, high implied volatility skew)
- High Y = Normal times (thin tails, stable markets)

**Key Feature**: Curvature concentrated near shoulders (10th and 90th percentiles), not at median

---

### Case 4: Location Channel (μ variations)

**Purpose**: Pure translation without shape change

**Specifications**:
- **Prior**: μ = 0, C = 0.6, G = M = 5, Y = 1.2, σ = 1.0
- **Greedy**: μ = +0.4 (shift right, optimistic drift)
- **Fearful**: μ = -0.4 (shift left, pessimistic drift)

**Economic Interpretation**:
- **Greedy**: Positive expected return belief
- **Fearful**: Negative expected return belief

**Expected PWF Shape**:
- Nearly affine (straight line) with single crossing near u = 0.5
- Greedy: Above diagonal for u < 0.5, below for u > 0.5
- Fearful: Below diagonal for u < 0.5, above for u > 0.5

**Behavioral Finance Mapping**:
- Positive μ = Bullish outlook, growth expectations
- Negative μ = Bearish outlook, recession fears

**Key Feature**: Minimal curvature; pure sentiment tilt

---

### Case 5: Joint Dispersion-Tail Channel

**Purpose**: Combine scale and tail effects

**Specifications**:
- **Prior**: C = 0.6, Y = 1.2, G = M = 5, σ = 1.0, μ = 0
- **Fearful**: Y = 0.8, C = 0.8 (heavier tails + more activity)
- **Greedy**: Y = 1.5, C = 0.4 (lighter tails + less activity)

**Economic Interpretation**:
- **Fearful**: Stress regime; both central dispersion and tail risk elevated
- **Greedy**: Calm regime; compressed volatility and thin tails

**Expected PWF Shape**:
- Fearful: Strong S-shape with pronounced tail emphasis
- Greedy: Inverted S with center concentration

**Behavioral Finance Mapping**:
- Combined stress = 2008 crisis, COVID-19 crash
- Combined calm = 2017 low-vol regime, "Goldilocks" economy

**Note**: This case does NOT fix variance. For fixed-variance version, see Section 4.3 methodology.

---

### Case 6: Quantile-Based Skew with Constant Variance

**Purpose**: Pure skew effect without changing overall dispersion

**Specifications** (Simplified):
- **Prior**: G = M = 5 (G×M = 25), C = 0.6, Y = 1.2, σ = 1.0, μ = 0
- **Greedy**: G = 6.5, M = 25/6.5 ≈ 3.85 (preserve G×M product)
- **Fearful**: G = 25/6.5 ≈ 3.85, M = 6.5

**Economic Interpretation**:
- Redistribute tail probabilities left-to-right without changing total tail mass

**Expected PWF Shape**:
- Nearly linear through mid-quantiles
- Curvature concentrated in extremes (upper and lower deciles)

**Technical Note**:
Full implementation requires:
1. Compute variance of prior
2. Adjust C or σ to match variance exactly
3. This is more complex and omitted from initial implementation

---

## Implementation Details

### Dependencies

```python
import numpy as np
import matplotlib.pyplot as plt
from temStaPy.distNTS import dnts, pnts, qnts
```

### Core Function: `compute_pwf()`

```python
def compute_pwf(u_grid, prior_params, post_params):
    """
    Compute PWF: w(u) = F_post(Q_prior(u))

    Parameters:
    -----------
    u_grid : array
        Probability levels in (0, 1), e.g., [0.01, 0.02, ..., 0.99]
    prior_params : list
        [alpha, theta, beta, gamma, mu] for prior NTS
    post_params : list
        [alpha, theta, beta, gamma, mu] for posterior NTS

    Returns:
    --------
    w : array
        PWF values for each u
    """
    # Step 1: Get prior quantiles
    q_prior = qnts(u_grid, prior_params)

    # Step 2: Evaluate posterior CDF at those quantiles
    w = pnts(q_prior, post_params)

    return w
```

### Parameter Conversion: CGMY → NTS

The temStaPy library uses NTS parameterization `[α, θ, β, γ, μ]`, while the theoretical framework uses CGMY `[C, G, M, Y, μ, σ]`.

**Conversion formula**:
```python
def cgmy_to_nts(C, G, M, Y, mu=0, sigma=1):
    alpha = Y  # Tail index
    theta = C * (G**Y + M**Y) / Y  # Scale from tempering
    beta_raw = (M**Y - G**Y) / (M**Y + G**Y)  # Asymmetry
    gamma = sigma  # Volatility
    beta = beta_raw * sigma  # Scaled asymmetry
    return [alpha, theta, beta, gamma, mu]
```

**Important**: This conversion is approximate. For exact relationships, see:
- Kim, Y.S. (2020). "Portfolio Optimization on the Dispersion Risk and the Asymmetric Tail Risk"
- https://arxiv.org/pdf/2007.13972.pdf

### Numerical Grid

```python
u_grid = np.linspace(0.01, 0.99, 99)
```

- Excludes u=0 and u=1 (boundary singularities)
- 99 points provides smooth curves
- Can increase to 199 or 999 for publication-quality smoothness

### Computational Performance

**Typical Timings** (per case):
- Quantile computation: ~0.5 seconds
- CDF computation: ~0.5 seconds
- Total per case: ~1 second
- All 6 cases: ~6 seconds

**Bottleneck**: FFT-based inversion in temStaPy

---

## Results Interpretation

### Reading PWF Plots

Each figure shows:
- **Black dashed line**: 45° diagonal (w(u) = u, no change)
- **Blue curve**: Fearful specification
- **Red curve**: Greedy specification

### Diagnostic Questions

1. **Does the curve cross the diagonal?**
   - No crossing (Case 2, 3): Pure one-sided effect
   - Single crossing (Case 4): Location shift
   - Multiple crossings: Complex redistribution

2. **Where is curvature strongest?**
   - Center (u ≈ 0.5): Scale/volatility effect
   - Shoulders (u ≈ 0.2, 0.8): Tail effect
   - Extremes (u < 0.1, u > 0.9): Asymmetry effect

3. **Is curvature symmetric?**
   - Symmetric S-shape: Pure scale or tail change
   - Asymmetric: Skew component present

4. **Distance from diagonal?**
   - Large deviations: Strong behavioral shift
   - Near diagonal: Weak effect

### Case-by-Case Expectations

| Case | Fearful Curve | Greedy Curve |
|------|---------------|--------------|
| 1 (σ) | Below diagonal in middle | Above diagonal in middle |
| 2 (G/M) | Entirely below, convex | Entirely above, concave |
| 3 (Y) | Below upper half, above lower | Above upper half, below lower |
| 4 (μ) | Below left, above right | Above left, below right |
| 5 (C,Y) | Strong S-shape | Inverted strong S |
| 6 (G/M const var) | Below, convex, flat middle | Above, concave, flat middle |

---

## Technical Notes

### Numerical Stability

**Potential Issues**:
1. **Quantile computation**: May fail for u too close to 0 or 1
2. **CDF computation**: Numerical integration errors in extreme tails
3. **Parameter ranges**: Very small Y or large C can cause instability

**Solutions**:
- Use u ∈ [0.01, 0.99] instead of [0, 1]
- Increase temStaPy grid resolution if needed
- Check for NaN or Inf values in output

### Validation Checks

After computing each PWF, verify:

```python
# 1. Monotonicity: w should be increasing
assert np.all(np.diff(w) >= 0), "PWF not monotonic!"

# 2. Boundary conditions
assert abs(w[0] - u_grid[0]) < 0.05, "Left boundary off"
assert abs(w[-1] - u_grid[-1]) < 0.05, "Right boundary off"

# 3. Range
assert np.all((w >= 0) & (w <= 1)), "PWF outside [0,1]"
```

### Comparison with Section 4.3

**Key Differences**:

| Aspect | Section 4.3 (TS) | This Work (NTS) |
|--------|------------------|-----------------|
| Distribution | Pure tempered-stable | Subordinated Brownian |
| Parameters | [C, G, M, Y, μ] | [C, G, M, Y, μ, σ] |
| Library | Custom FFT code | temStaPy |
| Moments | All exist | All exist |
| Calibration | Option-implied | Also fits returns |

**Why NTS Matters**:
- Conditional Gaussianity aids filtering and estimation
- Natural for stochastic volatility models
- Better empirical fit for high-frequency returns
- Easier to extend to multivariate settings

---

## Parameter Sensitivity

### Robustness Tests

To ensure results are not artifacts of specific parameter choices:

1. **Vary prior slightly**: Change C from 0.6 to 0.5 or 0.7
2. **Check multiple sigma ratios**: Try 0.7/1.0/1.3 instead of 0.8/1.0/1.2
3. **Test different Y baseline**: Use Y=1.0 or Y=1.4 as prior

Expected: PWF shapes should be qualitatively similar

### Known Sensitivities

- **Y near 0**: Computation becomes very slow (infinite small-jump activity)
- **Y near 2**: Approaches Gaussian, loses TS character
- **G, M very different**: Extreme skew can cause quantile solver issues
- **C very large**: May overflow in theta calculation

**Recommended Ranges**:
- Y ∈ [0.5, 1.8]
- C ∈ [0.1, 2.0]
- G, M ∈ [1, 10]
- σ ∈ [0.5, 2.0]

---

## Extensions and Future Work

### Case 7: Volatility-Adjusted Mean Channel

**Not yet implemented**. Would require:

```python
# Fix Sharpe-like ratio: mu / sigma
sharpe_benchmark = 0.0
sharpe_greedy = 0.3
sharpe_fearful = -0.3

# Adjust mu and sigma jointly
case7_greedy = {C, G, M, Y, mu: sharpe_greedy * sigma_greedy, sigma: sigma_greedy}
case7_fearful = {C, G, M, Y, mu: sharpe_fearful * sigma_fearful, sigma: sigma_fearful}
```

### Case 8: Scale-Skew Stress Test

**Not yet implemented**. Would combine:
- Fearful: High σ + left skew (G < M)
- Greedy: Low σ + right skew (G > M)

```python
case8_fearful = {sigma: 1.3, G: 3.5, M: 6.5, ...}
case8_greedy = {sigma: 0.7, G: 6.5, M: 3.5, ...}
```

### Multivariate Extension

NTS has natural multivariate generalizations (see temStaPy's `distMNTS.py`). Future work could:
1. Compute bivariate PWFs
2. Analyze copula structures
3. Portfolio-level probability weighting

### Empirical Calibration

Next steps:
1. Fit NTS to S&P 500 returns (normal times)
2. Fit NTS to crisis periods (2008, 2020)
3. Compare implied PWFs
4. Link to option-implied parameters

---

## References

### Primary Literature

1. **Kim, Y.S., Rachev, S.T., Bianchi, M.L., & Fabozzi, F.J.** (2010). Tempered stable and tempered infinitely divisible GARCH models. *Journal of Banking & Finance*, 34(9), 2096–2109.

2. **Kim, Y.S.** (2020). Portfolio Optimization on the Dispersion Risk and the Asymmetric Tail Risk. arXiv:2007.13972. https://arxiv.org/pdf/2007.13972.pdf

3. **Carr, P., Geman, H., Madan, D., & Yor, M.** (2002). The fine structure of asset returns: An empirical investigation. *Journal of Business*, 75(2), 305–332.

4. **Rosiński, J.** (2007). Tempering stable processes. *Stochastic Processes and their Applications*, 117(6), 677–707.

### Textbooks

5. **Cont, R., & Tankov, P.** (2004). *Financial Modelling with Jump Processes*. Chapman & Hall/CRC.

6. **Rachev, S.T., Kim, Y.S., Bianchi, M.L., & Fabozzi, F.J.** (2011). *Financial Models with Lévy Processes and Volatility Clustering*. Wiley.

7. **Sato, K.** (1999). *Lévy Processes and Infinitely Divisible Distributions*. Cambridge University Press.

### Related Work

8. **Barndorff-Nielsen, O.E.** (1997). Normal inverse Gaussian distributions and stochastic volatility modelling. *Scandinavian Journal of Statistics*, 24(1), 1–13.

9. **Koponen, I.** (1995). Analytic approach to the problem of convergence of truncated Lévy flights towards the Gaussian stochastic process. *Physical Review E*, 52(1), 1197.

10. **Applebaum, D.** (2009). *Lévy Processes and Stochastic Calculus* (2nd ed.). Cambridge University Press.

---

## Appendix A: Complete Parameter Table

### Prior (Benchmark) Specification

| Framework | Parameters |
|-----------|------------|
| **CGMY** | C=0.6, G=5.0, M=5.0, Y=1.2, μ=0.0, σ=1.0 |
| **NTS** | α=1.2, θ=6.899, β=0.0, γ=1.0, μ=0.0 |

### All Cases Summary

| Case | Channel | Fearful Spec | Greedy Spec |
|------|---------|--------------|-------------|
| **1** | σ | σ=1.2 | σ=0.8 |
| **2** | G/M | G=3.8, M=6.5 | G=6.5, M=3.8 |
| **3** | Y | Y=0.9 | Y=1.5 |
| **4** | μ | μ=-0.4 | μ=+0.4 |
| **5** | C,Y | C=0.8, Y=0.8 | C=0.4, Y=1.5 |
| **6** | G/M (var fix) | G=3.85, M=6.5 | G=6.5, M=3.85 |

---

## Appendix B: Diagnostic Flowchart

```
                    ┌─────────────────────┐
                    │   Run PWF Code      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Check for NaN/Inf   │
                    └──────────┬──────────┘
                               │
                          Yes ─┤
                    ┌─────────────────────┐
                    │ Adjust parameters:  │
                    │ • Smaller Y range   │
                    │ • Reduce C          │
                    │ • Check G, M        │
                    └─────────────────────┘
                               │
                          No ──┤
                               ▼
                    ┌─────────────────────┐
                    │ Verify monotonicity │
                    └──────────┬──────────┘
                               │
                         OK ───┤
                    ┌─────────────────────┐
                    │ Check visual shape  │
                    │ • Case 1: S-shape?  │
                    │ • Case 2: Concave?  │
                    │ • Case 4: Affine?   │
                    └──────────┬──────────┘
                               │
                      Looks ───┤
                      good     │
                               ▼
                    ┌─────────────────────┐
                    │ READY FOR PAPER     │
                    └─────────────────────┘
```

---

## Appendix C: Quick Start Guide

### For Someone Running This Code

1. **Install dependencies**:
   ```bash
   pip install numpy matplotlib scipy
   ```

2. **Clone temStaPy**:
   ```bash
   git clone https://github.com/aaron9011/temStaPy_v0.5
   ```

3. **Run the code**:
   ```bash
   python pwf_implementation.py
   ```

4. **Check outputs**:
   - Console: Summary table of w(0.5) values
   - Files: `Figure_4_4_e.png` through `Figure_4_4_j.png`

5. **Validate**:
   - All curves should be smooth
   - No NaN or Inf values
   - Shapes match expectations in table above

### Expected Runtime

- **Full script**: ~10 seconds
- **Per case**: ~1.5 seconds
- **Bottleneck**: CDF/quantile computations

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "Module not found: docx" | `pip install python-docx` |
| "temStaPy not found" | Check sys.path.insert() line |
| PWF not monotonic | Reduce parameter extremes |
| Very slow | Try smaller u_grid (50 points) |
| Plots look wrong | Verify CGMY→NTS conversion |

---

**End of Documentation**

*For questions or issues, contact: Akash Deep (akash.deep@ttu.edu)*
