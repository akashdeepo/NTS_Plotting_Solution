# Probability Weighting Functions for Normal Tempered-Stable Distributions

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Academic](https://img.shields.io/badge/license-Academic-green.svg)](LICENSE)

Implementation of probability weighting functions (PWFs) for Normal Tempered-Stable (NTS) and Rapidly Decreasing Tempered-Stable (RDTS) distributions, with application to behavioral finance and greed-fear analysis.

## Overview

This repository implements the computational framework for analyzing probability weighting functions through the lens of tempered stable distributions. The key innovation is using information-theoretic indices to quantify greed and fear in decision-making under uncertainty.

### Key Features

- **NTS Distribution Framework**: Complete implementation using temStaPy library
- **Information-Theoretic Indices**: Three novel greed-fear measures (G_FI, S_JS, E)
- **Reproducible Research**: All figures and analyses fully documented
- **Clean Codebase**: Well-documented, modular Python code

## Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
pip install numpy scipy matplotlib
```

### Installation

```bash
git clone https://github.com/yourusername/PWFs-for-NTS.git
cd PWFs-for-NTS
```

The `temStaPy` library is included in the repository (`lib/temStaPy_v0.5/`).

### Running the Code

**Generate Figure 2(e) - Information-Theoretic Indices:**

```bash
python generate_figure_2e_information_theoretic.py
```

This will create a dated folder (e.g., `figures_2025-10-14_corrected_formulas/`) with:
- 4 figure variants (PNG + PDF)
- Comprehensive README documentation

**Output location:** `figures_YYYY-MM-DD_corrected_formulas/`

## Repository Structure

```
PWFs-for-NTS/
├── README.md                                    # This file
├── CHANGELOG_2025-10-14.md                      # Latest changes
├── generate_figure_2e_information_theoretic.py  # Main script
├── figures_2025-10-14_corrected_formulas/       # Latest output (dated)
│   ├── Figure_2e1_LogitShift_GFI.pdf
│   ├── Figure_2e2_JensenShannon_SJS.pdf
│   ├── Figure_2e3_Elasticity_E.pdf
│   ├── Figure_2e4_Cumulative_Scores.pdf
│   └── README.md                                # Detailed documentation
├── diagnostic_scripts/                          # Verification tools
│   ├── analyze_corrected_plots.py
│   ├── check_sjs_index.py
│   ├── diagnostic_info_theoretic.py
│   └── verify_corrected_formula.py
└── lib/                                         # Dependencies
    └── temStaPy_v0.5/                           # NTS distribution library
```

## Mathematical Framework

### Information-Theoretic Greed-Fear Indices

The code implements three indices to measure behavioral biases:

#### 1. Logit-Shift Index (G_FI)

```
G_FI(p) = [logit(w(p)) - logit(p)] / √[p(1-p)]
```

Measures log-odds distortion normalized by Fisher information.

#### 2. Signed Jensen-Shannon Index (S_JS)

```
JSD(p) = ½[KL(w||m) + KL(p||m)]  where m = (w+p)/2
S_JS(p) = sgn(w(p) - p) · √(2·JSD(p))
```

Smooth, bounded measure based on information divergence.

#### 3. Log-Odds Elasticity (E)

```
E(p) = d/dp[logit(w(p)) - logit(p)]
     = w'(p)/[w(p)(1-w(p))] - 1/[p(1-p)]
```

Derivative measure showing rate of probability distortion.

### Probability Weighting Function (PWF)

```
w(u) = F_post(Q_prior(u))
```

Where:
- `Q_prior(u)` = Quantile function of prior (benchmark) distribution
- `F_post(x)` = CDF of posterior (greedy/fearful) distribution

## Code Usage Examples

### Basic Usage

```python
import numpy as np
import sys
sys.path.insert(0, r'lib/temStaPy_v0.5')
from temStaPy.distNTS import qnts, pnts

# Define NTS parameters [alpha, theta, beta, gamma, mu]
nts_params = [1.2, 3.6, 0.0, 1.0, 0.0]

# Compute PWF on grid
p_grid = np.linspace(0.01, 0.99, 100)
q_prior = qnts(p_grid, nts_params)
w = pnts(q_prior, nts_params)
```

### Computing Greed-Fear Indices

```python
# Logit-Shift Index
def logit_shift_GFI(w, p, epsilon=1e-6):
    w_clip = np.clip(w, epsilon, 1-epsilon)
    p_clip = np.clip(p, epsilon, 1-epsilon)

    logit_w = np.log(w_clip / (1 - w_clip))
    logit_p = np.log(p_clip / (1 - p_clip))

    G = logit_w - logit_p
    G_FI = G / np.sqrt(p_clip * (1 - p_clip))

    return G_FI
```

See `generate_figure_2e_information_theoretic.py` for complete implementations.

## Parameters

### Case 1: Volatility Channel (σ variation)

```python
# Benchmark (symmetric, moderate volatility)
C, Y, G, M, sigma = 0.6, 1.2, 5.0, 5.0, 1.0

# Fearful (higher volatility)
sigma_fearful = 1.4

# Greedy (lower volatility)
sigma_greedy = 0.7
```

### Case 2: Skewness Channel (G/M variation) - CORRECTED

```python
# Benchmark (symmetric)
C, Y, G, M, sigma = 0.6, 1.2, 5.0, 5.0, 1.0

# Fearful (left-skewed)
G_fearful, M_fearful, sigma_fearful = 3.0, 9.0, 0.92

# Greedy (right-skewed)
G_greedy, M_greedy, sigma_greedy = 9.0, 3.0, 0.92
```

**Note:** The σ=0.92 parameter for Case 2 was corrected on October 13, 2025 per Dr. Rachev's specifications.

## Figures Generated

All figures are saved as both PNG (300 DPI) and PDF (vector) formats:

| Figure | Description | Key Insight |
|--------|-------------|-------------|
| 2(e1) | Logit-Shift Index G_FI(p) | Shows log-odds distortion with opposite signs for greed/fear |
| 2(e2) | Jensen-Shannon Index S_JS(p) | Smooth bounded measure of information divergence |
| 2(e3) | Elasticity E(p) | Rate of change of probability weighting |
| 2(e4) | Cumulative Scores | Integrated greed-fear effects across probability range |

## Verification

Run diagnostic scripts to verify correctness:

```bash
# Verify formulas match specifications
python diagnostic_scripts/verify_corrected_formula.py

# Analyze plot behavior
python diagnostic_scripts/analyze_corrected_plots.py

# Check Jensen-Shannon index
python diagnostic_scripts/check_sjs_index.py
```

All scripts include detailed output explaining expected vs. actual behavior.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{deep2025pwf,
  title={Probability Weighting Functions for Normal Tempered-Stable Distributions},
  author={Deep, Akash and Trindade, Alex and Lindquist, W. Brent and Rachev, S. T. and Fabozzi, Frank},
  journal={In preparation},
  year={2025}
}
```

## Authors

- **Akash Deep** - PhD Student, Texas Tech University
- **Alex Trindade** - Texas Tech University
- **W. Brent Lindquist** - Texas Tech University
- **Svetlozar T. Rachev** - Texas Tech University
- **Frank Fabozzi** - EDHEC Business School

## License

This code is provided for academic research purposes. Please contact the authors for commercial use.

## Changelog

See [CHANGELOG_2025-10-14.md](CHANGELOG_2025-10-14.md) for recent updates.

### Recent Changes (2025-10-14)

- **CRITICAL FIX**: Corrected G_FI formula to use √[p(1-p)] normalization
- Verified all formulas against specifications
- Reorganized project structure with dated output folders
- Added comprehensive documentation

## Contact

For questions or collaboration:
- Akash Deep: akash.deep@ttu.edu
- Repository: https://github.com/yourusername/PWFs-for-NTS

## Acknowledgments

- **temStaPy Library**: Used for NTS distribution computations
- **Dr. Svetlozar Rachev**: Specifications and theoretical framework
- **Texas Tech University**: Institutional support

---

**Last Updated:** October 14, 2025
**Status:** Ready for review and publication
