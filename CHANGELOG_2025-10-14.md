# Changelog - October 14, 2025

## Critical Formula Correction and Clean Reorganization

---

## Summary

Fixed a critical error in the G_FI formula implementation and reorganized the project structure for better version tracking.

---

## Changes Made

### 1. **CRITICAL FIX: G_FI Formula Correction**

**Issue:** The logit-shift greed-fear index was incorrectly implemented
- ❌ **Incorrect:** `G_FI = G(p) / [p(1-p)]`
- ✅ **Corrected:** `G_FI = G(p) / √[p(1-p)]`

**How found:** Akash provided screenshots from Dr. Rachev's Word document showing the exact specifications. Comparison revealed the missing square root in the denominator.

**Impact:**
- G_FI values reduced by ~10x (from ±280 to ±28)
- Now matches Dr. Rachev's Equation (2) exactly
- More reasonable magnitudes while maintaining clear separation

**File modified:** `generate_figure_2e_information_theoretic.py` (line 51)

---

### 2. **Project Reorganization**

**Created new dated folder:**
```
figures_2025-10-14_corrected_formulas/
├── Figure_2e1_LogitShift_GFI.png
├── Figure_2e1_LogitShift_GFI.pdf
├── Figure_2e2_JensenShannon_SJS.png
├── Figure_2e2_JensenShannon_SJS.pdf
├── Figure_2e3_Elasticity_E.png
├── Figure_2e3_Elasticity_E.pdf
├── Figure_2e4_Cumulative_Scores.png
├── Figure_2e4_Cumulative_Scores.pdf
└── README.md (comprehensive documentation)
```

**Moved diagnostic scripts:**
```
diagnostic_scripts/
├── analyze_corrected_plots.py
├── check_sjs_index.py
├── diagnostic_info_theoretic.py
└── verify_corrected_formula.py
```

**Benefits:**
- Date-stamped folders track version history
- Diagnostic scripts separated from main code
- README documents all specifications and verification results
- Easy to compare versions if needed

---

### 3. **Updated Main Script**

**File:** `generate_figure_2e_information_theoretic.py`

**Changes:**
- Fixed G_FI formula (line 51): Added `np.sqrt()` to denominator
- Auto-generates dated output folder (lines 105-109)
- Maintains all other formulas (S_JS, E) which were already correct

---

## Verification Results

All figures regenerated and verified:

✅ **Figure 2(e1) - G_FI(p):**
- Range: Fearful ±26, Greedy ±46
- Mirror-symmetric with opposite signs
- Formula matches Dr. Rachev's Eq. (2)

✅ **Figure 2(e2) - S_JS(p):**
- Range: Fearful ±0.12, Greedy ±0.13
- Bounded within theoretical limit (±1.18)
- Formula matches Dr. Rachev's Eqs. (3-4)

✅ **Figure 2(e3) - E(p):**
- Derivative/elasticity measure
- Formula matches Dr. Rachev's Eq. (5)

✅ **Figure 2(e4) - Cumulative:**
- Peak magnitudes: G_FI (±1.5), S_JS (±0.04)
- Returns to zero at p=1 as expected

**All pattern checks passed:**
- Fearful: positive at p<0.5, negative at p>0.5 ✓
- Greedy: negative at p<0.5, positive at p>0.5 ✓
- Perfect mirror symmetry ✓
- Much better separation than old Γ(p) index ✓

---

## Files Modified

1. `generate_figure_2e_information_theoretic.py` - Fixed G_FI formula, added auto-dating
2. Created: `figures_2025-10-14_corrected_formulas/README.md` - Full documentation
3. Created: `CHANGELOG_2025-10-14.md` - This file
4. Moved 4 diagnostic scripts to `diagnostic_scripts/` folder

---

## Next Steps

1. **Email Dr. Rachev** with the corrected figures from `figures_2025-10-14_corrected_formulas/`
2. **Attach PDF files** for his review
3. **Wait for confirmation** that these plots show the expected "stronger visual and quantitative separation"
4. **If approved:** These become the final Figure 2(e) variants for the manuscript

---

## Technical Notes

**Formula sources:** Dr. Rachev's blue-text specifications (October 13, 2025)
- Equation (2): G_FI(p) = G(p) / √[p(1-p)]
- Equations (3-4): S_JS(p) via Jensen-Shannon divergence
- Equation (5): E(p) via numerical differentiation

**Implementation details:**
- Grid: 2000 points, uniform spacing
- Clipping: [10^-6, 1-10^-6] for numerical stability
- PWF computation: temStaPy library (FFT-based)
- Derivatives: 2nd-order finite differences (Dr. Rachev suggests 4th-order for final publication)

---

**Date:** October 14, 2025
**Status:** Ready for Dr. Rachev's review
**Critical fix:** G_FI formula corrected ✅
