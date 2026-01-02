# Buckling Analysis GUI - Usage Guide

## Overview
The extended GUI (`gui_buckling_extended.py`) provides three analysis modes in a tabbed interface:
1. **FULL Analysis** - Single design point with detailed results
2. **Sensitivity (SENS)** - One-at-a-time parameter studies
3. **Sobol (UQ)** - Variance-based global sensitivity analysis

---

## Getting Started

### Run the GUI
```bash
python gui_buckling_extended.py
```

**Requirements:**
- Python 3.7+
- `buckling_analysis_M3a.py` in same directory
- Libraries: tkinter, numpy, pandas, matplotlib

---

## TAB 1: FULL Analysis

### Purpose
Analyze a single beam configuration to get:
- Critical buckling load (Pcr)
- Critical deflection (δcr)
- Load-deflection curves (linear + nonlinear)
- Mode shape parameters (α*, β*, λx*)

### Workflow
1. **Set parameters** (left panel):
   - Geometry: L, b_root, b_tip, h_root, h_tip, w_f
   - Laminates: face/web thickness and ply angles
   - Materials: Ef, Em, Gf, νf, νm, Vf
   - Boundary: Ktheta_root_per_m (root spring stiffness)

2. **Choose solver**:
   - **M3**: Two-term Ritz with root spring (accurate, slower)
   - **M2**: One-term clamped (fast approximation)

3. **Click "▶ RUN ANALYSIS"**

4. **View results** (right panel):
   - Text summary with Pcr, δcr, mode parameters
   - Load-deflection plot showing linear, nonlinear, and total curves

5. **Export**:
   - 📊 Export Plot → Save as PNG/PDF
   - 📄 Export Results → Save text summary

### Configuration Management
- **📁 Load**: Load saved parameter sets from JSON
- **💾 Save**: Save current parameters to JSON for reuse

---

## TAB 2: Sensitivity (SENS)

### Purpose
Study how Pcr changes when individual parameters vary (one-at-a-time).
Identifies which parameters have the strongest influence on buckling.

### Workflow
1. **Select parameters** to study:
   - Check boxes next to parameters (e.g., ☑ Ef, ☑ Vf, ☑ L)
   - Set **±%**: Variation range (default 10%)
   - Set **Pts**: Number of sample points (default 5)

2. **Choose solver**: M3 or M2
   - M2 recommended for speed when studying many parameters

3. **Click "▶ RUN SENSITIVITY"**
   - Progress bar shows % complete
   - May take minutes depending on number of evaluations

4. **View results**:
   - Subplots showing Pcr vs each parameter
   - Red dashed line = baseline value

5. **Export**: 📊 Export SENS Plot

### Example
```
Study effect of material uncertainty:
☑ Ef    ±%: 10    Pts: 7   → 7 evaluations
☑ Em    ±%: 15    Pts: 5   → 5 evaluations
☑ Vf    ±%: 5     Pts: 9   → 9 evaluations
Total: 21 evaluations
```

**Interpretation:**
- Steep slope → parameter is influential
- Flat line → parameter has little effect
- Nonlinear curve → complex interaction

---

## TAB 3: Sobol (UQ)

### Purpose
Quantify uncertainty in Pcr due to multiple uncertain parameters simultaneously.
Uses Sobol variance-based indices to rank parameter importance.

### Default Values (Pre-filled)
All parameters have default ranges representing ±10% uncertainty:

| Parameter | Low | High | Notes |
|-----------|-----|------|-------|
| L | 1.35 m | 1.65 m | Length tolerance |
| b_root | 0.072 m | 0.088 m | Width variability |
| b_tip | 0.036 m | 0.044 m | |
| Ef | 207 GPa | 253 GPa | Fiber modulus scatter |
| Em | 3.15 GPa | 3.85 GPa | Matrix properties |
| Vf | 0.54 | 0.66 | Manufacturing control |
| Ktheta_root | 0.9e9 | 1.1e9 | Joint stiffness uncertainty |
| t_face_total | 1.8 mm | 2.2 mm | Thickness tolerance |
| t_web_total | 1.35 mm | 1.65 mm | |

### Workflow
1. **Select uncertain parameters**:
   - Check boxes next to parameters with uncertainty
   - Review/edit **Low** and **High** bounds
   - Default values are ±10% around baseline

2. **Update from baseline** (optional):
   - Modify parameters in FULL tab
   - Enter uncertainty % (e.g., 15)
   - Click **📐 Update from Baseline**
   - Automatically sets Low/High = baseline ± %

3. **Sobol Settings**:
   - **N_base**: Sample size (default 100)
     - Total evaluations = N × (2 + k), where k = # parameters
     - Example: N=100, k=5 → 700 evaluations
   - **Random seed**: For reproducibility (default 1234)

4. **Choose solver**:
   - **M2 (fast)**: Recommended for UQ (can be hundreds of evaluations)
   - **M3 (slow)**: Use only for critical studies (may take hours)

5. **Click "▶ RUN SOBOL"**
   - Progress bar shows % complete
   - **Warning**: Large N_base with M3 can take very long

6. **View results**:
   - Bar chart with two indices per parameter:
     - **S_i** (blue): First-order effect (individual contribution)
     - **S_Ti** (orange): Total effect (including interactions)
   - Parameters sorted by S_Ti (most influential first)

7. **Export**: 📊 Export Sobol Plot

### Sobol Index Interpretation

**S_i (First-order index):**
- Fraction of output variance due to parameter alone
- S_i = 0.5 means parameter causes 50% of variance by itself

**S_Ti (Total index):**
- Fraction including all interactions with other parameters
- S_Ti > S_i indicates strong interactions

**Example Results:**
```
Parameter  S_i    S_Ti   Interpretation
---------  ----   ----   --------------
Ef         0.65   0.72   Dominant (72% of variance)
Vf         0.12   0.18   Moderate (includes interactions)
L          0.03   0.05   Minor influence
Em         0.01   0.02   Negligible
```

**Decision:** Focus quality control on Ef (fiber modulus)

---

## Recommended Analysis Sequence

### 1. Baseline Design (FULL tab)
- Set nominal design parameters
- Run FULL analysis with M3
- Record Pcr, verify it meets requirements
- Save configuration (💾 Save)

### 2. Identify Key Parameters (SENS tab)
- Check 8-10 parameters likely to vary
- Set ±10%, 5-7 points each
- Run with M2 for speed
- Identify 3-5 most influential parameters

### 3. Quantify Uncertainty (SOBOL tab)
- Check the 3-5 key parameters from SENS
- Set realistic Low/High bounds (manufacturing data)
- N_base = 100-200 (compromise speed/accuracy)
- Run with M2
- Report S_Ti values for key parameters

### 4. Design Iterations (FULL tab)
- Adjust nominal values of influential parameters
- Re-run FULL analysis
- Compare new Pcr to baseline
- Iterate until requirements satisfied

---

## Performance Tips

### Speed Optimization
1. **Use M2 for SENS/SOBOL**: 2-3× faster than M3
2. **Reduce nx_min**: Default 1801 → try 901 for screening
3. **Start with small N_base**: Try N=50 first, increase if needed
4. **Limit UQ parameters**: Focus on top 3-5 from SENS

### Accuracy vs Speed Trade-off

| Analysis | Recommended Core | Typical Time |
|----------|------------------|--------------|
| FULL (single) | M3 | 5-15 sec |
| SENS (5 params × 5 pts) | M2 | 1-2 min |
| SOBOL (5 params, N=100) | M2 | 5-10 min |
| SOBOL (5 params, N=100) | M3 | 30-60 min |

---

## Troubleshooting

### GUI won't start
- Check `buckling_analysis_M3a.py` is in same folder
- Verify matplotlib backend: `python -c "import matplotlib; print(matplotlib.get_backend())"`

### Analysis fails
- Check input validation: Vf must be 0 < Vf < 1
- Verify positive values: L, widths, thicknesses > 0
- Review ply angles format: "0,45,-45,90" (comma-separated)

### Sobol "low >= high" error
- Ensure Low < High for all checked parameters
- Use "📐 Update from Baseline" button to auto-populate correctly

### Out of memory
- Reduce N_base (try N=50)
- Reduce nx_min (default 1801 → 901)
- Close other applications

---

## Example Use Case: Carbon Fiber Beam

### Scenario
Design a tapered carbon/epoxy beam with uncertain material properties.

### Step 1: FULL Analysis
```
Geometry:
  L = 1.5 m
  b_root = 0.08 m → b_tip = 0.04 m

Materials (nominal):
  Ef = 230 GPa  (±10% uncertainty expected)
  Em = 3.5 GPa
  Vf = 0.6      (±5% manufacturing control)

Boundary:
  Ktheta_root = 1e9 N·m/m (bolted joint)

Result: Pcr = 4523 N
```

### Step 2: SENS
```
Check: Ef, Em, Vf, L, t_face_total
Result: Ef has steepest slope → most influential
```

### Step 3: SOBOL
```
Uncertain parameters:
  Ef:  207 GPa to 253 GPa  (supplier data range)
  Vf:  0.57 to 0.63        (manufacturing capability)

N_base = 150, Solver = M2

Results:
  Ef:  S_Ti = 0.78  → Dominates uncertainty
  Vf:  S_Ti = 0.15  → Secondary

Conclusion:
- Pcr uncertainty driven by fiber modulus variation
- Recommend tighter Ef specification or acceptance testing
```

---

## Keyboard Shortcuts

- **Tab**: Navigate between input fields
- **Enter**: (in some fields) Trigger default action
- **Ctrl+S**: (in text areas) Focus save button
- **Mouse wheel**: Scroll parameter lists

---

## File Formats

### Configuration JSON
```json
{
  "L": 1.5,
  "b_root": 0.08,
  "Ef": 230000000000.0,
  "core": "m3",
  ...
}
```

### Results Text Export
```
==================================================
  BUCKLING ANALYSIS RESULTS
==================================================

Solver: M3
--------------------------------------------------

Critical Load:
  Pcr = 4523.45 N
      = 4.523 kN
...
```

---

## Support

**Common Issues:**
- Parameter validation errors → Check units (Pa not GPa, m not mm)
- Long run times → Use M2 core, reduce N_base
- Plot not updating → Check solver finished (status = "Complete ✓")

**For bugs or feature requests:**
Review the source code comments in `gui_buckling_extended.py`
