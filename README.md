# Composite Beam Buckling Analysis – M3 (Tkinter GUI)

A professional GUI tool for **local buckling analysis** of a **double-tapered composite box beam** using a Ritz (Rayleigh–Ritz) formulation.

This tool supports:
- Single-case buckling
- One-at-a-time sensitivity (OAT)
- Global uncertainty quantification (Sobol UQ)

---

## Key Features
- **M3 (Accurate)**: 2-term Ritz model with **root rotational spring** boundary condition
- **M2 (Fast)**: 1-term Ritz model for rapid evaluation (approximation)
- Load–deflection curves (Linear / Nonlinear / Total) + mode contour
- OAT Sensitivity: parameter sweep around baseline (±%)
- Sobol UQ: variance-based global sensitivity (S1 / ST)
- Export: plots (PNG/PDF), results text, config (JSON)

---

## Requirements
- Python **3.8+**
- pip
- (Optional) Pillow (PIL) is recommended for high-quality image scaling in the GUI.

---

## Installation

1) Clone or download this repository.

2) (Recommended) Create and activate a virtual environment:

Mac/Linux:
- Create: `python3 -m venv venv`
- Activate: `source venv/bin/activate`

Windows:
- Create: `python -m venv venv`
- Activate: `venv\Scripts\activate`

3) Install dependencies:
- `pip install -r requirements.txt`

---

## Run

Option A) Run Script (Mac/Linux)
- Grant permission (first time): `chmod +x run_gui.sh`
- Run: `./run_gui.sh`

Option B) Direct Python Run
- Run: `python3 gui_buckling.py`

---

## GUI Overview (3 Tabs)

### 1) Buckling (Single Case)
Single design point evaluation.

Inputs:
- Geometry, laminates, material properties, boundary conditions, solver settings

Outputs:
- Critical buckling load: **Pcr**
- Critical deflection: **dcr** (when mode/curve is available)
- Mode parameters: **alpha\***, **beta\***, **lambda_x\***
- Load–deflection curves: Linear / Nonlinear / Total
- Mode contour: buckling mode shape visualization

### 2) Sensitivity (OAT)
One-at-a-time sweep for selected parameters.

How to use:
- Select parameters (checkbox)
- Set perturbation range (±%) and number of points (Pts)
- Click Run

Output:
- Grid plots of **Pcr vs parameter value**, baseline marked

### 3) Uncertainty (Sobol UQ)
Variance-based global sensitivity analysis.

How to use:
- Select uncertain parameters and define Low/High bounds
- Set N_base (total evaluations approx. N_base × (k + 2))
- Click Run

Outputs:
- S1 (first-order Sobol index)
- ST (total-order Sobol index)
- Bar chart sorted by importance

---

## Export / Config
- Save Config (JSON): save current baseline input
- Load Config (JSON): load a saved baseline input
- Export Plot: save plots as PNG/PDF
- Export Results: save summary text as TXT
