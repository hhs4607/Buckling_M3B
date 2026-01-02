#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tkinter GUI for Buckling Analysis M3 - Extended Version
--------------------------------------------------------
Includes FULL, SENS (sensitivity), and SOBOL (UQ) analysis modes
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from pathlib import Path
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import the analysis functions
from buckling_analysis_M3a import (
    eval_m2_Pcr,
    eval_m3_Pcr_and_mode,
    koiter_curves_from_mode,
)


class BucklingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Composite Beam Buckling Analysis - M3 Extended")
        self.root.geometry("1400x850")

        # Storage for results
        self.current_results = None
        self.is_running = False

        # Create main layout with notebook (tabs)
        self.create_layout()
        self.load_defaults()

    def create_layout(self):
        """Create the main GUI layout with tabs"""

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Tab 1: FULL Analysis (original)
        self.full_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(self.full_frame, text="  FULL Analysis  ")

        # Tab 2: SENS Analysis
        self.sens_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(self.sens_frame, text="  Sensitivity (SENS)  ")

        # Tab 3: SOBOL Analysis
        self.sobol_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(self.sobol_frame, text="  Sobol (UQ)  ")

        # Create each tab's content
        self.create_full_tab()
        self.create_sens_tab()
        self.create_sobol_tab()

    # ========== TAB 1: FULL ANALYSIS ==========
    def create_full_tab(self):
        """Create FULL analysis tab (original functionality)"""

        # Split into left (inputs) and right (results)
        self.full_frame.columnconfigure(1, weight=1)
        self.full_frame.rowconfigure(0, weight=1)

        # Left panel - Inputs (scrollable)
        left_frame = ttk.Frame(self.full_frame, padding="5")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        canvas = tk.Canvas(left_frame, width=450)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Right panel - Results
        right_frame = ttk.Frame(self.full_frame, padding="5")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create input sections
        self.create_input_sections()

        # Create results section
        self.create_results_section(right_frame)

        # Mousewheel
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

    def create_input_sections(self):
        """Create all input parameter sections"""
        self.entries = {}
        row = 0

        # GEOMETRY
        row = self.create_section_header("GEOMETRY", row)
        params = [
            ("L", "Beam Length (m)", 1.5),
            ("b_root", "Root Width (m)", 0.08),
            ("b_tip", "Tip Width (m)", 0.04),
            ("h_root", "Root Core Height (m)", 0.025),
            ("h_tip", "Tip Core Height (m)", 0.015),
            ("w_f", "Flange Width (m)", 0.02),
        ]
        for key, label, default in params:
            row = self.create_input_row(key, label, default, row)

        # FACE LAMINATE
        row = self.create_section_header("FACE LAMINATE", row)
        params = [
            ("t_face_total", "Total Face Thickness (m)", 0.002),
            ("face_angles", "Face Ply Angles (deg)", "0,45,-45,90"),
        ]
        for key, label, default in params:
            row = self.create_input_row(key, label, default, row)

        # WEB LAMINATE
        row = self.create_section_header("WEB LAMINATE", row)
        params = [
            ("t_web_total", "Total Web Thickness (m)", 0.0015),
            ("web_angles", "Web Ply Angles (deg)", "0,90"),
        ]
        for key, label, default in params:
            row = self.create_input_row(key, label, default, row)

        # MATERIAL PROPERTIES
        row = self.create_section_header("MATERIAL PROPERTIES", row)
        params = [
            ("Ef", "Fiber Modulus (Pa)", 230e9),
            ("Em", "Matrix Modulus (Pa)", 3.5e9),
            ("Gf", "Fiber Shear Modulus (Pa)", 90e9),
            ("nuf", "Fiber Poisson's Ratio", 0.2),
            ("num", "Matrix Poisson's Ratio", 0.35),
            ("Vf", "Fiber Volume Fraction", 0.6),
        ]
        for key, label, default in params:
            row = self.create_input_row(key, label, default, row)

        # BOUNDARY CONDITIONS
        row = self.create_section_header("BOUNDARY CONDITIONS", row)
        params = [("Ktheta_root_per_m", "Root Spring Stiffness (N·m/m)", 1e9)]
        for key, label, default in params:
            row = self.create_input_row(key, label, default, row)

        # ANALYSIS SETTINGS
        row = self.create_section_header("ANALYSIS SETTINGS", row)
        core_frame = ttk.Frame(self.scrollable_frame)
        core_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5, padx=10)
        ttk.Label(core_frame, text="Solver Core:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        self.core_var = tk.StringVar(value="m3")
        ttk.Radiobutton(core_frame, text="M3 (Two-term)", variable=self.core_var, value="m3").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(core_frame, text="M2 (Fast)", variable=self.core_var, value="m2").pack(side=tk.LEFT, padx=5)
        row += 1

        params = [
            ("PPW", "Points Per Wavelength", 60),
            ("nx_min", "Minimum Grid Points", 1801),
        ]
        for key, label, default in params:
            row = self.create_input_row(key, label, default, row)

        # ACTION BUTTONS
        row += 1
        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=20)
        self.run_button = ttk.Button(button_frame, text="▶ RUN ANALYSIS", command=self.run_full_analysis, width=20)
        self.run_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📁 Load", command=self.load_config, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="💾 Save", command=self.save_config, width=12).pack(side=tk.LEFT, padx=5)

        # Progress
        row += 1
        self.progress = ttk.Progressbar(self.scrollable_frame, mode='indeterminate')
        self.progress.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=10, pady=10)
        row += 1
        self.status_label = ttk.Label(self.scrollable_frame, text="Ready", foreground="green")
        self.status_label.grid(row=row, column=0, columnspan=2, pady=5)

    def create_section_header(self, title, row):
        frame = ttk.Frame(self.scrollable_frame)
        frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(15, 5), padx=5)
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=(0, 5))
        ttk.Label(frame, text=title, font=('Arial', 10, 'bold')).pack(anchor='w')
        return row + 1

    def create_input_row(self, key, label, default, row):
        ttk.Label(self.scrollable_frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=10, pady=3)
        entry = ttk.Entry(self.scrollable_frame, width=25)
        entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=10, pady=3)
        entry.insert(0, str(default))
        self.entries[key] = entry
        return row + 1

    def create_results_section(self, parent):
        """Results display for FULL tab"""
        title_label = ttk.Label(parent, text="RESULTS", font=('Arial', 12, 'bold'))
        title_label.pack(pady=(0, 10))

        text_frame = ttk.LabelFrame(parent, text="Analysis Summary", padding="10")
        text_frame.pack(fill=tk.BOTH, pady=(0, 10))
        self.results_text = tk.Text(text_frame, height=10, width=50, font=('Courier', 9))
        self.results_text.pack(fill=tk.BOTH, expand=True)

        plot_frame = ttk.LabelFrame(parent, text="Load-Deflection Curve", padding="10")
        plot_frame.pack(fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Tip Deflection δ [m]")
        self.ax.set_ylabel("Load P [N]")
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.text(0.5, 0.5, 'Run analysis to see results', ha='center', va='center',
                    transform=self.ax.transAxes, fontsize=12, alpha=0.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        export_frame = ttk.Frame(parent)
        export_frame.pack(pady=10)
        ttk.Button(export_frame, text="📊 Export Plot", command=self.export_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="📄 Export Results", command=self.export_results).pack(side=tk.LEFT, padx=5)

    # ========== TAB 2: SENS ANALYSIS ==========
    def create_sens_tab(self):
        """Create sensitivity analysis tab"""

        # Info label
        info = ttk.Label(self.sens_frame, text="One-At-a-Time (OAT) Sensitivity Analysis",
                        font=('Arial', 11, 'bold'))
        info.pack(pady=10)

        # Split: left controls, right results
        content = ttk.Frame(self.sens_frame)
        content.pack(fill=tk.BOTH, expand=True, padx=10)

        left = ttk.Frame(content)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

        right = ttk.Frame(content)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Left: Parameter selection
        ttk.Label(left, text="Select parameters to vary:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 10))

        # Scrollable parameter list
        canvas_sens = tk.Canvas(left, width=350, height=400)
        scroll_sens = ttk.Scrollbar(left, orient="vertical", command=canvas_sens.yview)
        param_frame = ttk.Frame(canvas_sens)
        param_frame.bind("<Configure>", lambda e: canvas_sens.configure(scrollregion=canvas_sens.bbox("all")))
        canvas_sens.create_window((0, 0), window=param_frame, anchor="nw")
        canvas_sens.configure(yscrollcommand=scroll_sens.set)
        canvas_sens.pack(side="left", fill="both", expand=True)
        scroll_sens.pack(side="right", fill="y")

        # Parameter checkboxes with range inputs
        self.sens_vars = {}
        sens_params = ["L", "b_root", "b_tip", "h_root", "h_tip", "w_f", "t_face_total", "t_web_total",
                       "Ef", "Em", "Gf", "Vf", "Ktheta_root_per_m"]

        for i, param in enumerate(sens_params):
            frame = ttk.Frame(param_frame)
            frame.pack(fill=tk.X, pady=2, padx=5)

            var = tk.BooleanVar(value=False)
            check = ttk.Checkbutton(frame, text=param, variable=var, width=20)
            check.pack(side=tk.LEFT)

            ttk.Label(frame, text="±%:").pack(side=tk.LEFT, padx=(5, 2))
            pct_entry = ttk.Entry(frame, width=8)
            pct_entry.insert(0, "10")
            pct_entry.pack(side=tk.LEFT, padx=2)

            ttk.Label(frame, text="Pts:").pack(side=tk.LEFT, padx=(5, 2))
            pts_entry = ttk.Entry(frame, width=6)
            pts_entry.insert(0, "5")
            pts_entry.pack(side=tk.LEFT)

            self.sens_vars[param] = {"enabled": var, "pct": pct_entry, "pts": pts_entry}

        # Solver selection
        solver_frame = ttk.LabelFrame(left, text="Solver", padding="10")
        solver_frame.pack(fill=tk.X, pady=10)
        self.sens_core_var = tk.StringVar(value="m3")
        ttk.Radiobutton(solver_frame, text="M3", variable=self.sens_core_var, value="m3").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(solver_frame, text="M2", variable=self.sens_core_var, value="m2").pack(side=tk.LEFT, padx=10)

        # Run button
        ttk.Button(left, text="▶ RUN SENSITIVITY", command=self.run_sens_analysis, width=25).pack(pady=10)

        # Progress
        self.sens_progress = ttk.Progressbar(left, mode='determinate')
        self.sens_progress.pack(fill=tk.X, padx=10, pady=5)
        self.sens_status = ttk.Label(left, text="Ready", foreground="green")
        self.sens_status.pack(pady=5)

        # Right: Results plot
        ttk.Label(right, text="Sensitivity Results", font=('Arial', 11, 'bold')).pack(pady=(0, 10))

        self.sens_fig = Figure(figsize=(7, 6), dpi=100)
        self.sens_canvas = FigureCanvasTkAgg(self.sens_fig, master=right)
        self.sens_canvas.draw()
        self.sens_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ttk.Button(right, text="📊 Export SENS Plot", command=self.export_sens_plot).pack(pady=5)

    # ========== TAB 3: SOBOL ANALYSIS ==========
    def create_sobol_tab(self):
        """Create Sobol/UQ analysis tab"""

        info = ttk.Label(self.sobol_frame, text="Sobol Variance-Based Sensitivity (UQ)",
                        font=('Arial', 11, 'bold'))
        info.pack(pady=10)

        # Split layout
        content = ttk.Frame(self.sobol_frame)
        content.pack(fill=tk.BOTH, expand=True, padx=10)

        left = ttk.Frame(content)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

        right = ttk.Frame(content)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Left: UQ parameter setup
        ttk.Label(left, text="Uncertain Parameters:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 10))

        # Scrollable
        canvas_sobol = tk.Canvas(left, width=380, height=350)
        scroll_sobol = ttk.Scrollbar(left, orient="vertical", command=canvas_sobol.yview)
        uq_frame = ttk.Frame(canvas_sobol)
        uq_frame.bind("<Configure>", lambda e: canvas_sobol.configure(scrollregion=canvas_sobol.bbox("all")))
        canvas_sobol.create_window((0, 0), window=uq_frame, anchor="nw")
        canvas_sobol.configure(yscrollcommand=scroll_sobol.set)
        canvas_sobol.pack(side="left", fill="both", expand=True)
        scroll_sobol.pack(side="right", fill="y")

        # UQ parameter inputs with default ranges (±10% typical uncertainty)
        self.sobol_vars = {}

        # Define default low/high values based on baseline ± typical uncertainty
        sobol_defaults = {
            "L": (1.35, 1.65),                    # ±10% around 1.5 m
            "b_root": (0.072, 0.088),             # ±10% around 0.08 m
            "b_tip": (0.036, 0.044),              # ±10% around 0.04 m
            "Ef": (207e9, 253e9),                 # ±10% around 230e9 Pa
            "Em": (3.15e9, 3.85e9),               # ±10% around 3.5e9 Pa
            "Vf": (0.54, 0.66),                   # ±10% around 0.6
            "Ktheta_root_per_m": (0.9e9, 1.1e9), # ±10% around 1e9 N·m/m
            "t_face_total": (0.0018, 0.0022),     # ±10% around 0.002 m
            "t_web_total": (0.00135, 0.00165),    # ±10% around 0.0015 m
        }

        for param, (low_default, high_default) in sobol_defaults.items():
            frame = ttk.Frame(uq_frame)
            frame.pack(fill=tk.X, pady=3, padx=5)

            var = tk.BooleanVar(value=False)
            check = ttk.Checkbutton(frame, text=param, variable=var, width=18)
            check.pack(side=tk.LEFT)

            ttk.Label(frame, text="Low:").pack(side=tk.LEFT, padx=(5, 2))
            low_entry = ttk.Entry(frame, width=10)
            low_entry.insert(0, f"{low_default:.4g}")
            low_entry.pack(side=tk.LEFT, padx=2)

            ttk.Label(frame, text="High:").pack(side=tk.LEFT, padx=(5, 2))
            high_entry = ttk.Entry(frame, width=10)
            high_entry.insert(0, f"{high_default:.4g}")
            high_entry.pack(side=tk.LEFT)

            self.sobol_vars[param] = {"enabled": var, "low": low_entry, "high": high_entry}

        # Control settings
        control_frame = ttk.LabelFrame(left, text="Sobol Settings", padding="10")
        control_frame.pack(fill=tk.X, pady=10)

        ttk.Label(control_frame, text="N_base (sample size):").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.sobol_N = ttk.Entry(control_frame, width=12)
        self.sobol_N.insert(0, "100")
        self.sobol_N.grid(row=0, column=1, pady=3, padx=5)

        ttk.Label(control_frame, text="Random seed:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.sobol_seed = ttk.Entry(control_frame, width=12)
        self.sobol_seed.insert(0, "1234")
        self.sobol_seed.grid(row=1, column=1, pady=3, padx=5)

        ttk.Label(control_frame, text="Uncertainty:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.sobol_pct = ttk.Entry(control_frame, width=12)
        self.sobol_pct.insert(0, "10")
        self.sobol_pct.grid(row=2, column=1, pady=3, padx=5)
        ttk.Label(control_frame, text="% (for Update button)").grid(row=2, column=2, sticky=tk.W, pady=3)

        # Helper button to update ranges from current baseline
        ttk.Button(control_frame, text="📐 Update from Baseline",
                  command=self.update_sobol_from_baseline, width=25).grid(row=3, column=0, columnspan=3, pady=5)

        # Solver
        solver_frame = ttk.LabelFrame(left, text="Solver", padding="10")
        solver_frame.pack(fill=tk.X, pady=10)
        self.sobol_core_var = tk.StringVar(value="m2")
        ttk.Radiobutton(solver_frame, text="M3 (slow)", variable=self.sobol_core_var, value="m3").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(solver_frame, text="M2 (fast)", variable=self.sobol_core_var, value="m2").pack(side=tk.LEFT, padx=10)

        # Run
        ttk.Button(left, text="▶ RUN SOBOL", command=self.run_sobol_analysis, width=25).pack(pady=10)

        # Progress
        self.sobol_progress = ttk.Progressbar(left, mode='determinate')
        self.sobol_progress.pack(fill=tk.X, padx=10, pady=5)
        self.sobol_status = ttk.Label(left, text="Ready", foreground="green")
        self.sobol_status.pack(pady=5)

        # Right: Sobol plot
        ttk.Label(right, text="Sobol Indices", font=('Arial', 11, 'bold')).pack(pady=(0, 10))

        self.sobol_fig = Figure(figsize=(7, 6), dpi=100)
        self.sobol_canvas = FigureCanvasTkAgg(self.sobol_fig, master=right)
        self.sobol_canvas.draw()
        self.sobol_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ttk.Button(right, text="📊 Export Sobol Plot", command=self.export_sobol_plot).pack(pady=5)

    # ========== HELPER FUNCTIONS ==========
    def get_values(self):
        """Extract values from entries"""
        vals = {}
        numeric_keys = ["L", "b_root", "b_tip", "h_root", "h_tip", "w_f",
                       "t_face_total", "t_web_total", "Ef", "Em", "Gf",
                       "nuf", "num", "Vf", "Ktheta_root_per_m", "PPW", "nx_min"]
        try:
            for key in numeric_keys:
                vals[key] = float(self.entries[key].get().strip())
            vals["face_angles"] = self.entries["face_angles"].get().strip()
            vals["web_angles"] = self.entries["web_angles"].get().strip()
            return vals
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid numeric value: {e}")
            return None

    def validate_inputs(self, vals):
        """Validate inputs"""
        errors = []
        if vals["L"] <= 0: errors.append("Length must be positive")
        if not (0 < vals["Vf"] < 1): errors.append("Vf must be 0 < Vf < 1")
        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors))
            return False
        return True

    # ========== FULL ANALYSIS ==========
    def run_full_analysis(self):
        """Run FULL mode analysis"""
        if self.is_running:
            return
        vals = self.get_values()
        if vals is None or not self.validate_inputs(vals):
            return

        self.is_running = True
        self.run_button.config(state='disabled')
        self.progress.start(10)
        self.status_label.config(text="Running...", foreground="orange")

        thread = threading.Thread(target=self.full_worker, args=(vals,))
        thread.start()

    def full_worker(self, vals):
        """Worker for FULL analysis"""
        try:
            core = self.core_var.get()
            PPW = int(vals["PPW"])
            nx_min = int(vals["nx_min"])

            if core == "m2":
                Pcr = eval_m2_Pcr(vals, PPW=PPW, nx_min=nx_min)
                results = {"core": "M2", "Pcr": Pcr, "mode_available": False}
            else:
                mode = eval_m3_Pcr_and_mode(vals, PPW=PPW, nx_min=nx_min, return_mode=True)
                P, dlin, dloc, dtot, dcr, more = koiter_curves_from_mode(mode)
                results = {
                    "core": "M3", "Pcr": mode["Pcr"], "dcr": dcr,
                    "alpha_star": mode["alpha_star"], "beta_star": mode["beta_star"],
                    "lambda_x": mode["lambda_x"], "mode_available": True,
                    "P": P, "dlin": dlin, "dloc": dloc, "dtot": dtot, "mode": mode
                }
            self.root.after(0, self.full_complete, results)
        except Exception as e:
            self.root.after(0, self.full_error, str(e))

    def full_complete(self, results):
        """FULL analysis complete"""
        self.is_running = False
        self.run_button.config(state='normal')
        self.progress.stop()
        self.status_label.config(text="Complete ✓", foreground="green")
        self.current_results = results
        self.display_full_results(results)
        if results["mode_available"]:
            self.plot_full_results(results)

    def full_error(self, msg):
        """FULL analysis error"""
        self.is_running = False
        self.run_button.config(state='normal')
        self.progress.stop()
        self.status_label.config(text="Failed ✗", foreground="red")
        messagebox.showerror("Error", f"Analysis failed:\n{msg}")

    def display_full_results(self, results):
        """Display FULL results"""
        self.results_text.delete(1.0, tk.END)
        text = f"{'='*50}\n  BUCKLING ANALYSIS RESULTS\n{'='*50}\n\n"
        text += f"Solver: {results['core']}\n{'-'*50}\n\n"
        text += f"Critical Load:\n  Pcr = {results['Pcr']:.2f} N = {results['Pcr']/1000:.3f} kN\n\n"
        if results["mode_available"]:
            text += f"Critical Deflection:\n  δcr = {results['dcr']*1000:.4f} mm\n\n"
            text += f"Mode Parameters:\n"
            text += f"  α* = {results['alpha_star']:.4f} 1/m\n"
            text += f"  β* = {results['beta_star']:.4f} 1/m\n"
            text += f"  λx* = {results['lambda_x']:.4f} m\n"
        self.results_text.insert(1.0, text)

    def plot_full_results(self, results):
        """Plot FULL results"""
        self.ax.clear()
        P = results["P"]; dlin = results["dlin"]; dloc = results["dloc"]; dtot = results["dtot"]
        Pcr = results["Pcr"]; dcr = results["dcr"]
        dmax = 1.5 * dcr; Pmax = 1.5 * Pcr

        def clip(P_arr, d_arr):
            m = (P_arr <= Pmax) & (d_arr <= dmax + 1e-15)
            return d_arr[m], P_arr[m]

        dloc_c, P_loc_c = clip(P, dloc)
        dtot_c, P_tot_c = clip(P, dtot)
        fb = float(np.dot(P, dlin) / max(np.dot(P, P), 1e-18))
        d_lin = np.linspace(0.0, dmax, 400)
        P_lin = d_lin / max(fb, 1e-18)

        self.ax.plot(d_lin, P_lin, '--', lw=2, label='Linear', color='blue')
        self.ax.plot(dloc_c, P_loc_c, ':', lw=2.2, alpha=0.95, label='Nonlinear', color='orange')
        self.ax.plot(dtot_c, P_tot_c, '-', lw=2.4, label='Total', color='green')

        d_at_Pcr = float(np.interp(Pcr, P_tot_c if len(P_tot_c) > 2 else P, dtot_c if len(dtot_c) > 2 else dtot))
        self.ax.scatter([d_at_Pcr], [Pcr], s=70, ec='k', zorder=4, color='red', label=f'Pcr={Pcr:.0f}N')

        self.ax.set_xlabel("Tip Deflection δ [m]"); self.ax.set_ylabel("Load P [N]")
        self.ax.set_xlim(0, dmax); self.ax.set_ylim(0, Pmax)
        self.ax.grid(True, ls='--', alpha=0.3); self.ax.legend(loc='lower right')
        self.ax.set_title("Load-Deflection"); self.fig.tight_layout(); self.canvas.draw()

    # ========== SENS ANALYSIS ==========
    def run_sens_analysis(self):
        """Run sensitivity analysis"""
        if self.is_running:
            return

        vals = self.get_values()
        if vals is None or not self.validate_inputs(vals):
            return

        # Get enabled parameters
        enabled = [(k, v) for k, v in self.sens_vars.items() if v["enabled"].get()]
        if not enabled:
            messagebox.showwarning("Warning", "Select at least one parameter")
            return

        self.is_running = True
        self.sens_status.config(text="Running...", foreground="orange")

        thread = threading.Thread(target=self.sens_worker, args=(vals, enabled))
        thread.start()

    def sens_worker(self, vals0, enabled):
        """SENS worker"""
        try:
            core = self.sens_core_var.get()
            results = []
            total = sum(int(v["pts"].get()) for _, v in enabled)
            count = 0

            for param, widgets in enabled:
                pct = float(widgets["pct"].get()) / 100.0
                n = int(widgets["pts"].get())
                v0 = float(vals0[param])
                grid = np.linspace(v0 * (1 - pct), v0 * (1 + pct), n)

                for val in grid:
                    v = vals0.copy()
                    v[param] = float(val)

                    if core == "m3":
                        Pcr = eval_m3_Pcr_and_mode(v, return_mode=False)
                    else:
                        Pcr = eval_m2_Pcr(v)

                    results.append({"param": param, "value": val, "Pcr": Pcr})
                    count += 1
                    progress = int(100 * count / total)
                    self.root.after(0, lambda p=progress: self.sens_progress.config(value=p))

            self.root.after(0, self.sens_complete, results, vals0, core)
        except Exception as e:
            self.root.after(0, self.sens_error, str(e))

    def sens_complete(self, results, vals0, core):
        """SENS complete"""
        self.is_running = False
        self.sens_status.config(text="Complete ✓", foreground="green")
        self.sens_progress.config(value=100)

        # Plot
        self.sens_fig.clear()
        df = pd.DataFrame(results)
        params = df["param"].unique()

        n = len(params)
        cols = 2
        rows = (n + cols - 1) // cols

        for i, param in enumerate(params):
            ax = self.sens_fig.add_subplot(rows, cols, i + 1)
            data = df[df["param"] == param]
            ax.plot(data["value"], data["Pcr"], 'o-', lw=2)
            ax.set_xlabel(param); ax.set_ylabel("Pcr [N]")
            ax.grid(True, ls='--', alpha=0.3)
            ax.axvline(vals0[param], color='r', ls='--', alpha=0.5, label='Baseline')

        self.sens_fig.suptitle(f"Sensitivity Analysis (core={core})")
        self.sens_fig.tight_layout()
        self.sens_canvas.draw()

        self.sens_results = results  # Store for export

    def sens_error(self, msg):
        """SENS error"""
        self.is_running = False
        self.sens_status.config(text="Failed ✗", foreground="red")
        messagebox.showerror("Error", f"SENS failed:\n{msg}")

    # ========== SOBOL ANALYSIS ==========
    def run_sobol_analysis(self):
        """Run Sobol analysis"""
        if self.is_running:
            return

        vals = self.get_values()
        if vals is None or not self.validate_inputs(vals):
            return

        # Get enabled UQ parameters
        enabled = [(k, v) for k, v in self.sobol_vars.items() if v["enabled"].get()]
        if not enabled:
            messagebox.showwarning("Warning", "Select at least one UQ parameter")
            return

        # Validate ranges
        try:
            uq_params = {}
            for param, widgets in enabled:
                low = float(widgets["low"].get())
                high = float(widgets["high"].get())
                if low >= high:
                    raise ValueError(f"{param}: low must be < high")
                uq_params[param] = {"low": low, "high": high}
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        try:
            N_base = int(self.sobol_N.get())
            seed = int(self.sobol_seed.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid N_base or seed")
            return

        self.is_running = True
        self.sobol_status.config(text="Running...", foreground="orange")

        thread = threading.Thread(target=self.sobol_worker, args=(vals, uq_params, N_base, seed))
        thread.start()

    def sobol_worker(self, vals0, uq_params, N_base, seed):
        """Sobol worker"""
        try:
            core = self.sobol_core_var.get()
            names = list(uq_params.keys())
            k = len(names)

            rng = np.random.default_rng(seed)

            def sample(low, high, N):
                return low + (high - low) * rng.random(N)

            # Saltelli sampling
            A = np.vstack([sample(uq_params[n]["low"], uq_params[n]["high"], N_base) for n in names]).T
            B = np.vstack([sample(uq_params[n]["low"], uq_params[n]["high"], N_base) for n in names]).T

            total_evals = N_base * 2 + N_base * k
            count = 0

            def eval_row(row):
                v = vals0.copy()
                for j, name in enumerate(names):
                    v[name] = float(row[j])
                if core == "m3":
                    return float(eval_m3_Pcr_and_mode(v, return_mode=False))
                else:
                    return float(eval_m2_Pcr(v))

            YA = []
            for n in range(N_base):
                YA.append(eval_row(A[n, :]))
                count += 1
                self.root.after(0, lambda c=count, t=total_evals: self.sobol_progress.config(value=int(100*c/t)))
            YA = np.array(YA)

            YB = []
            for n in range(N_base):
                YB.append(eval_row(B[n, :]))
                count += 1
                self.root.after(0, lambda c=count, t=total_evals: self.sobol_progress.config(value=int(100*c/t)))
            YB = np.array(YB)

            YAB_list = []
            for i in range(k):
                M = A.copy()
                M[:, i] = B[:, i]
                YAB = []
                for n in range(N_base):
                    YAB.append(eval_row(M[n, :]))
                    count += 1
                    self.root.after(0, lambda c=count, t=total_evals: self.sobol_progress.config(value=int(100*c/t)))
                YAB_list.append(np.array(YAB))

            Y_all = np.concatenate([YA, YB])
            V = np.var(Y_all, ddof=1) if len(Y_all) > 1 else 1.0

            S = np.zeros(k)
            ST = np.zeros(k)
            for i in range(k):
                YAB = YAB_list[i]
                S[i] = np.mean(YB * (YAB - YA)) / V if V > 0 else 0.0
                ST[i] = 0.5 * np.mean((YA - YAB) ** 2) / V if V > 0 else 0.0

            self.root.after(0, self.sobol_complete, names, S, ST, core, N_base)
        except Exception as e:
            self.root.after(0, self.sobol_error, str(e))

    def sobol_complete(self, names, S, ST, core, N_base):
        """Sobol complete"""
        self.is_running = False
        self.sobol_status.config(text="Complete ✓", foreground="green")
        self.sobol_progress.config(value=100)

        # Plot
        self.sobol_fig.clear()
        ax = self.sobol_fig.add_subplot(111)

        order = np.argsort(-ST)
        names_sorted = [names[i] for i in order]
        S_sorted = S[order]
        ST_sorted = ST[order]

        x = np.arange(len(names_sorted))
        ax.bar(x - 0.18, S_sorted, 0.36, label="S_i (First-order)")
        ax.bar(x + 0.18, ST_sorted, 0.36, label="S_Ti (Total)")

        ax.set_xticks(x)
        ax.set_xticklabels(names_sorted, rotation=30, ha="right")
        ax.set_ylabel("Sobol Index")
        ax.set_title(f"Sobol Indices (core={core}, N={N_base})")
        ax.legend()
        ax.grid(True, axis='y', ls='--', alpha=0.3)

        self.sobol_fig.tight_layout()
        self.sobol_canvas.draw()

        self.sobol_results = {"names": names_sorted, "S": S_sorted, "ST": ST_sorted}

    def sobol_error(self, msg):
        """Sobol error"""
        self.is_running = False
        self.sobol_status.config(text="Failed ✗", foreground="red")
        messagebox.showerror("Error", f"Sobol failed:\n{msg}")

    def update_sobol_from_baseline(self):
        """Update Sobol ranges based on current baseline values in FULL tab"""
        vals = self.get_values()
        if vals is None:
            return

        try:
            pct = float(self.sobol_pct.get()) / 100.0
        except ValueError:
            messagebox.showerror("Error", "Invalid uncertainty percentage")
            return

        # Update each parameter's low/high based on current baseline ± pct
        for param, widgets in self.sobol_vars.items():
            if param in vals:
                baseline = float(vals[param])
                low = baseline * (1 - pct)
                high = baseline * (1 + pct)

                widgets["low"].delete(0, tk.END)
                widgets["low"].insert(0, f"{low:.4g}")

                widgets["high"].delete(0, tk.END)
                widgets["high"].insert(0, f"{high:.4g}")

        messagebox.showinfo("Success", f"Updated Sobol ranges with ±{pct*100:.1f}% uncertainty from baseline")

    # ========== EXPORT FUNCTIONS ==========
    def export_plot(self):
        """Export FULL plot"""
        if self.current_results is None or not self.current_results["mode_available"]:
            messagebox.showwarning("Warning", "No plot available")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".png",
                                               filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if filename:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Saved to {filename}")

    def export_results(self):
        """Export FULL results"""
        if self.current_results is None:
            messagebox.showwarning("Warning", "No results")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if filename:
            with open(filename, 'w') as f:
                f.write(self.results_text.get(1.0, tk.END))
            messagebox.showinfo("Success", f"Saved to {filename}")

    def export_sens_plot(self):
        """Export SENS plot"""
        if not hasattr(self, 'sens_results'):
            messagebox.showwarning("Warning", "No SENS results")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".png",
                                               filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if filename:
            self.sens_fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Saved to {filename}")

    def export_sobol_plot(self):
        """Export Sobol plot"""
        if not hasattr(self, 'sobol_results'):
            messagebox.showwarning("Warning", "No Sobol results")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".png",
                                               filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if filename:
            self.sobol_fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Saved to {filename}")

    def load_config(self):
        """Load config"""
        filename = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                for key, value in config.items():
                    if key in self.entries:
                        self.entries[key].delete(0, tk.END)
                        self.entries[key].insert(0, str(value))
                    elif key == "core":
                        self.core_var.set(value)
                messagebox.showinfo("Success", "Config loaded")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def save_config(self):
        """Save config"""
        vals = self.get_values()
        if vals is None:
            return
        vals["core"] = self.core_var.get()
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(vals, f, indent=2)
                messagebox.showinfo("Success", "Config saved")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def load_defaults(self):
        """Load defaults"""
        pass


def main():
    root = tk.Tk()
    app = BucklingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
